"""nanorl.capture's core ideas on a toy env, in plain PyTorch (needs CUDA).

    python examples/toy_capture.py
    python examples/toy_capture.py --without repin   # or hoist, masked

ToyEnv is written the way stock mjlab is: Python picks which worlds reset
(`.nonzero()`), physics replaces its state tensor, and resets write Python
scalars. Captured shows the four changes that make the same step one CUDA
graph, then checks it against the stock loop step by step.
"""

import argparse

import torch

N, DEVICE = 8, "cuda"


class ToyEnv:
    def __init__(self):
        self.pos = torch.zeros(N, device=DEVICE)
        self.age = torch.zeros(N, dtype=torch.long, device=DEVICE)

    def physics(self, action):
        self.pos = self.pos + 0.1 * action  # Replaces the tensor instead of writing into it.

    def reset(self, ids):
        self.pos[ids] = 0.0  # A Python scalar: becomes a CPU->GPU copy.
        self.age[ids] = 0

    def stock_step(self, action):
        self.physics(action)
        self.age += 1
        done = self.age >= 5
        ids = done.nonzero().flatten()  # The CPU waits for the GPU here.
        if len(ids):
            self.reset(ids)
        return self.pos.clone(), done


class Captured:
    def __init__(self, env, without=None):
        self.env, self.without = env, without
        # 1. Fixed input buffer: a graph reads the same addresses on every replay.
        self.action = torch.zeros(N, device=DEVICE)
        self.ids = torch.arange(N, device=DEVICE)
        self.pinned_pos = env.pos
        # 2. Hoisted constants: GPU copies made before capture (nanorl/hoist.py).
        self.zero_f = torch.zeros((), device=DEVICE)
        self.zero_l = torch.zeros((), dtype=torch.long, device=DEVICE)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):  # Warm up lazy allocations outside the graph.
                self._step()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.outputs = self._step()  # Records kernels; nothing executes yet.
        env.pos.zero_()  # Warmup advanced the state; start fresh.
        env.age.zero_()

    def _repin(self):
        # 3. Re-pinning: copy a replaced tensor back into the buffer the graph reads.
        if self.without != "repin" and self.env.pos is not self.pinned_pos:
            self.pinned_pos.copy_(self.env.pos)
            self.env.pos = self.pinned_pos

    def _reset(self, ids):
        if self.without == "hoist":
            return self.env.reset(ids)  # Python scalars: CPU->GPU copies inside the graph.
        self.env.pos[ids] = self.zero_f
        self.env.age[ids] = self.zero_l

    def _masked(self, mask, fn):
        # 4. Masked commit: run fn for every world, keep its writes where mask is true.
        self._repin()
        saved = [self.env.pos.clone(), self.env.age.clone()]
        fn(self.ids)
        self._repin()
        for state, old in zip((self.env.pos, self.env.age), saved):
            state.copy_(torch.where(mask, state, old))

    def _step(self):
        env = self.env
        env.physics(self.action)
        env.age += 1
        done = env.age >= 5
        if self.without == "masked":
            ids = done.nonzero().flatten()  # Stock selection: waits for the GPU.
            if len(ids):
                self._reset(ids)
        else:
            self._masked(done, self._reset)  # Was: ids = done.nonzero(); if len(ids): reset(ids)
        self._repin()
        return env.pos, done

    def step(self, action):
        self.action.copy_(action)
        self.graph.replay()  # One launch replays every recorded kernel.
        return self.outputs[0].clone(), self.outputs[1].clone()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--without", choices=("repin", "hoist", "masked"))
    args = parser.parse_args()
    stock, captured = ToyEnv(), Captured(ToyEnv(), args.without)
    actions = torch.randn(20, N, device=DEVICE, generator=torch.Generator(DEVICE).manual_seed(0))
    for i, action in enumerate(actions):
        (a, done_a), (b, done_b) = stock.stock_step(action), captured.step(action)
        assert torch.equal(done_a, done_b) and torch.allclose(a, b), f"diverged at step {i}"
    print("20 steps: captured graph matches the stock loop, including 4 rounds of resets")


if __name__ == "__main__":
    main()
