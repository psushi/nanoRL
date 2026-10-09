# nanoRL

Faster robotics reinforcement learning in [mjlab](https://github.com/mujocolab/mjlab),
without giving up its Python task definitions.

```python
import nanorl
env = nanorl.capture(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))  # instead of RslRlVecEnvWrapper(...)
```

## Why

In mjlab, physics and task data already live on the GPU, but Python drives
every step: it launches thousands of small GPU operations for observations,
rewards, commands and resets, and sometimes waits for GPU results (for
example, to find which worlds need resetting). The GPU sits idle in between.

`nanorl.capture` records one complete environment step, physics and task
logic together, as a CUDA graph. Every later step replays that graph with a
single call. The math is the same as stock mjlab; Python is no longer in the
loop.

## Results

Environment stepping on an RTX 4090, random actions, no learner:

| Task | 1,024 worlds | 4,096 worlds |
|---|---|---|
| Cartpole Balance | 311k → 541k steps/s (**1.74x**) | 1.19M → 1.95M (1.64x) |
| Lift-Cube | 143k → 280k (**1.96x**) | 493k → 738k (1.50x) |
| G1 flat locomotion | 57k → 107k (**1.87x**) | 169k → 248k (1.47x) |
| G1 motion tracking (synthetic motion) | 45k → 94k (**2.08x**) | 141k → 215k (1.53x) |

The gain is a roughly fixed saving per step, so it is largest with heavy task
logic and fewer worlds, and shrinks as physics dominates. In real Lift-Cube
training with the stock RSL-RL learner (4,096 worlds, 3 seeds), training
throughput rose from about 135k to 164k steps/s (1.22x) and time to 85% lift
success fell from 4.0 to 3.2 minutes.

Correctness: on CPU, captured and stock mjlab match exactly on eight tasks
(observations, rewards, joint positions, partial resets, commands, pushes).
On GPU they match exactly or within stock's own run-to-run variation.

## Install

```bash
git clone git@github.com:psushi/nanoRL.git && cd nanoRL
uv venv --python 3.12
# CPU (correctness checks, e.g. on a Mac):
uv pip install -e .
# Linux with an NVIDIA GPU:
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128
uv pip install -e .
```

## Use it

```bash
# Does nanorl match stock mjlab for this task? (exact on CPU, no GPU needed)
python -m nanorl check Mjlab-Lift-Cube-Yam --device cpu
# Capture on GPU and compare again
python -m nanorl check Mjlab-Lift-Cube-Yam
# Steps per second, stock vs captured
python -m nanorl bench Mjlab-Lift-Cube-Yam --worlds 1024 4096
# Train with mjlab's normal launcher and flags
python -m nanorl train Mjlab-Lift-Cube-Yam --env.scene.num-envs 4096
```

Add `--import my_tasks` to use tasks registered in your own module. Set
`MUJOCO_GL=disable` on machines without a display.

To see the core ideas in about 100 lines of plain PyTorch, run
`python examples/toy_capture.py` on a GPU, then again with
`--without repin`, `--without hoist` or `--without masked` to watch each
mechanism fail.

## How it works

1. **Fixed buffers.** A graph reuses the memory addresses it recorded, so
   actions go into a fixed input buffer, and tensors that task code replaces
   (`self.x = f(self.x)`) are copied back into their original memory.
2. **Masked resets.** Instead of picking finished worlds on the CPU, reset
   code runs for every world and only finished worlds keep the result.
3. **No CPU decisions.** Branches on GPU values become tensor operations or
   GPU-side conditionals; counters and curricula live in tensors.
4. **GPU constants.** Python numbers and index lists used inside a step are
   turned into GPU tensors before recording.

Built-in mjlab terms that need these changes are fixed once inside the
library, so tested tasks need no changes.

## Writing your own terms

Pure tensor terms (most rewards, observations, terminations) work as is.
Stateful terms (commands, curricula, events) should:

1. not read GPU values in Python (`.item()`, `.nonzero()`, `if tensor:`);
2. keep anything that changes during training in a tensor, not a Python number;
3. expect reset and event code to be called for all worlds.

Three optional helpers make this easy, and also work in stock mjlab:

```python
nanorl.when(env, mask, fn)   # fn(env_ids) only for the worlds in mask
nanorl.cond(env, pred, fn)   # fn() only when a GPU bool is true
nanorl.step(env)             # training step count as a GPU tensor
```

`nanorl check` and `nanorl.capture` report lines that break these rules.

## Status

Tested on Cartpole, Lift-Cube, G1 and Go1 velocity (flat and rough) and G1
motion tracking. Not supported yet: camera tasks, recorders, and a few event
options (rejected with a clear error).

**In progress:** capturing the PPO learner as well (policy inference, rollout
storage and updates; `nanorl.ppo`, `nanorl.runner`, and the parity checks in
`scripts/`). Try it with `nanorl train ... --learner captured`; it is still
being validated, so the default is the stock RSL-RL learner.
