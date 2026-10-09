"""Seconds per PPO training iteration: stock, captured env only, everything captured.

Same task, config and world count for each; real PPO (rollout + update) from a
fresh policy. Warmup iterations (and capture) are excluded from timing.
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
import nanorl
from nanorl.runner import CapturedRunner
from nanorl.tasks import pop_imports

SETUPS = {
    "stock": (RslRlVecEnvWrapper, MjlabOnPolicyRunner),
    "captured env": (nanorl.capture, MjlabOnPolicyRunner),
    "captured env + PPO": (nanorl.capture, CapturedRunner),
}


def measure(task, worlds, setup, warmup, iterations, checkpoint=None):
    wrap, runner_cls = SETUPS[setup]
    cfg, agent = load_env_cfg(task), load_rl_cfg(task)
    cfg.scene.num_envs = worlds
    env = wrap(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
    runner = runner_cls(env, asdict(agent), log_dir=None, device="cuda:0")
    if checkpoint:
        # A trained policy grasps, so physics is as contact-heavy as in late training.
        runner.load(checkpoint, map_location="cuda:0")
    phases = {"update": 0.0}
    update = runner.alg.update if runner_cls is not CapturedRunner else None
    if update is not None:
        def timed_update(*a, **k):
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = update(*a, **k)
            phases["update"] += time.perf_counter() - t
            return out
        runner.alg.update = timed_update
    start = time.perf_counter()
    runner.learn(num_learning_iterations=warmup)
    torch.cuda.synchronize()
    setup_s = time.perf_counter() - start
    phases["update"] = 0.0
    if update is None:
        graph = runner.update_graph
        call = graph.__call__

        def timed_captured():
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = call()
            phases["update"] += time.perf_counter() - t
            return out
        runner.update_graph = timed_captured
    start = time.perf_counter()
    runner.learn(num_learning_iterations=iterations)
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / iterations
    env.close()
    samples = worlds * agent.num_steps_per_env
    return {"setup": setup, "worlds": worlds, "s_per_iteration": seconds,
            "update_s": phases["update"] / iterations, "sps": samples / seconds, "warmup_s": setup_s}


def main():
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?", default="Mjlab-Lift-Cube-Yam")
    parser.add_argument("--worlds", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", help="start from a trained policy (contact-heavy phase)")
    args = parser.parse_args(pop_imports(sys.argv[1:]))
    configure_torch_backends()
    rows = []
    for worlds in args.worlds:
        for setup in SETUPS:
            rows.append(measure(args.task, worlds, setup, args.warmup, args.iterations, args.checkpoint))
            r = rows[-1]
            print(f"{args.task} {worlds:5d} worlds | {setup:20s} | {1000 * r['s_per_iteration']:7.1f} ms/iteration "
                  f"(update {1000 * r['update_s']:5.1f} ms) | {r['sps']:9,.0f} SPS", flush=True)
    output = args.output or Path(f"artifacts/bench-training-{args.task}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"task": args.task, "gpu": torch.cuda.get_device_name(), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
