"""Environment steps per second: stock mjlab vs nanorl.capture, random actions, no learner."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
import nanorl
from nanorl.tasks import pop_imports


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def run(name, wrap, task, worlds, warmup, steps, device="cuda:0"):
    cfg = load_env_cfg(task)
    cfg.scene.num_envs, cfg.seed = worlds, 42
    begin = perf_counter()
    env = wrap(ManagerBasedRlEnv(cfg=cfg, device=device))
    sync(device)
    setup_s = perf_counter() - begin
    try:
        generator = torch.Generator(device=device).manual_seed(1)
        actions = 2 * torch.rand((warmup + steps, worlds, env.num_actions), device=device,
                                 generator=generator) - 1
        with torch.inference_mode():
            for action in actions[:warmup]:
                env.step(action)
            sync(device)
            begin = perf_counter()
            for action in actions[warmup:]:
                obs, reward, _, _ = env.step(action)
            sync(device)
            elapsed = perf_counter() - begin
        assert all(torch.isfinite(v).all() for v in obs.values()) and torch.isfinite(reward).all()
        return {"backend": name, "worlds": worlds, "setup_s": setup_s,
                "env_sps": worlds * steps / elapsed, "ms_per_step": 1000 * elapsed / steps}
    finally:
        env.close()


def parse(argv, description=__doc__):
    parser = argparse.ArgumentParser(prog="python -m nanorl bench", description=description)
    parser.add_argument("task")
    parser.add_argument("--worlds", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(pop_imports(argv))


def sweep(args, backends):
    rows = []
    for worlds in args.worlds:
        for name, wrap in backends:
            rows.append(run(name, wrap, args.task, worlds, args.warmup, args.steps, args.device))
            print(json.dumps(rows[-1]), flush=True)
    output = args.output or Path(f"artifacts/benchmarks/bench-{args.task}-{args.device.split(':')[0]}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu"
    output.write_text(json.dumps({"task": args.task, "device": gpu, "results": rows}, indent=2) + "\n")
    return rows


def main(argv=None):
    import sys
    args = parse(sys.argv[1:] if argv is None else argv)
    sweep(args, [("stock", RslRlVecEnvWrapper), ("nanorl.capture", nanorl.capture)])


if __name__ == "__main__":
    main()
