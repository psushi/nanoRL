"""Time rsl_rl's PPO.update against the captured update on the same rollout (needs CUDA)."""

import argparse
import time
from dataclasses import asdict

import torch
import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from nanorl.ppo import CapturedUpdate


def timed(fn, repeats):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?", default="Mjlab-Lift-Cube-Yam")
    parser.add_argument("--worlds", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    configure_torch_backends()
    cfg, agent = load_env_cfg(args.task), load_rl_cfg(args.task)
    cfg.scene.num_envs = args.worlds
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
    alg = MjlabOnPolicyRunner(env, asdict(agent), log_dir=None, device="cuda:0").alg
    alg.train_mode()
    with torch.inference_mode():
        obs = env.get_observations()
        for _ in range(agent.num_steps_per_env):
            actions = alg.act(obs)
            obs, rewards, dones, extras = env.step(actions)
            alg.process_env_step(obs, rewards, dones, extras)
    last = obs.clone()
    steps = agent.num_steps_per_env

    def stock():
        alg.storage.step = steps
        alg.compute_returns(last)
        alg.update()
    stock()  # Warm up.
    stock_s = timed(stock, args.repeats)
    update = CapturedUpdate(alg, last)
    start = time.perf_counter()
    alg.storage.step = steps
    update()
    setup_s = time.perf_counter() - start

    def captured():
        alg.storage.step = steps
        update()
    captured_s = timed(captured, args.repeats)
    print(f"{args.task}, {args.worlds} worlds, {alg.num_learning_epochs} epochs x {alg.num_mini_batches} minibatches")
    print(f"  rsl_rl PPO.update (with returns): {1000 * stock_s:7.1f} ms")
    print(f"  captured update   (with returns): {1000 * captured_s:7.1f} ms  ({stock_s / captured_s:.2f}x)  "
          f"| capture setup {setup_s:.1f} s")


if __name__ == "__main__":
    main()
