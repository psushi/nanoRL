"""Captured PPO update vs rsl_rl's PPO.update on the same rollout (needs CUDA).

Fills the rollout storage once with real stock-env data, then runs two
consecutive updates both ways from identical weights, optimizer state and
minibatch order, and compares weights, learning rate and losses.
"""

import argparse
from dataclasses import asdict

import torch
import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from nanorl.ppo import CapturedUpdate


def weights(alg):
    return {f"{n}.{k}": v.detach().clone() for n, m in (("actor", alg.actor), ("critic", alg.critic))
            for k, v in m.state_dict().items()}


def load(alg, saved):
    with torch.no_grad():
        for n, m in (("actor", alg.actor), ("critic", alg.critic)):
            for k, v in m.state_dict().items():
                v.copy_(saved[f"{n}.{k}"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?", default="Mjlab-Lift-Cube-Yam")
    parser.add_argument("--worlds", type=int, default=512)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="TF32 matmuls (mjlab's default); --no-tf32 isolates numerics")
    args = parser.parse_args()
    configure_torch_backends(allow_tf32=args.tf32)
    cfg, agent = load_env_cfg(args.task), load_rl_cfg(args.task)
    cfg.scene.num_envs, cfg.seed = args.worlds, 0
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
    runner = MjlabOnPolicyRunner(env, asdict(agent), log_dir=None, device="cuda:0")
    alg = runner.alg
    alg.train_mode()
    with torch.inference_mode():  # The stock rollout loop, unchanged.
        obs = env.get_observations()
        for _ in range(agent.num_steps_per_env):
            actions = alg.act(obs)
            obs, rewards, dones, extras = env.step(actions)
            alg.process_env_step(obs, rewards, dones, extras)
    last_obs = obs.clone()
    initial, lr0 = weights(alg), alg.learning_rate
    stored = {k: getattr(alg.storage, k).clone() for k in ("returns", "advantages")}
    perm = torch.randperm(args.worlds * agent.num_steps_per_env, device="cuda:0")
    stock_randperm = torch.randperm
    torch.randperm = lambda *a, **k: perm.clone()  # Same minibatch order for both paths.
    try:
        stock_losses = []
        for _ in range(2):
            alg.storage.step = agent.num_steps_per_env
            alg.compute_returns(last_obs)
            stock_losses.append(alg.update())
        stock_weights, stock_lr = weights(alg), alg.learning_rate

        load(alg, initial)
        alg.learning_rate = lr0
        alg.optimizer = type(alg.optimizer)([p for g in alg.optimizer.param_groups for p in g["params"]], lr=lr0)
        for k, v in stored.items():
            setattr(alg.storage, k, v.clone())
        update = CapturedUpdate(alg, last_obs)
        captured_losses = []
        for _ in range(2):
            alg.storage.step = agent.num_steps_per_env
            captured_losses.append(update())
        captured_weights = weights(alg)
    finally:
        torch.randperm = stock_randperm

    # Compare each tensor on its own scale (normalizer counts are ~1e4, weights ~1).
    relative = {k: float((stock_weights[k] - captured_weights[k]).abs().max()
                         / stock_weights[k].abs().max().clamp(min=1e-6)) for k in stock_weights}
    worst = max(relative, key=relative.get)
    print(f"{args.task}, {args.worlds} worlds, 2 updates x {alg.num_learning_epochs} epochs x {alg.num_mini_batches} minibatches")
    print(f"  max relative weight difference {relative[worst]:.3g} ({worst})")
    print(f"  learning rate: stock {stock_lr:.6g}, captured {alg.learning_rate:.6g}")
    for i, (a, b) in enumerate(zip(stock_losses, captured_losses)):
        print(f"  update {i}: stock " + ", ".join(f"{k} {a[k]:.6g}" for k in a)
              + " | captured " + ", ".join(f"{k} {b[k]:.6g}" for k in b))
    # Adam divides by running gradient magnitudes, amplifying rounding in small gradients;
    # TF32 matmuls round more. FP32 differences come from capturable Adam's tensor arithmetic.
    assert relative[worst] <= (1e-3 if args.tf32 else 1e-4), "weights differ"
    assert abs(stock_lr - alg.learning_rate) <= 1e-6 * stock_lr, "learning rate differs"
    print("  passed: weights, learning rate and losses match rsl_rl's PPO.update")


if __name__ == "__main__":
    main()
