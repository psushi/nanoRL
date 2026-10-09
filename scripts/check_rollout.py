"""Captured rollout vs RSL-RL's rollout loop from the same state (needs CUDA).

The reference is the stock loop (act, env.step, process_env_step) over the
captured env. Actions use the policy mean in both paths and the env is kept
free of timeouts, command expiry and pushes, so both see identical inputs.
"""

import argparse
from dataclasses import asdict

import torch
import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
import nanorl
from nanorl.check import quiet, restore, snapshot
import nanorl.ppo as ppo_module
import nanorl.runner as runner_module
from nanorl.runner import CapturedRunner
from nanorl.tasks import pop_imports

FIELDS = ("actions", "rewards", "dones", "values", "actions_log_prob")


def storage_state(alg):
    st = alg.storage
    out = {k: getattr(st, k).clone() for k in FIELDS}
    out.update({f"obs.{k}": v.clone() for k, v in st.observations.items()})
    out.update({f"dist.{i}": p.clone() for i, p in enumerate(st.distribution_params)})
    out.update({f"norm.{n}": b.clone() for n, b in alg.actor.named_buffers()})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?", default="Mjlab-Lift-Cube-Yam")
    parser.add_argument("--worlds", type=int, default=64)
    args = parser.parse_args(pop_imports(__import__("sys").argv[1:]))
    configure_torch_backends(allow_tf32=False)
    cfg, agent = load_env_cfg(args.task), load_rl_cfg(args.task)
    cfg.scene.num_envs, cfg.seed = args.worlds, 0
    # Observation noise draws differ between paths; compare noise-free observations.
    for group in cfg.observations.values():
        group.enable_corruption = False
    env = nanorl.capture(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
    base = env.unwrapped
    runner = CapturedRunner(env, asdict(agent), log_dir=None, device="cuda:0")
    alg = runner.alg
    alg.train_mode()
    # Deterministic actions in both paths (the policy mean). Set before _build: the
    # graph records whatever sampling is in place when it is captured.
    runner_module.capture_safe_sampling = ppo_module.capture_safe_sampling = lambda model: None
    alg.actor.distribution.sample = lambda: alg.actor.distribution._distribution.loc.clone()
    runner._build(env.get_observations())

    with torch.inference_mode():
        quiet(base)
        saved_env = snapshot(base)
        saved_norm = {n: b.clone() for n, b in alg.actor.named_buffers()}
        saved_norm_critic = {n: b.clone() for n, b in alg.critic.named_buffers()}

        def reset_to_saved():
            restore(base, saved_env, env)
            for model, saved in ((alg.actor, saved_norm), (alg.critic, saved_norm_critic)):
                for n, b in model.named_buffers():
                    b.copy_(saved[n])
            alg.storage.clear()
            # get_observations returns mjlab's cached buffer; recompute it from the restored state.
            base.observation_manager._obs_buffer = None

        def reference_rollout():
            reset_to_saved()
            obs = env.get_observations()
            for _ in range(agent.num_steps_per_env):  # The stock RSL-RL rollout loop.
                actions = alg.act(obs)
                obs, rewards, dones, extras = env.step(actions)
                alg.process_env_step(obs, rewards, dones, extras)
            result = storage_state(alg), obs.clone()
            # The eager loop replaced the normalizer's _std; point it back at the graph's buffer.
            runner._repin_policy()
            return result

        # Run the reference twice: GPU contact forces vary run to run, and the closed
        # loop (observation -> action -> contact) amplifies that over the rollout.
        reference, reference_obs = reference_rollout()
        repeat, repeat_obs = reference_rollout()

        reset_to_saved()
        start = env.get_observations()
        for k in runner.obs.keys():
            runner.obs[k].copy_(start[k])
        runner._collect()
        captured, captured_obs = storage_state(alg), runner.obs.clone()

    for t in range(agent.num_steps_per_env):
        row = {k: float((reference[k][t].float() - captured[k][t].float()).abs().max())
               for k in ("obs.actor", "actions", "rewards", "values", "dones")}
        if max(row.values()) > 1e-4 or t < 2:
            print(f"  step {t:2d}: " + ", ".join(f"{k} {v:.2g}" for k, v in row.items()))
    def differences(a, a_obs, b, b_obs):
        out = {k: float((a[k].float() - b[k].float()).abs().max()) for k in a}
        out["next_obs"] = max(float((a_obs[k] - b_obs[k]).abs().max()) for k in a_obs.keys())
        return out
    worst = differences(reference, reference_obs, captured, captured_obs)
    spread = differences(reference, reference_obs, repeat, repeat_obs)
    print(f"{args.task}, {args.worlds} worlds, {agent.num_steps_per_env} steps (captured | reference vs itself)")
    for k, v in sorted(worst.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {k:28s} max |difference| {v:.3g} | {spread[k]:.3g}")
    failed = [k for k in worst if worst[k] > max(1e-3, 2 * spread[k])]
    assert not failed, f"captured rollout differs from the RSL-RL loop beyond its own spread: {failed}"
    print("  passed: storage, normalizer and next observation match the RSL-RL rollout loop")
    env.close()


if __name__ == "__main__":
    main()
