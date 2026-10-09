"""Check nanorl.capture against stock mjlab on any task.

    python -m nanorl check TASK [--device cpu] [--import MODULE]

On CUDA this checks the captured graph; on CPU the same step runs eagerly,
which checks the dense-with-masked-commit rewrite without a GPU.
"""

import argparse
import sys

import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
import nanorl
from nanorl.state import WorldState
from nanorl.tasks import pop_imports


def snapshot(env):
    return [(slot, slot.get().clone()) for slot in WorldState(env).slots
            if isinstance(slot.get(), torch.Tensor)]


def restore(env, saved, wrapped=None):
    for slot, value in saved:
        slot.get().copy_(value)
    warmstart = wp.to_torch(env.sim.wp_data.qacc_warmstart)
    kept = warmstart.clone()
    env.sim.forward()
    warmstart.copy_(kept)
    env.sim.sense()
    for sensor in env.scene._sensors.values():
        sensor._invalidate_cache()
    if wrapped is not None:
        # Eager stock calls can replace tensors; point them back at the graph's buffers.
        wrapped.world.repin()


def quiet(env):
    # No timeouts, command expiry, or pushes unless a test asks for one.
    env.episode_length_buf.zero_()
    for term in env.command_manager._terms.values():
        term.time_left.fill_(10)
    for timer in env.event_manager._interval_term_time_left:
        timer.fill_(10)


def probe(env):
    """Internal reward state, for locating divergence."""
    out = {}
    if env.scene.env_origins is not None:
        out["env_origins"] = env.scene.env_origins.clone()
    for term in env.reward_manager._term_cfgs:
        if hasattr(term.func, "peak_heights"):
            out["peak_heights"] = term.func.peak_heights.clone()
    for name, sensor in env.scene._sensors.items():
        air = getattr(sensor, "_air_time_state", None)
        if air is not None:
            out.update({f"{name}.{k}": v.clone() for k, v in vars(air).items() if isinstance(v, torch.Tensor)})
    return out


def record(env, obs, reward):
    group = "critic" if "critic" in obs.keys() else next(iter(obs.keys()))
    return (obs[group].clone(), reward.clone(), wp.to_torch(env.sim.wp_data.qpos).clone(),
            env.reward_manager._step_reward.clone(), probe(env))


def stock_step(env, action):
    obs, reward, terminated, timeout, _ = env.step(action)
    return record(env, obs, reward), terminated | timeout


def captured_step(env, wrapped, action):
    obs, reward, dones, extras = wrapped.step(action)
    return record(env, obs, reward), dones.bool(), extras


def worst(a, b, worlds=slice(None)):
    return [max(float((x[i][worlds] - y[i][worlds]).abs().max()) for x, y in zip(a, b))
            for i in range(3)]


def explain(env, stock, captured, group_name):
    names = env.observation_manager.active_terms[group_name]
    dims = [d[0] for d in env.observation_manager._group_obs_term_dim[group_name]]
    for step, (a, b) in enumerate(zip(stock, captured)):
        start, bad = 0, []
        for name, dim in zip(names, dims):
            err = float((a[0][:, start:start + dim] - b[0][:, start:start + dim]).abs().max())
            start += dim
            if err > 1e-4:
                bad.append(f"obs:{name}={err:.2g}")
        for i, name in enumerate(env.reward_manager.active_terms):
            err = float((a[3][:, i] - b[3][:, i]).abs().max())
            if err > 1e-4:
                bad.append(f"rew:{name}={err:.2g}")
        for k in a[4]:
            err = float((a[4][k].float() - b[4][k].float()).abs().max())
            if err > 1e-4:
                bad.append(f"state:{k}={err:.2g}")
        if bad:
            print(f"  step {step}:", ", ".join(bad))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m nanorl check", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task")
    parser.add_argument("--worlds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(pop_imports(sys.argv[1:] if argv is None else argv))
    cfg = load_env_cfg(args.task)
    cfg.scene.num_envs, cfg.seed = args.worlds, 42
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    try:
        with torch.inference_mode():
            env.reset()
            group = "critic" if "critic" in env.observation_manager.active_terms else \
                next(iter(env.observation_manager.active_terms))
            quiet(env)
            saved = snapshot(env)
            shape = env.action_space.shape
            actions = torch.linspace(-0.05, 0.05, args.steps * shape[0] * shape[1],
                                     device=env.device).reshape(args.steps, *shape)
            stock, repeat = [], []
            for run in (stock, repeat):
                restore(env, saved)
                for action in actions:
                    row, done = stock_step(env, action)
                    assert not done.any()
                    run.append(row)
            # One reset step in stock: world 0 times out, nothing else happens.
            restore(env, saved)
            env.episode_length_buf[0] = env.max_episode_length - 1
            stock_reset, done = stock_step(env, actions[0])
            assert done[0] and not done[1:].any()

            wrapped = nanorl.capture(env)
            restore(env, saved, wrapped)
            captured = []
            for action in actions:
                row, done, _ = captured_step(env, wrapped, action)
                assert not done.any()
                captured.append(row)
            spread, error = worst(stock, repeat), worst(stock, captured)
            print(f"{args.task} on {args.device}")
            print("  stock vs stock-repeat max |error| (obs, reward, qpos):", spread)
            print("  stock vs nanorl.capture max |error| (obs, reward, qpos):", error)
            explain(env, stock, captured, group)
            # Contact forces vary run to run on GPU; allow twice stock's own spread.
            limits = [max(1e-4, 2 * e) for e in spread]
            assert all(e <= l for e, l in zip(error, limits)), "parity failed"

            # Masked commit: when only world 0 resets, worlds 1.. must match stock.
            restore(env, saved, wrapped)
            env.episode_length_buf[0] = env.max_episode_length - 1
            row, done, extras = captured_step(env, wrapped, actions[0])
            assert done[0] and not done[1:].any(), "only world 0 resets"
            others = worst([stock_reset], [row], slice(1, None))
            rewards = float((stock_reset[1] - row[1]).abs().max())
            print("  reset step, worlds 1.. max |error| (obs, reward, qpos):", others,
                  "| reward, all worlds:", rewards)
            assert all(e <= l for e, l in zip(others, limits)) and rewards <= limits[1], \
                "a reset changed other worlds"
            if "env_origins" in row[4]:
                # Terrain curriculum moves the reset world; it must move it as stock does.
                moved = float((stock_reset[4]["env_origins"] - row[4]["env_origins"]).abs().max())
                print("  reset step, terrain origins max |error| (all worlds):", moved)
                assert moved <= 1e-6, "terrain curriculum differs from stock"
            assert env.episode_length_buf[0] == 0 and (env.episode_length_buf[1:] == 1).all()
            assert any(k.startswith("Episode_Reward/") for k in extras["log"]), "episode logs"

            # Independent events: world 1's command expires, world 2 is pushed.
            command = next(iter(env.command_manager._terms.values()), None)
            pushes = env.event_manager._interval_term_time_left
            if command is not None or pushes:
                restore(env, saved, wrapped)
                lengths = env.episode_length_buf.clone()
                if command is not None:
                    command.time_left[1] = 0.001
                    counter = command.command_counter.clone()
                if pushes:
                    pushes[0][2] = 0.001
                motion = getattr(command, "motion", None)
                if motion is not None:
                    # World 3 reaches the end of its reference motion this step.
                    command.time_steps[3] = motion.time_step_total - 1
                    frames = command.time_steps.clone()
                _, done, _ = captured_step(env, wrapped, actions[0])
                if motion is not None:
                    assert command.time_steps[3] < motion.time_step_total, "motion wraparound"
                    others = [0, 2, 4]  # World 1 resampled from command expiry.
                    assert (command.time_steps[others] == frames[others] + 1).all(), "other worlds advance"
                assert not done.any() and (env.episode_length_buf == lengths + 1).all()
                # Kinematics must already reflect any state a command wrote (e.g. a teleport).
                xpos = wp.to_torch(env.sim.wp_data.xpos).clone()
                env.sim.forward()
                stale = float((wp.to_torch(env.sim.wp_data.xpos) - xpos).abs().max())
                assert stale <= 1e-5, f"kinematics stale after command resample: {stale}"
                if command is not None:
                    assert command.command_counter[1] == counter[1] + 1, "command expiry"
                    assert (command.command_counter[[0, 2, 3, 4]] == counter[[0, 2, 3, 4]]).all()
                if pushes:
                    low = env.event_manager._mode_term_cfgs["interval"][0].interval_range_s[0]
                    assert pushes[0][2] >= low - 1e-6 and (pushes[0][[0, 1, 3, 4]] < 10).all()
            print(f"  passed: parity, reset isolation, logging"
                  + (", command expiry" if command is not None else "")
                  + (", interval push" if pushes else "")
                  + (", motion wraparound" if getattr(command, "motion", None) is not None else ""), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
