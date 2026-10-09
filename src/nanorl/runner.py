"""mjlab training with the whole PPO iteration captured: rollout graph, then update graph.

    runner = CapturedRunner(nanorl.capture(env), agent_cfg, log_dir, device="cuda:0")
    runner.learn(num_iterations)

Same config, checkpoints, export and logging as MjlabOnPolicyRunner. Per
iteration, Python launches two graphs and reads one batch of statistics:

rollout graph: num_steps_per_env x (policy act -> env step -> store), recorded
    unrolled, so storage writes land at fixed slots. Episode statistics go to
    GPU buffers instead of RSL-RL's per-step `.nonzero()` and `.tolist()`.
update graph: nanorl.ppo.CapturedUpdate (returns, advantages, all minibatches).

Supported as in CapturedUpdate: feedforward policies, one GPU, no RND or
symmetry. The env must come from nanorl.capture on CUDA.
"""

import os
import time

import torch
import warp as wp
from tensordict import TensorDict

from mjlab.rl import MjlabOnPolicyRunner
from nanorl.graph import CapturedEnv
from nanorl.ppo import CapturedUpdate, capture_safe_sampling


def with_captured_learning(runner_cls):
    """Combine CapturedRunner's learn() with a task's runner (its save/export/init)."""
    if runner_cls is None or runner_cls is MjlabOnPolicyRunner:
        return CapturedRunner
    if not issubclass(runner_cls, MjlabOnPolicyRunner):
        raise ValueError(f"{runner_cls.__name__} does not derive from MjlabOnPolicyRunner")
    for cls in runner_cls.__mro__[:runner_cls.__mro__.index(MjlabOnPolicyRunner)]:
        if "learn" in cls.__dict__:
            raise ValueError(f"{cls.__name__} overrides learn(); cannot capture its loop")
    return type(f"Captured{runner_cls.__name__}", (CapturedRunner, runner_cls), {})


class CapturedRunner(MjlabOnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        if not isinstance(env, CapturedEnv) or not env.cuda:
            raise ValueError("CapturedRunner needs an env from nanorl.capture on a CUDA device")
        if self.is_distributed:
            raise ValueError("multi-GPU training is not supported yet")
        self.rollout_graph = None

    # Captured rollout body, recorded once.

    def _rollout(self):
        env, alg = self.env, self.alg
        obs = self.obs
        for t in range(self.steps):
            actions = alg.act(obs)
            if env.clip_actions is not None:
                actions = actions.clamp(-env.clip_actions, env.clip_actions)
            env.actions.copy_(actions)
            observations, rewards, terminated, time_outs = env._step()
            dones = (terminated | time_outs).long()
            next_obs = TensorDict(dict(observations), batch_size=[env.num_envs])
            extras = {} if env.cfg.is_finite_horizon else {"time_outs": time_outs}
            alg.process_env_step(next_obs, rewards, dones, extras)
            self._record(t, rewards, dones.bool())
            obs = next_obs
        for key in self.obs.keys():
            self.obs[key].copy_(obs[key])
        self._repin_policy()

    def _record(self, t, rewards, done):
        """Per-step statistics RSL-RL's logger would compute on the host."""
        env, history = self.env, self.history
        self.episode_return += rewards
        self.episode_length += 1
        history["return"][t].copy_(self.episode_return)
        history["length"][t].copy_(self.episode_length)
        history["done"][t].copy_(done)
        self.episode_return.masked_fill_(done, 0.0)
        self.episode_length.masked_fill_(done, 0.0)
        mask = done.float()
        sums = {f"step:{k}": v for k, v in env.step_logs.items() if isinstance(v, torch.Tensor)}
        # Stock logs per-episode means (terminations: counts) over the worlds that finished.
        sums.update({f"episode:{k}": (v * mask).sum() for k, v in env.episode.items()})
        for name, state in env.unwrapped.curriculum_manager._curriculum_state.items():
            if isinstance(state, dict):
                sums.update({f"curriculum:Curriculum/{name}/{k}": v for k, v in state.items()})
        sums["count"] = mask.sum()
        for key, value in sums.items():
            if key not in history:
                history[key] = torch.zeros(self.steps, device=self.device)
            history[key][t].copy_(value.float().reshape(()))

    def _repin_policy(self):
        # The observation normalizer reassigns its _std buffer every update; keep the
        # graph reading and writing the original allocation.
        for module, name, pinned in self.policy_buffers:
            current = module._buffers[name]
            if current is not pinned:
                pinned.copy_(current)
                module._buffers[name] = pinned

    # Setup.

    def _build(self, obs):
        env, alg = self.env, self.alg
        capture_safe_sampling(alg.actor)
        self.steps = self.cfg["num_steps_per_env"]
        self.obs = TensorDict({k: v.clone() for k, v in obs.items()}, batch_size=obs.batch_size)
        self.episode_return = torch.zeros(env.num_envs, device=self.device)
        self.episode_length = torch.zeros(env.num_envs, device=self.device)
        self.history = {"return": torch.zeros(self.steps, env.num_envs, device=self.device),
                        "length": torch.zeros(self.steps, env.num_envs, device=self.device),
                        "done": torch.zeros(self.steps, env.num_envs, dtype=torch.bool, device=self.device)}
        self.policy_buffers = [(m, n, b) for model in (alg.actor, alg.critic)
                               for m in model.modules() for n, b in m._buffers.items() if b is not None]
        with env._scope(), env.hoist:
            # Eager pass: allocates storage fields, history buffers and lazy policy state.
            self._rollout()
            alg.storage.clear()
            env.stream.synchronize()
            self.rollout_graph = torch.cuda.CUDAGraph()
            # Warp tracks captures by its Stream object; use the one this scope made current.
            warp_stream = wp.get_stream(env.unwrapped.sim.wp_device)
            with torch.cuda.graph(self.rollout_graph, stream=env.stream):
                with wp.ScopedCapture(stream=warp_stream, external=True,
                                      capture_mode=wp.CaptureMode.GLOBAL) as warp_capture:
                    self._rollout()
            self._warp_capture = warp_capture
            env.stream.synchronize()
        alg.storage.clear()
        self.update_graph = CapturedUpdate(alg, self.obs)

    # One iteration.

    def _collect(self):
        env, base = self.env, self.env.unwrapped
        with env._scope():
            if base.common_step_counter != env._expected_counter:
                env.step_count.fill_(base.common_step_counter)  # A checkpoint restored the counter.
            self.rollout_graph.replay()
        base.common_step_counter += self.steps
        base._sim_step_counter += self.steps * base.cfg.decimation
        env._expected_counter = base.common_step_counter
        self.alg.storage.step = self.steps
        if self.cfg.get("check_for_nan", True):
            storage = self.alg.storage
            if torch.isnan(storage.rewards).any() or any(torch.isnan(v).any() for v in storage.observations.values()):
                raise ValueError("NaN in observations or rewards during the captured rollout")
        if self.logger.writer is not None:
            self._flush_logs()

    def _flush_logs(self):
        history, logger = self.history, self.logger
        done = history["done"]
        logger.rewbuffer.extend(history["return"][done].tolist())
        logger.lenbuffer.extend(history["length"][done].tolist())
        keys = [k for k in history if k not in ("return", "length", "done")]
        values = torch.stack([history[k] for k in keys]).cpu()  # One transfer per iteration.
        rows = dict(zip(keys, values))
        for t in range(self.steps):
            entry = {k.split(":", 1)[1]: rows[k][t] for k in keys if k.startswith("step:")}
            count = float(rows["count"][t])
            if count > 0:
                for key in keys:
                    kind, name = key.split(":", 1) if ":" in key else (key, key)
                    if kind == "episode":
                        total = rows[key][t]
                        entry[name] = total if name.startswith("Episode_Termination/") else total / count
                    elif kind == "curriculum":
                        entry[name] = rows[key][t]
            logger.ep_extras.append(entry)

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        self.logger.init_logging_writer()
        if self.rollout_graph is None:
            self._build(obs)
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            self._collect()
            torch.cuda.synchronize()
            collect_time = time.time() - start
            start = time.time()
            loss_dict = self.update_graph()
            learn_time = time.time() - start
            self.current_learning_iteration = it
            self.logger.log(it=it, start_it=start_it, total_it=total_it, collect_time=collect_time,
                            learn_time=learn_time, loss_dict=loss_dict,
                            learning_rate=self.alg.learning_rate,
                            action_std=self.alg.get_policy().output_std, rnd_weight=None)
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()
