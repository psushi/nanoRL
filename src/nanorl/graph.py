"""Capture a stock mjlab environment step into one CUDA graph.

    env = nanorl.capture(ManagerBasedRlEnv(cfg, device="cuda:0"))

The result is a drop-in RslRlVecEnvWrapper. The step order matches
ManagerBasedRlEnv.step. Where stock code selects worlds by index, nanorl runs
the stock code for every world and keeps the writes of the selected ones.
On CPU the same step runs eagerly, which checks that transformation without
a GPU; only graph capture and constant hoisting are CUDA-specific.
"""

from contextlib import ExitStack, contextmanager

import torch
import warp as wp
from tensordict import TensorDict

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sim.sim_data import TorchArray, WarpBridge
from nanorl import compat
from nanorl.arena import ScratchArena
from nanorl.hoist import HoistConstants
from nanorl.primitives import cond, when
from nanorl.state import WorldState, changing_values, python_values


def _bind_stream(bridge, stream):
    for value in bridge._wrapped_cache.values():
        if isinstance(value, TorchArray):
            value._torch_stream = stream
        elif isinstance(value, WarpBridge):
            _bind_stream(value, stream)


def _hint(code):
    """Most common fix for a line that syncs with the host."""
    if ".item()" in code or ".tolist()" in code or "float(" in code or "int(" in code:
        return "keep the value as a tensor; log it outside the step or combine with torch.where"
    if ".nonzero(" in code or "torch.where(" in code and "[0]" in code:
        return "selects worlds by index; compute for all worlds and pick with torch.where(mask, new, old)"
    if code.startswith(("if ", "elif ", "while ", "assert ")):
        return "Python branch on a GPU value; replace with torch.where"
    if "torch.tensor(" in code:
        return "builds a tensor from Python data every step; create it once at setup"
    if "[" in code and "]" in code:
        return "boolean-mask or list indexing; use torch.where, or index with a tensor made at setup"
    return "reads a GPU value on the host; keep it on the GPU"


class CapturedEnv(RslRlVecEnvWrapper):
    """The env's `_nanorl` runtime: `world`, `ids`, `step_count` and `cond` back the primitives."""

    def __init__(self, env, clip_actions=None):
        super().__init__(env, clip_actions)
        base = self.unwrapped
        self.cuda = self.device.type == "cuda"
        if not base.cfg.auto_reset or base.cfg.recorders:
            raise ValueError("nanorl.capture supports auto-reset envs without recorders")
        for term in base.event_manager._mode_term_cfgs.get("reset", []):
            if term.min_step_count_between_reset:
                raise ValueError("min_step_count_between_reset is not supported yet")
        for term in base.event_manager._mode_term_cfgs.get("interval", []):
            if term.is_global_time:
                raise ValueError("global-time interval events are not supported yet")
        self.stream = torch.cuda.Stream(device=self.device) if self.cuda else None
        with self._scope():
            if self.cuda:
                _bind_stream(base.sim.data, self.stream)
                _bind_stream(base.sim.model, self.stream)
            # The physics kernels go into our graph instead of mjlab's own graphs.
            base.sim.use_cuda_graph = False
            n = base.num_envs
            self.ids = torch.arange(n, device=self.device)
            self.actions = torch.zeros(base.action_space.shape, device=self.device)
            self.done = torch.zeros(n, dtype=torch.bool, device=self.device)
            self.command_dt = torch.zeros(n, device=self.device)
            self.step_count = torch.tensor([base.common_step_counter], device=self.device)
            self._conds = {}
            base._nanorl = self
            compat.install(base)
            self.world = WorldState(base)
            self._reset_all()
            # Rediscover: the first reset creates some buffers lazily.
            self.world = WorldState(base)
            rng = torch.get_rng_state(), torch.cuda.get_rng_state(self.device) if self.cuda else None
            self._warmup()
            if self.cuda:
                self._capture()
            torch.set_rng_state(rng[0])
            if self.cuda:
                torch.cuda.set_rng_state(rng[1], self.device)
            # Warmup advanced the task; start training from a fresh reset.
            self._reset_all()
        self._expected_counter = base.common_step_counter

    def _warmup(self):
        """Two eager steps: create lazy buffers, then report what capture cannot handle."""
        base = self.unwrapped
        self.hoist = HoistConstants(self.device) if self.cuda else None
        with self.hoist if self.cuda else ExitStack():
            self._step()  # Creates lazy buffers and GPU copies of Python constants.
            first = python_values(base)
            if self.cuda:
                self._check_syncs()
            else:
                self._step()
        changed = changing_values(first, python_values(base))
        if changed:
            print("[nanorl] warning: these Python values change every step; a CUDA graph freezes "
                  "them at capture. Keep them in tensors (see nanorl.step / nanorl.when):\n  "
                  + "\n  ".join(changed), flush=True)

    def _capture(self):
        self.warp_stream = wp.get_stream(self.unwrapped.sim.wp_device)
        self.world = WorldState(self.unwrapped)
        self.stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        self.hoist.frozen = True
        with self.hoist, torch.cuda.graph(self.graph, stream=self.stream):
            with wp.ScopedCapture(stream=self.warp_stream, external=True,
                                  capture_mode=wp.CaptureMode.GLOBAL) as warp_capture:
                self.outputs = self._step()
        self._warp_capture = warp_capture
        self.stream.synchronize()

    def _check_syncs(self):
        """Warm up lazy allocations and list every line that syncs with the host."""
        import linecache
        import warnings
        sites = {}
        torch.cuda.set_sync_debug_mode("warn")
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self._step()
                for w in caught:
                    if "synchroniz" in str(w.message):
                        sites.setdefault((w.filename, w.lineno), None)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        if sites:
            lines = []
            for filename, lineno in sites:
                code = linecache.getline(filename, lineno).strip()
                lines.append(f"  {filename}:{lineno}\n      {code}\n      -> {_hint(code)}")
            raise RuntimeError("nanorl.capture: these lines wait for GPU results on the host, "
                               "which a CUDA graph cannot do:\n" + "\n".join(lines) +
                               "\nSee the capture rules in README.md.")

    @contextmanager
    def _scope(self):
        """Inference mode, plus our own CUDA stream ordered after the caller's."""
        with ExitStack() as stack:
            stack.enter_context(torch.inference_mode())
            if self.cuda:
                current = torch.cuda.current_stream(self.device)
                self.stream.wait_stream(current)
                stack.enter_context(torch.cuda.stream(self.stream))
                stack.enter_context(wp.ScopedStream(wp.stream_from_torch(self.stream)))
            yield
        if self.cuda:
            current.wait_stream(self.stream)

    # Captured MDP step: same order as ManagerBasedRlEnv.step.

    def _step(self):
        env = self.unwrapped
        env.extras["log"] = {}
        env.action_manager.process_action(self.actions)
        for _ in range(env.cfg.decimation):
            env.action_manager.apply_action()
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(dt=env.physics_dt)
            env.metrics_manager.compute_substep()
        env.episode_length_buf += 1
        self.step_count += 1
        self.done.copy_(env.termination_manager.compute())
        reward = env.reward_manager.compute(dt=env.step_dt)
        env.metrics_manager.compute()
        if "step" in env.event_manager.available_modes:
            env.event_manager.apply(mode="step", dt=env.step_dt)
        self._interval_events()
        self.step_logs = dict(env.extras["log"])
        self._snapshot_episode()
        cached = self._snapshot_sensor_caches()
        self.command_dt.copy_(torch.where(self.done, 0.0, env.step_dt))
        obs = self._reset_and_observe(self.done, cached)
        return obs, reward, env.termination_manager.terminated, env.termination_manager.time_outs

    def _reset_and_observe(self, mask, cached=None):
        """Shared tail of step and reset: reset selected worlds, refresh, observe."""
        env = self.unwrapped
        # Stock reset path, run for every world; only selected worlds keep it.
        when(env, mask, env._reset_idx)
        env.scene.write_data_to_sim()
        env.sim.forward()
        self._advance_commands(self.command_dt)
        if cached is not None:
            # Before sense(): sensors that sense() refreshes recompute afterwards as in stock.
            self._restore_sensor_caches(cached, mask.any())
        env.sim.sense()
        obs = env.observation_manager.compute(update_history=True)
        self.world.repin()
        return obs

    def _interval_events(self):
        env = self.unwrapped
        manager = env.event_manager
        for timer, term in zip(manager._interval_term_time_left,
                               manager._mode_term_cfgs.get("interval", []), strict=True):
            timer -= env.step_dt
            expired = timer < 1e-6
            low, high = term.interval_range_s
            timer.copy_(torch.where(expired, torch.rand_like(timer) * (high - low) + low, timer))
            when(env, expired, lambda ids, term=term: term.func(env, ids, **term.params))

    def _advance_commands(self, dt):
        env = self.unwrapped
        for term in env.command_manager._terms.values():
            term._update_metrics()
            term.time_left -= dt
            expired = term.time_left <= 0.0
            when(env, expired, term._resample)
            if hasattr(term, "_pending_forward"):
                # Commands that move sim state on resample refresh physics only when one did.
                cond(env, expired.any(), env.sim.forward)
                term._pending_forward = False
            term._update_command(None)

    def cond(self, pred, fn):
        """Run fn only when the GPU bool pred is true (a conditional graph node)."""
        if not self.cuda:
            if bool(pred):
                fn()
            return
        device = str(self.device)  # Warp takes device names, not torch.device
        key = (getattr(fn, "__func__", fn), id(getattr(fn, "__self__", None)))
        if key not in self._conds:
            # Conditional bodies cannot allocate; record fn's Warp scratch once.
            flag = torch.zeros(1, dtype=torch.int32, device=self.device)
            arena = ScratchArena(device)
            wp.set_device_allocator(device, arena)
            try:
                fn()
            finally:
                wp.set_device_allocator(device, None)
            self._conds[key] = (flag, wp.from_torch(flag, dtype=wp.int32), arena)
        flag, flag_wp, arena = self._conds[key]
        flag.copy_(pred.reshape(1).to(torch.int32))

        def body():
            arena.cursor = 0
            wp.set_device_allocator(device, arena)
            try:
                fn()
                assert arena.cursor == len(arena.blocks), "conditional body allocation schedule changed"
            finally:
                wp.set_device_allocator(device, None)
        wp.capture_if(flag_wp, on_true=body)

    # Stock observations reuse reward-time sensor data unless any world reset.

    def _snapshot_sensor_caches(self):
        cached = {}
        for sensor in self.unwrapped.scene._sensors.values():
            if getattr(sensor, "_cache_valid", False) and sensor._cached_data is not None:
                cached[sensor] = {k: v.clone() for k, v in vars(sensor._cached_data).items()
                                  if isinstance(v, torch.Tensor)}
        return cached

    def _restore_sensor_caches(self, cached, any_reset):
        for sensor, old in cached.items():
            data = sensor.data
            for name, value in old.items():
                setattr(data, name, torch.where(any_reset, getattr(data, name), value))

    # Logging: stock logs reset-time episode statistics for finished worlds.

    def _snapshot_episode(self):
        env = self.unwrapped
        self.episode = {}
        for k, v in env.reward_manager._episode_sums.items():
            self.episode[f"Episode_Reward/{k}"] = v / env.max_episode_length_s
        metrics = env.metrics_manager
        if hasattr(metrics, "_term_cfgs"):
            counts = metrics._step_count.float().clamp(min=1.0)
            for idx, key in enumerate(metrics._episode_sums):
                reduce = metrics._term_cfgs[idx].reduce
                if reduce == "max":
                    value = metrics._episode_max[key]
                elif reduce == "last":
                    value = metrics._step_values[:, idx]
                elif reduce == "sum":
                    value = metrics._episode_sums[key]
                else:
                    value = metrics._episode_sums[key] / counts
                self.episode[f"Episode_Metrics/{key}"] = value.clone()
        for name, term in env.command_manager._terms.items():
            for k, v in term.metrics.items():
                self.episode[f"Metrics/{name}/{k}"] = v.clone()
        for k, v in env.termination_manager._term_dones.items():
            self.episode[f"Episode_Termination/{k}"] = v.float()

    def _logs(self):
        env = self.unwrapped
        logs = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in self.step_logs.items()}
        ids = self.done.nonzero().flatten()
        if ids.numel():
            for k, v in self.episode.items():
                logs[k] = v[ids].sum() if k.startswith("Episode_Termination/") else v[ids].mean()
            for name, state in env.curriculum_manager._curriculum_state.items():
                if isinstance(state, dict):
                    logs.update({f"Curriculum/{name}/{k}": v.clone() for k, v in state.items()})
                elif state is not None:
                    logs[f"Curriculum/{name}"] = state
        return logs

    # RSL-RL interface.

    def _reset_all(self):
        self.unwrapped.extras["log"] = {}
        self.command_dt.zero_()
        return self._reset_and_observe(torch.ones_like(self.done))

    @property
    def episode_length_buf(self):
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        # RSL randomizes starting lengths by assignment; keep the graph's buffer.
        self.unwrapped.episode_length_buf.copy_(value)

    def _publish(self, obs):
        # Graph outputs are overwritten by the next replay; RSL keeps the previous ones.
        return TensorDict({k: v.clone() for k, v in obs.items()}, batch_size=[self.num_envs])

    def get_observations(self):
        with self._scope():
            return self._publish(self.unwrapped.observation_manager.compute())

    def step(self, actions):
        env = self.unwrapped
        with self._scope():
            if self.clip_actions is not None:
                actions = actions.clamp(-self.clip_actions, self.clip_actions)
            if env.common_step_counter != self._expected_counter:
                # Checkpoint loading restores the Python counter.
                self.step_count.fill_(env.common_step_counter)
            self.actions.copy_(actions)
            if self.cuda:
                self.graph.replay()
            else:
                self.outputs = self._step()
            env.common_step_counter += 1
            env._sim_step_counter += env.cfg.decimation
            self._expected_counter = env.common_step_counter
            obs, reward, terminated, time_outs = self.outputs
            obs = self._publish(obs)
            dones = (terminated | time_outs).long()
            extras = {"log": self._logs()}
            if not self.cfg.is_finite_horizon:
                extras["time_outs"] = time_outs.clone()
            reward = reward.clone()
        return obs, reward, dones, extras

    def reset(self):
        with self._scope():
            return self._publish(self._reset_all()), {}

    def close(self):
        if self.cuda:
            self.stream.synchronize()
        return super().close()
