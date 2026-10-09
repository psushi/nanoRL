"""Capture-safe versions of the built-in mjlab terms that break the capture rules.

Each keeps the stock formula and differs only where stock reads GPU values on
the host, selects worlds by index, or keeps changing values in Python. They
apply only to objects that belong to a captured env; stock envs in the same
process keep the stock methods.
"""

import math

import torch
from mjlab.envs.mdp.curriculums import reward_curriculum
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.curriculum_manager import CurriculumManager
from mjlab.managers.reward_manager import RewardManager
from mjlab.managers.termination_manager import TerminationManager
from mjlab.sensor.raycast_sensor import RayCastSensor
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.tasks.velocity.mdp import curriculums
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
from mjlab.utils.lab_api.math import sample_uniform, wrap_to_pi
from nanorl.primitives import cond, step, when

_STOCK = {}


def _captured(obj):
    return getattr(obj, "_nanorl_captured", False) or hasattr(getattr(obj, "_env", None), "_nanorl")


def _override(cls, name, safe):
    """Use `safe` for captured envs' objects and the stock method for everything else."""
    if (cls, name) in _STOCK:
        return
    stock = _STOCK[cls, name] = cls.__dict__[name]

    def method(self, *args, **kwargs):
        return (safe if _captured(self) else stock)(self, *args, **kwargs)
    setattr(cls, name, method)


# Reset logging: stock calls .item(); nanorl logs finished worlds outside the graph.

def _command_reset(self, env_ids):
    extras = {}
    for name, value in self.metrics.items():
        extras[name] = value[env_ids].mean()
        value[env_ids] = 0.0
    self.command_counter[env_ids] = 0
    self._resample(env_ids)
    return extras


def _termination_reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    extras = {f"Episode_Termination/{k}": torch.count_nonzero(v[ids])
              for k, v in self._term_dones.items()}
    for term_cfg in self._class_term_cfgs:
        term_cfg.func.reset(env_ids=env_ids)
    return extras


def _curriculum_reset(self, env_ids=None):
    extras = {}
    for name, value in self._curriculum_state.items():
        if isinstance(value, dict):
            extras.update({f"Curriculum/{name}/{k}": v for k, v in value.items()})
        elif value is not None:
            extras[f"Curriculum/{name}"] = value
    for term_cfg in self._class_term_cfgs:
        term_cfg.func.reset(env_ids=env_ids)
    return extras


# Velocity command: ranges live in a GPU tensor that the curriculum updates.

def velocity_ranges(term):
    """[lin_vel_x, lin_vel_y, ang_vel_z] x [low, high], shared with the curriculum."""
    if not hasattr(term, "_nanorl_ranges"):
        r = term.cfg.ranges
        term._nanorl_ranges = torch.tensor([r.lin_vel_x, r.lin_vel_y, r.ang_vel_z], device=term.device)
    return term._nanorl_ranges


def _velocity_resample(self, env_ids):
    ranges = velocity_ranges(self)
    r = torch.empty(len(env_ids), device=self.device)
    for axis in range(3):
        low, high = ranges[axis, 0], ranges[axis, 1]
        self.vel_command_b[env_ids, axis] = r.uniform_(0.0, 1.0) * (high - low) + low
    if self.cfg.heading_command:
        self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
        self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
    self.is_world_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_world_envs
    self.vel_command_w[env_ids] = self.vel_command_b[env_ids]
    self.is_forward_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_forward_envs
    command = self.vel_command_b[env_ids]
    forward = torch.stack((command[:, 0].abs().clamp(min=0.3),
                           torch.zeros_like(command[:, 0]), torch.zeros_like(command[:, 0])), dim=1)
    self.vel_command_b[env_ids] = torch.where(self.is_forward_env[env_ids, None], forward, command)


def _velocity_update(self, env_ids=None):
    ranges = velocity_ranges(self)
    heading = self.robot.data.heading_w
    if self.cfg.heading_command:
        self.heading_error = wrap_to_pi(self.heading_target - heading)
        angular = torch.clip(self.cfg.heading_control_stiffness * self.heading_error,
                             min=ranges[2, 0], max=ranges[2, 1])
        self.vel_command_b[:, 2] = torch.where(self.is_heading_env, angular, self.vel_command_b[:, 2])
    cos_h, sin_h = torch.cos(heading), torch.sin(heading)
    vx, vy = self.vel_command_w[:, 0], self.vel_command_w[:, 1]
    rotated = torch.stack((cos_h * vx + sin_h * vy, -sin_h * vx + cos_h * vy), dim=1)
    self.vel_command_b[:, :2] = torch.where(self.is_world_env[:, None], rotated, self.vel_command_b[:, :2])
    standing = self.is_standing_env[:, None]
    self.vel_command_b.copy_(torch.where(standing, 0.0, self.vel_command_b))
    self.vel_command_w.copy_(torch.where(standing, 0.0, self.vel_command_w))


# Curricula: switch on the GPU step counter instead of Python values.

def commands_vel(env, env_ids, command_name, velocity_stages):
    term = env.command_manager.get_term(command_name)
    ranges = velocity_ranges(term)
    if not hasattr(term, "_nanorl_stages"):
        names = ("lin_vel_x", "lin_vel_y", "ang_vel_z")
        term._nanorl_stages = [(stage["step"], axis, torch.tensor(stage[n], device=env.device))
                               for stage in velocity_stages for axis, n in enumerate(names)
                               if stage.get(n) is not None]
    for start, axis, value in term._nanorl_stages:
        ranges[axis].copy_(torch.where(step(env)[0] >= start, value, ranges[axis]))
    return {"lin_vel_x_min": ranges[0, 0], "lin_vel_x_max": ranges[0, 1],
            "lin_vel_y_min": ranges[1, 0], "lin_vel_y_max": ranges[1, 1],
            "ang_vel_z_min": ranges[2, 0], "ang_vel_z_max": ranges[2, 1]}


def terrain_levels_vel(env, env_ids, command_name, asset_cfg=None):
    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    terrain = env.scene.terrain
    generator = terrain.cfg.terrain_generator
    command = env.command_manager.get_command(command_name)
    distance = torch.norm(asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    move_up = distance > generator.size[0] / 2
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # Stock freezes levels on the reset before any env step.
    first = step(env)[0] == 0
    terrain.update_env_origins(env_ids, move_up & ~first, move_down & ~first)
    levels = terrain.terrain_levels.float()
    result = {"mean": levels.mean(), "max": levels.max()}
    names = list(generator.sub_terrains.keys())
    if terrain.terrain_origins.shape[1] == len(names):
        for i, name in enumerate(names):
            # Stock omits empty types; here an empty type logs 0.
            mask = (terrain.terrain_types == i).float()
            result[name] = (levels * mask).sum() / mask.sum().clamp(min=1)
    return result


class RewardWeightCurriculum:
    """Stock reward_curriculum for weight-only stages, with the weight in a GPU tensor."""

    def __init__(self, stock, env):
        self._term_cfg = stock._term_cfg
        if any("params" in stage for stage in stock._stages):
            raise ValueError("reward_curriculum stages that change params are not supported yet")
        self._term_cfg._nanorl_weight = torch.tensor(float(self._term_cfg.weight), device=env.device)
        self._stages = [(stage["step"], torch.tensor(float(stage["weight"]), device=env.device))
                        for stage in stock._stages if "weight" in stage]
        self.__name__ = "reward_curriculum"

    def __call__(self, env, env_ids, **params):
        weight = self._term_cfg._nanorl_weight
        for start, value in self._stages:
            weight.copy_(torch.where(step(env)[0] >= start, value, weight))
        return {"weight": weight}

    def reset(self, env_ids=None):
        pass


def _reward_compute(self, dt):
    # Stock compute, reading the GPU weight when a curriculum owns it.
    self._reward_buf[:] = 0.0
    scale = dt if self._scale_by_dt else 1.0
    for index, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs, strict=False)):
        weight = getattr(term_cfg, "_nanorl_weight", None)
        if weight is None:
            if term_cfg.weight == 0.0:
                self._step_reward[:, index] = 0.0
                continue
            weight = term_cfg.weight
        value = term_cfg.func(self._env, **term_cfg.params)
        self._check_term_shape(name, value)
        value = torch.nan_to_num(value * weight * scale, nan=0.0, posinf=0.0, neginf=0.0)
        self._reward_buf += value
        self._episode_sums[name] += value
        self._step_reward[:, index] = value / scale
    return self._reward_buf


# Motion tracking: worlds that reach the end of their reference motion resample.

def _motion_update(self, env_ids=None):
    self.time_steps += 1
    wrapped = self.time_steps >= self.motion.time_step_total
    when(self._env, wrapped, self._resample_command)
    # Resampling teleports the robot; refresh kinematics before relative poses.
    self._pending_forward = False
    cond(self._env, wrapped.any(), self._env.sim.forward)
    self.update_relative_body_poses()
    if self.cfg.sampling_mode == "adaptive":
        self.bin_failed_count = (self.cfg.adaptive_alpha * self._current_bin_failed
                                 + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count)
        self._current_bin_failed.zero_()


def _motion_adaptive(self, env_ids):
    # Failure counts by scatter, kept only when some episode failed (stock branches on it).
    failed = self._env.termination_manager.terminated[env_ids].float()
    bins = torch.clamp((self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1),
                       0, self.bin_count - 1)[env_ids]
    counts = torch.zeros_like(self._current_bin_failed).scatter_add_(0, bins, failed)
    self._current_bin_failed.copy_(torch.where(failed.sum() > 0, counts, self._current_bin_failed))
    probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
    probabilities = torch.nn.functional.pad(probabilities[None, None],
                                            (0, self.cfg.adaptive_kernel_size - 1), mode="replicate")
    probabilities = torch.nn.functional.conv1d(probabilities, self.kernel.view(1, 1, -1)).view(-1)
    probabilities = probabilities / probabilities.sum()
    sampled = torch.multinomial(probabilities, len(env_ids), replacement=True)
    self.time_steps[env_ids] = ((sampled + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
                                / self.bin_count * (self.motion.time_step_total - 1)).long()
    entropy = -(probabilities * (probabilities + 1e-12).log()).sum()
    self.metrics["sampling_entropy"][:] = entropy / math.log(self.bin_count) if self.bin_count > 1 else 1.0
    top_prob, top_bin = probabilities.max(dim=0)
    self.metrics["sampling_top1_prob"][:] = top_prob
    self.metrics["sampling_top1_bin"][:] = top_bin.float() / self.bin_count


def _yaw_rotation(self, rot_mat):
    # Stock falls back to the y axis when x is near vertical; select it by mask.
    x = rot_mat[:, :, 0].clone()
    x[:, 2] = 0
    norm = x.norm(dim=1)
    y = rot_mat[:, :, 1].clone()
    y[:, 2] = 0
    y = y / y.norm(dim=1).clamp(min=1e-6)[:, None]
    fallback = torch.stack((y[:, 1], -y[:, 0], torch.zeros_like(y[:, 0])), dim=1)
    singular = norm < 0.1
    x = torch.where(singular[:, None], fallback, x)
    x = x / torch.where(singular, 1.0, norm).clamp(min=1e-6)[:, None]
    yaw = torch.zeros_like(rot_mat)
    yaw[:, 0, 0], yaw[:, 1, 0] = x[:, 0], x[:, 1]
    yaw[:, 0, 1], yaw[:, 1, 1] = -x[:, 1], x[:, 0]
    yaw[:, 2, 2] = 1
    return yaw


# Curriculum functions are referenced from this env's config, so they are swapped per env.
CURRICULA = {curriculums.commands_vel: commands_vel,
             curriculums.terrain_levels_vel: terrain_levels_vel}


def install(env):
    _override(CommandTerm, "reset", _command_reset)
    _override(TerminationManager, "reset", _termination_reset)
    _override(CurriculumManager, "reset", _curriculum_reset)
    _override(RewardManager, "compute", _reward_compute)
    _override(UniformVelocityCommand, "_resample_command", _velocity_resample)
    _override(UniformVelocityCommand, "_update_command", _velocity_update)
    _override(MotionCommand, "_update_command", _motion_update)
    _override(MotionCommand, "_adaptive_sampling", _motion_adaptive)
    _override(RayCastSensor, "_extract_yaw_rotation", _yaw_rotation)
    for sensor in env.scene._sensors.values():
        sensor._nanorl_captured = True  # Sensors have no env reference to check.
    for term_cfg in getattr(env.curriculum_manager, "_term_cfgs", []):
        if isinstance(term_cfg.func, reward_curriculum):
            term_cfg.func = RewardWeightCurriculum(term_cfg.func, env)
        term_cfg.func = CURRICULA.get(term_cfg.func, term_cfg.func)
        if term_cfg.func not in CURRICULA.values() and not isinstance(term_cfg.func, RewardWeightCurriculum):
            # Curricula often write Python config values, which a graph freezes.
            print(f"[nanorl] warning: curriculum {term_cfg.func.__name__} is not known to be "
                  "capture-safe; it must write tensors, not Python values.", flush=True)
    for term in env.command_manager._terms.values():
        if isinstance(term, UniformVelocityCommand) and term.cfg.init_velocity_prob:
            raise ValueError("init_velocity_prob > 0 is not supported yet")
