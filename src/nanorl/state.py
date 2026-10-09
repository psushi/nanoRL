"""Find per-world state on an mjlab env and keep selected worlds' writes.

A "slot" is a place that holds a tensor: an object attribute, a dict entry,
or a list element. Slots let us restore state even when code replaces a
tensor instead of writing into it.
"""

from contextlib import contextmanager
import dataclasses

import torch
import warp as wp

# MuJoCo state that defines a world; derived fields are recomputed by forward().
PHYSICS_FIELDS = ("qpos", "qvel", "act", "history", "qacc", "qacc_warmstart", "ctrl",
                  "qfrc_applied", "xfrc_applied", "eq_active", "mocap_pos", "mocap_quat",
                  "time", "tree_asleep")
# nanorl's own objects (the wrapper the env points back to) are not env state.
_SKIP_MODULES = ("torch", "warp", "mujoco", "numpy", "builtins", "collections", "typing",
                 "tensordict", "gymnasium", "rsl_rl", "nanorl")


@dataclasses.dataclass(eq=False)
class Slot:
    container: object
    key: object
    pinned: torch.Tensor
    per_world: bool

    def get(self):
        if self.container is None:
            return self.pinned  # Inside a tuple: no slot to re-read, the tensor itself is the state.
        if isinstance(self.container, (dict, list)):
            return self.container[self.key]
        return getattr(self.container, self.key)

    def set(self, value):
        if isinstance(self.container, (dict, list)):
            self.container[self.key] = value
        else:
            setattr(self.container, self.key, value)


def _walkable(value):
    module = type(value).__module__.split(".")[0]
    return hasattr(value, "__dict__") and module not in _SKIP_MODULES and not isinstance(value, type)


def discover(env):
    """Every tensor slot reachable from the env, plus physics state.

    All slots are re-pinned when code replaces their tensor; only per-world
    slots (leading dimension = number of worlds) take part in masked commits.
    """
    num_worlds, device = env.num_envs, torch.device(env.device)
    slots, seen = [], {id(env.sim)}

    def visit(container, key, value):
        if isinstance(value, torch.Tensor):
            # Broadcast views (stride 0) share memory between elements; they are not state.
            expanded = any(st == 0 and sz > 1 for st, sz in zip(value.stride(), value.shape))
            if value.device == device and not expanded:
                per_world = value.ndim > 0 and value.shape[0] == num_worlds
                slots.append(Slot(container, key, value, per_world))
            return
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, dict):
            for k, v in list(value.items()):
                visit(value, k, v)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                visit(value, i, v)
        elif isinstance(value, tuple):
            for v in value:
                visit(None, None, v)
        elif _walkable(value):
            for k, v in list(vars(value).items()):
                visit(value, k, v)

    for k, v in list(vars(env).items()):
        visit(env, k, v)
    holder = {}
    for name in PHYSICS_FIELDS:
        holder[name] = wp.to_torch(getattr(env.sim.wp_data, name))
    # Batched model fields (domain randomization) carry a world dimension.
    for name, value in vars(env.sim.wp_model).items():
        if isinstance(value, wp.array) and value.ndim and value.shape[0] == num_worlds:
            holder[f"model.{name}"] = wp.to_torch(value)
    for k, v in holder.items():
        visit(holder, k, v)
    # Tensors inside tuples have no writable slot (container None); they are
    # still restored in place, just never re-pinned.
    return slots


class WorldState:
    def __init__(self, env):
        self.slots = discover(env)
        # Restore each memory region once even if several slots view it.
        unique = {}
        for slot in self.slots:
            if not slot.per_world:
                continue
            t = slot.pinned
            unique.setdefault((t.data_ptr(), t.dtype, tuple(t.shape), t.stride()), slot)
        self.unique = list(unique.values())

    def repin(self):
        """Copy replaced tensors back into their original allocations."""
        for slot in self.slots:
            if slot.container is None:
                continue
            current = slot.get()
            if current is not slot.pinned and isinstance(current, torch.Tensor) \
                    and current.shape == slot.pinned.shape:
                slot.pinned.copy_(current)
                slot.set(slot.pinned)

    @contextmanager
    def masked(self, mask):
        """Run code for all worlds; keep its writes only where mask is true."""
        # Earlier code this step may have replaced tensors; snapshot current values.
        self.repin()
        saved = [s.pinned.clone() for s in self.unique]
        yield
        self.repin()
        for slot, old in zip(self.unique, saved, strict=True):
            t = slot.pinned
            keep = mask.reshape(-1, *([1] * (t.ndim - 1)))
            t.copy_(torch.where(keep, t, old))


def python_values(env):
    """Python numbers and bools reachable from the env, by readable path.

    A graph replays the kernels recorded at capture; Python values it read are
    frozen at their capture-time value. Comparing these across steps finds
    state that would silently stop changing.
    """
    values, seen = {}, {id(env.sim), id(env)}

    def visit(path, value):
        if isinstance(value, (bool, int, float)):
            values[path] = value
            return
        if isinstance(value, torch.Tensor) or id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, dict):
            for k, v in list(value.items()):
                visit(f"{path}[{k!r}]", v)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                visit(f"{path}[{i}]", v)
        elif _walkable(value):
            for k, v in list(vars(value).items()):
                visit(f"{path}.{k}", v)

    for k, v in list(vars(env).items()):
        visit(f"env.{k}", v)
    return values


def changing_values(before, after):
    """Paths whose Python value changed, ignoring NaN-to-NaN."""
    return sorted(k for k, v in after.items()
                  if k in before and before[k] != v and not (before[k] != before[k] and v != v))
