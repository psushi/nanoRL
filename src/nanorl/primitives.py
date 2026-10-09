"""Helpers for stateful terms (commands, curricula, events) that must run in a graph.

Each works in a captured env and in stock mjlab, so a term written with them
runs either way. `env` is the ManagerBasedRlEnv (a term's `self._env`).

    nanorl.when(env, mask, fn)   fn(env_ids) for the worlds where mask is true
    nanorl.cond(env, pred, fn)   fn() only if the GPU bool pred is true
    nanorl.step(env)             training step count as a GPU tensor of shape [1]
"""

import torch


def _runtime(env):
    return getattr(env, "_nanorl", None)


def when(env, mask, fn):
    """Call fn(env_ids) for the selected worlds.

    Captured: fn runs for every world and only the selected worlds keep its
    writes to per-world state, so world counts never depend on data. Stock:
    fn runs for the selected worlds only. Writes to global state happen in
    both cases only if fn makes them; captured, they happen every step.
    """
    runtime = _runtime(env)
    if runtime is None:
        ids = mask.nonzero().flatten()
        if ids.numel():
            fn(ids)
        return
    with runtime.world.masked(mask):
        fn(runtime.ids)


def cond(env, pred, fn):
    """Call fn() only when pred (a GPU bool) is true; a conditional node when captured."""
    runtime = _runtime(env)
    if runtime is None:
        if bool(pred):
            fn()
        return
    runtime.cond(pred, fn)


def step(env):
    """The training step counter, on the GPU so a graph sees it advance."""
    runtime = _runtime(env)
    if runtime is None:
        return torch.tensor([env.common_step_counter], device=env.device)
    return runtime.step_count
