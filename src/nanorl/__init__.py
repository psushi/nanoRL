"""Capture stock mjlab environments into CUDA graphs.

    env = nanorl.capture(ManagerBasedRlEnv(cfg, device="cuda:0"))

`when`, `cond` and `step` are optional helpers for writing stateful terms
(commands, curricula, events) that follow the capture rules.
"""

from nanorl.primitives import cond, step, when


def capture(env, clip_actions=None):
    """Return a drop-in RSL-RL wrapper whose step is one CUDA graph replay."""
    from nanorl.graph import CapturedEnv
    return CapturedEnv(env, clip_actions)


__all__ = ["capture", "cond", "step", "when"]
