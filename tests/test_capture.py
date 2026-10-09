"""CPU tests: nanorl must match stock mjlab, and must flag what a graph would break.

    pytest            # Cartpole and Lift-Cube (a few minutes)
    pytest --full     # plus G1, Go1, rough terrain and motion tracking (much longer)

On CPU the captured step runs eagerly, so these check nanorl's rewrite of the
step (masked resets, re-pinning, term fixes) with exact comparisons. GPU
capture itself is checked with `python -m nanorl check TASK` on a CUDA machine.
"""

from types import SimpleNamespace

import pytest
import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.registry import load_env_cfg
import nanorl
from nanorl import check

FAST = ["Mjlab-Cartpole-Balance", "Mjlab-Lift-Cube-Yam"]
SLOW = ["Mjlab-Cartpole-Swingup", "Mjlab-Velocity-Flat-Unitree-G1", "Mjlab-Velocity-Flat-Unitree-Go1",
        "Mjlab-Velocity-Rough-Unitree-G1", "Mjlab-Velocity-Rough-Unitree-Go1"]


@pytest.mark.parametrize("task", FAST)
def test_matches_stock(task):
    # Raises AssertionError on any difference from stock mjlab.
    check.main([task, "--device", "cpu"])


@pytest.mark.full
@pytest.mark.parametrize("task", SLOW)
def test_matches_stock_slow(task):
    check.main([task, "--device", "cpu"])


@pytest.mark.full
def test_matches_stock_tracking():
    from examples.tracking import make_motion
    if not make_motion.OUTPUT.exists():
        make_motion.main([])
    check.main(["Nanorl-Tracking-Flat-G1-Synthetic", "--import", "examples.tracking.tasks", "--device", "cpu"])


class counting_reward:
    """A term with Python state that changes every step; a graph would freeze it."""

    def __init__(self, cfg, env):
        self.calls = 0

    def __call__(self, env):
        self.calls += 1
        return torch.zeros(env.num_envs, device=env.device)


@pytest.mark.parametrize("add_counter", [False, True])
def test_warns_about_changing_python_state(add_counter, capsys):
    cfg = load_env_cfg("Mjlab-Cartpole-Balance")
    cfg.scene.num_envs = 4
    if add_counter:
        cfg.rewards["counter"] = RewardTermCfg(func=counting_reward, weight=1e-9)
    nanorl.capture(ManagerBasedRlEnv(cfg=cfg, device="cpu")).close()
    output = capsys.readouterr().out
    if add_counter:
        assert "env.reward_manager._term_cfgs[1].func.calls" in output
    else:
        assert "change every step" not in output


def test_primitives_without_capture():
    # Uncaptured, the helpers behave like ordinary Python so terms run in stock mjlab.
    env = SimpleNamespace(common_step_counter=7, device="cpu")
    picked, ran = [], []
    nanorl.when(env, torch.tensor([False, True, False, True]), picked.append)
    nanorl.when(env, torch.zeros(4, dtype=torch.bool), picked.append)
    nanorl.cond(env, torch.tensor(True), lambda: ran.append(1))
    nanorl.cond(env, torch.tensor(False), lambda: ran.append(2))
    assert len(picked) == 1 and picked[0].tolist() == [1, 3]
    assert ran == [1]
    assert nanorl.step(env).tolist() == [7]


def test_check_catches_a_broken_masked_commit(monkeypatch):
    # Guards the checker itself: if resets leaked into worlds that did not finish,
    # the parity check must fail.
    from contextlib import contextmanager
    import nanorl.state

    @contextmanager
    def keep_every_world(self, mask):
        yield
    monkeypatch.setattr(nanorl.state.WorldState, "masked", keep_every_world)
    with pytest.raises(AssertionError):
        check.main(["Mjlab-Cartpole-Balance", "--device", "cpu"])
