"""Stock mjlab training (config, logging, checkpoints) with nanorl capture.

    python -m nanorl train TASK [--learner captured|rsl] [--import MODULE] [mjlab train flags...]

--learner rsl (default): the task's RSL-RL runner over the captured env.
--learner captured (experimental): rollout and PPO update captured too
    (nanorl.runner), combined with the task's runner class for saving and export.
"""

import sys

import tyro
import mjlab
import mjlab.scripts.train as stock
from nanorl.graph import CapturedEnv
from nanorl.runner import with_captured_learning
from nanorl.tasks import pop_imports


def main(argv=None):
    argv = pop_imports(sys.argv[1:] if argv is None else argv)
    learner = "rsl"
    if "--learner" in argv:
        i = argv.index("--learner")
        learner = argv[i + 1]
        del argv[i:i + 2]
    if not argv or argv[0].startswith("-") or learner not in ("captured", "rsl"):
        sys.exit(__doc__)
    task = argv[0]
    cfg = tyro.cli(stock.TrainConfig, default=stock.TrainConfig.from_task(task),
                   args=argv[1:], config=mjlab.TYRO_FLAGS)
    if cfg.video or cfg.gpu_ids != [0]:
        raise ValueError("nanorl training supports headless GPU 0 only")
    # The stock launcher builds its env wrapper and runner by these names.
    stock.RslRlVecEnvWrapper = CapturedEnv
    if learner == "captured":
        runner_cls = with_captured_learning(stock.load_runner_cls(task))
        stock.load_runner_cls = lambda task_id: runner_cls
        print(f"[nanorl] training with {runner_cls.__name__}", flush=True)
    stock.launch_training(task, cfg)


if __name__ == "__main__":
    main()
