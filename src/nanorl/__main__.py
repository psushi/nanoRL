"""nanorl command line.

    python -m nanorl check TASK [--device cpu]      parity against stock mjlab
    python -m nanorl train TASK [mjlab train flags]  stock training, captured env
    python -m nanorl bench TASK [--worlds 1024]     env SPS, stock vs captured

Each command takes --import MODULE (repeatable) to register your own tasks.
"""

import importlib
import sys

COMMANDS = {"check": "nanorl.check", "train": "nanorl.train", "bench": "nanorl.bench"}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(__doc__)
    command, argv = sys.argv[1], sys.argv[2:]
    importlib.import_module(COMMANDS[command]).main(argv)


if __name__ == "__main__":
    main()
