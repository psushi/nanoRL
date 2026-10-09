"""Register tasks for the nanorl commands: stock mjlab tasks plus --import modules."""

import importlib


def pop_imports(argv):
    """Remove `--import MODULE` pairs from argv and import those modules."""
    import mjlab.tasks  # noqa: F401  Registers the stock tasks.
    rest, i = [], 0
    while i < len(argv):
        if argv[i] == "--import":
            importlib.import_module(argv[i + 1])
            i += 2
        else:
            rest.append(argv[i])
            i += 1
    return rest
