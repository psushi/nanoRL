"""Hoist Python constants used by tensor indexing onto the GPU before capture.

Stock code writes `x[ids] = 0.0`, indexes with `x[:, [1, 3]]`, and calls
`torch.tensor([...], device=...)` inside the step. Each copies a small CPU
tensor to the GPU, which CUDA graph capture forbids. During warmup this mode
creates a GPU constant for every such value; capture then reuses it.
"""

import numbers

import numpy as np
import torch
from torch.overrides import TorchFunctionMode


def _python_ints(value):
    return (isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0
            and all(isinstance(v, (numbers.Integral, np.integer)) and not isinstance(v, bool)
                    for v in np.asarray(value).reshape(-1).tolist()))


class HoistConstants(TorchFunctionMode):
    def __init__(self, device, frozen=False):
        super().__init__()
        self.device, self.frozen, self.cache = device, frozen, {}

    def constant(self, key, make):
        if key not in self.cache:
            if self.frozen:
                raise RuntimeError(f"nanorl: new constant during capture: {key[:2]}")
            self.cache[key] = make()
        return self.cache[key]

    def index(self, index):
        if isinstance(index, tuple):
            return tuple(self.index(i) for i in index)
        if isinstance(index, list) and _python_ints(index):
            key = ("index", tuple(index))
            return self.constant(key, lambda: torch.tensor(index, device=self.device))
        return index

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.Tensor.__getitem__ and args[0].is_cuda:
            args = (args[0], self.index(args[1]))
        elif func is torch.Tensor.__setitem__ and args[0].is_cuda:
            target, index, value = args
            if isinstance(value, (numbers.Number, bool)):
                key = ("value", value, target.dtype)
                value = self.constant(key, lambda: torch.tensor(value, dtype=target.dtype,
                                                                device=target.device))
            args = (target, self.index(index), value)
        elif func is torch.tensor and _is_cuda(kwargs.get("device")):
            data = args[0]
            if isinstance(data, (numbers.Number, list, tuple)):
                key = ("tensor", repr(data), str(kwargs.get("dtype")))
                return self.constant(key, lambda: func(*args, **kwargs))
        return func(*args, **kwargs)


def _is_cuda(device):
    return device is not None and torch.device(device).type == "cuda"
