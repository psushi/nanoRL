"""Reserve Warp scratch memory for GPU conditional graph bodies."""

import warp as wp


class ScratchArena:
    """Reserve temporary allocations before entering conditional CUDA graphs.

    One dry run records allocation sizes. Capture reuses the same pointers in
    the same order. This relies on a fixed allocation schedule for this model;
    a changed schedule fails explicitly. Blocks live until the owning process exits.
    """
    def __init__(self, device="cuda:0"):
        self.allocator = wp.get_device(device).get_allocator()
        self.blocks = []
        self.cursor = None

    def allocate(self, size):
        if self.cursor is None:
            pointer = self.allocator.allocate(size)
            self.blocks.append((size, pointer))
            return pointer
        expected_size, pointer = self.blocks[self.cursor]
        assert size == expected_size, "MJWarp scratch allocation schedule changed"
        self.cursor += 1
        return pointer

    def deallocate(self, pointer, size):
        pass  # Keep every scratch block alive through capture and serialization.
