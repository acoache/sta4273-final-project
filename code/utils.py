import numpy as np
from typing import Any, Tuple
import os.path

"""
# Misc functions or classes

* Replay -- Replay buffer
* directory -- Function to define new directory
"""

# completely copied code from Osband et al. notebook, convenient way to store algorithm memory used to update network
class Replay(object):

    def __init__(self, capacity: int):
        self._data = None
        self.capacity = capacity
        self._num_added = 0

    def add(self, transition: Tuple[Any]) -> None:
        if self._data is None:
            self._preallocate(transition)

        for d, item in zip(self._data, transition):
            d[self._num_added % self.capacity] = item

        self._num_added += 1

    def sample(self, batch_size: int = 1) -> Tuple[np.ndarray]:
        """Returns a transposed/stacked minibatch. Each array has shape [B, ...]."""
        indices = np.random.randint(self.size, size=batch_size)
        return [d[indices] for d in self._data]

    @property
    def size(self) -> int:
        return min(self.capacity, self._num_added)

    def _preallocate(self, items: Tuple[Any]) -> None:
        """Assume flat structure of items."""
        items_np = [np.asarray(x) for x in items]

        if sum([x.nbytes for x in items_np]) * self.capacity > 1e9:
            raise ValueError('This replay buffer would preallocate > 1GB of memory.')

        self._data = [np.zeros(dtype=x.dtype, shape=(self.capacity,) + x.shape)
                      for x in items_np]

def directory(file):
    if os.path.exists(file):
        return
    else:
        os.mkdir(file)
    return