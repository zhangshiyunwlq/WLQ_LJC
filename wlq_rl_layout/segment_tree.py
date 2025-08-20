class SegmentTree:
    def __init__(self, capacity, operation, neutral_element):
        self.capacity = capacity
        self.operation = operation
        self.neutral_element = neutral_element

        self.tree = [neutral_element for _ in range(2 * capacity)]

    def _reduce(self, start=0, end=None):
        if end is None:
            end = self.capacity

        start += self.capacity
        end += self.capacity

        result = self.neutral_element
        while start < end:
            if start & 1:
                result = self.operation(result, self.tree[start])
                start += 1
            if end & 1:
                end -= 1
                result = self.operation(result, self.tree[end])
            start >>= 1
            end >>= 1

        return result

    def __setitem__(self, idx, val):
        idx += self.capacity
        self.tree[idx] = val

        idx >>= 1
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx >>= 1

    def __getitem__(self, idx):
        return self.tree[idx + self.capacity]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(capacity, operation=lambda a, b: a + b, neutral_element=0.0)

    def sum(self, start=0, end=None):
        return self._reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1

        while idx < self.capacity:
            if self.tree[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self.tree[2 * idx]
                idx = 2 * idx + 1

        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(capacity, operation=min, neutral_element=float('inf'))

    def min(self, start=0, end=None):
        return self._reduce(start, end)