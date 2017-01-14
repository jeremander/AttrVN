from operator import lt, gt

class BinaryHeap:
    """Implements a binary heap. 
    See: https://interactivepython.org/runestone/static/pythonds/Trees/BinaryHeapImplementation.html"""
    def __init__(self, increasing = True):
        self.increasing = increasing
        (self.lt, self.gt) = (lt, gt) if self.increasing else (gt, lt)
        self.heap_list = [0]
        self.current_size = 0
    def percolate_up(self, i):
        while (i // 2 > 0):
            if self.lt(self.heap_list[i], self.heap_list[i // 2]):
                tmp = self.heap_list[i // 2]
                self.heap_list[i // 2] = self.heap_list[i]
                self.heap_list[i] = tmp
            i = i // 2
    def insert(self, k):
        self.heap_list.append(k)
        self.current_size = self.current_size + 1
        self.percolate_up(self.current_size)
    def percolate_down(self, i):
        while ((i * 2) <= self.current_size):
            mc = self.min_child(i)
            if self.gt(self.heap_list[i], self.heap_list[mc]):
                tmp = self.heap_list[i]
                self.heap_list[i] = self.heap_list[mc]
                self.heap_list[mc] = tmp
            i = mc
    def min(self):
        return self.heap_list[1]
    def min_child(self, i):
        if (i * 2 + 1 > self.current_size):
            return i * 2
        else:
            if self.lt(self.heap_list[i * 2], self.heap_list[i * 2 + 1]):
                return i * 2
            else:
                return i * 2 + 1
    def delete_min(self):
        retval = self.heap_list[1]
        self.heap_list[1] = self.heap_list[self.current_size]
        self.current_size = self.current_size - 1
        self.heap_list.pop()
        self.percolate_down(1)
        return retval
    @classmethod
    def build(cls, keys, increasing = True):
        heap = cls(increasing = increasing)
        i = len(keys) // 2
        heap.current_size = len(keys)
        heap.heap_list = [0] + keys[:]
        while (i > 0):
            heap.percolate_down(i)
            i = i - 1
        return heap
    def __repr__(self):
        return str(self.heap_list[1:])

class KeyValuePair():
    def __init__(self, key, value):
        self.pair = (key, value)
    def __getitem__(self, i):
        return self.pair[i]
    def __repr__(self):
        return str(self.pair)
    def __gt__(self, other):
        return (self.pair[1] > other.pair[1])
    def __lt__(self, other):
        return (self.pair[1] < other.pair[1])
    def __eq__(self, other):
        return (self.pair[1] == other.pair[1])
