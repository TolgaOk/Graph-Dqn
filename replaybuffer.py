from collections import namedtuple
from random import sample as randsample


class UniformBuffer(object):

    Transition = namedtuple("Transition", ("state",
                                           "action",
                                           "reward",
                                           "next_state",
                                           "terminal")
                            )

    def __init__(self, size):
        self.queue = []
        self.cycle = 0
        self.size = size

    def __len__(self):
        return len(self.queue)

    def push(self, **transition):
        if self.size != len(self.queue):
            self.queue.append(self.Transition(**transition))
        else:
            self.queue.append(self.Transition(**transition))
            self.cycle = (self.cycle + 1) % self.size

    def sample(self, batchsize):
        if batchsize > len(self.queue):
            return None
        batch = randsample(self.queue, batchsize)
        return self.Transition(*zip(*batch))
