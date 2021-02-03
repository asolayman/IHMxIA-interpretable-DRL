from random import sample
import numpy as np


class ReplayMemory:
    def __init__(self, args):
        channels = 1
        state_shape = (args.memory_size, args.channels, args.img_width, args.img_height)
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(args.memory_size, dtype=np.int32)
        self.r = np.zeros(args.memory_size, dtype=np.float32)
        self.isterminal = np.zeros(args.memory_size, dtype=np.float32)

        self.capacity = args.memory_size
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos, :, :] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :, :] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]
