import numpy as np
import cv2


class World:
    """2D binary world (Conway's Game of Life)."""

    def __init__(self, width=20, height=20):
        self.width = width
        self.height = height
        self.state = np.random.randint(0, 2, size=(height, width)).astype(np.uint8)
        self.time_step = 0

    def get_value(self, x, y):
        return self.state[y % self.height, x % self.width]

    def set_value(self, x, y, value):
        self.state[y % self.height, x % self.width] = value

    def step(self):
        p = np.pad(self.state, 1, mode='wrap')
        neighbours = (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
                      p[1:-1, :-2]                + p[1:-1, 2:] +
                      p[2:,   :-2] + p[2:,  1:-1] + p[2:,  2:])
        survive = (self.state == 1) & ((neighbours == 2) | (neighbours == 3))
        born    = (self.state == 0) &  (neighbours == 3)
        self.state = (survive | born).view(np.uint8)
        self.time_step += 1
