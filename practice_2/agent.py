import numpy as np
import cv2


_DIRECTIONS = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}


class Agent:
    """Agent that can perceive and move in the 2D world."""

    def __init__(self, world, start_pos=(0, 0), perception_radius=1):
        self.world = world
        self.position = list(start_pos)
        self.perception_radius = perception_radius

    def perceive(self):
        r = self.perception_radius
        x, y = self.position
        ys = np.arange(y - r, y + r + 1) % self.world.height
        xs = np.arange(x - r, x + r + 1) % self.world.width
        return self.world.state[np.ix_(ys, xs)]

    def move(self, direction):
        dx, dy = _DIRECTIONS.get(direction, (0, 0))
        self.position[0] = (self.position[0] + dx) % self.world.width
        self.position[1] = (self.position[1] + dy) % self.world.height

    def toggle_cell(self):
        x, y = self.position
        self.world.set_value(x, y, 1 - self.world.get_value(x, y))

    def set_cell(self, value):
        self.world.set_value(self.position[0], self.position[1], value)
