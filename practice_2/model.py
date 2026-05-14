import numpy as np
import cv2

from objects import ObjectTracker


# _UNCERTAIN = (128, 128, 128)  # grey – never observed
# _DEAD      = (20,  20,  20)   # dark – known dead
# _ALIVE     = (255, 255, 255)  # white – known alive
# _AGENT     = (0,   0,   255)  # red – agent position
# _BBOX_CLR  = (0,   200, 80)   # green – object bounding box


class WorldModel:
    """
    Agent's internal map of the world built incrementally from observations.

    Cells that have never been inside the agent's perception patch are
    'uncertain' and rendered grey.  Cells that have been observed carry
    their last known value (alive = white, dead = dark).

    When a SaliencyMap is attached, objects (connected clusters of live cells)
    are detected each frame and stored with world-coordinate bounding boxes.
    """

    def __init__(self, width: int, height: int, saliency=None):
        self.width   = width
        self.height  = height
        self.saliency = saliency
        self._values  = np.zeros((height, width), dtype=np.uint8)
        self._known   = np.zeros((height, width), dtype=bool)
        self._tracker = ObjectTracker(width, height)

    @property
    def objects(self) -> dict:
        return self._tracker.objects

    def update(self, perception: np.ndarray, agent_pos: list,
               perception_radius: int, world_time: int = 0):
        """Stamp the latest perception patch and, if saliency is set, detect objects."""
        r = perception_radius
        x, y = agent_pos
        ys = np.arange(y - r, y + r + 1) % self.height
        xs = np.arange(x - r, x + r + 1) % self.width
        idx = np.ix_(ys, xs)
        self._values[idx] = perception
        self._known[idx]  = True

        if self.saliency is not None:
            self.saliency.update(perception)
            blobs = self.saliency.segment()
            self._tracker.update(blobs, agent_pos, r, world_time, self._values)

    @property
    def coverage(self) -> float:
        return self._known.mean()

