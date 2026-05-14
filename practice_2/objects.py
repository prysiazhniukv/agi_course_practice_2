from collections import deque
from dataclasses import dataclass, field
import numpy as np


_HISTORY_MAXLEN   = 64    # frames buffered; covers periods up to 32 with 2× margin
_PERIOD_THRESHOLD = 0.05  # mean per-pixel abs diff below this → frames considered identical
_MIN_CONFIRM      = 3     # consecutive matching pairs required to confirm a period
_CHECK_EVERY      = 20    # re-run period detection every N newly recorded frames


@dataclass
class ObjectHistory:
    """
    Temporal record of a WorldObject's observed binary states.

    Crops are extracted from world_values anchored to the tracked centroid
    (not the per-frame bbox), so the sequence stays spatially stable across
    GoL evolution and saliency jitter.
    """
    half_size:           int                      # half-width of the square crop
    frames:              deque = field(default_factory=lambda: deque(maxlen=_HISTORY_MAXLEN))
    period:              int | None  = None       # 1=still, 2=blinker, …
    cycle:               list | None = None       # list[np.ndarray], one full period
    _frames_since_check: int = field(default=0, repr=False)

    def record(self, frame: np.ndarray):
        """Append a new binary frame and, when due, re-run period detection."""
        self.frames.append(frame.copy())
        self._frames_since_check += 1
        n = len(self.frames)
        if n >= 4 and (self.period is None or self._frames_since_check >= _CHECK_EVERY):
            self._try_detect_period()
            self._frames_since_check = 0

    def _try_detect_period(self):
        n = len(self.frames)
        if n < 4:
            return
        frames_list = list(self.frames)
        for k in range(1, n // 2 + 1):
            n_confirm = min(_MIN_CONFIRM, n - k)
            confirmed = all(
                np.abs(
                    frames_list[-(j + 1)].astype(np.float32) -
                    frames_list[-(j + 1 + k)].astype(np.float32)
                ).mean() <= _PERIOD_THRESHOLD
                for j in range(n_confirm)
            )
            if confirmed:
                if k != self.period:
                    print(f'[History] period detected: {k}  (was {self.period})')
                    self.period = k
                    self.cycle  = frames_list[-k:]
                return


@dataclass
class WorldObject:
    id:            int
    centroid:      np.ndarray        # (x, y) float, world coordinates
    area:          int
    bbox:          tuple             # (x, y, w, h) world coordinates
    motion_score:  float
    first_seen:    int
    last_seen:     int
    snapshot:      np.ndarray | None   = None   # RGB crop at tracked centroid
    history:       ObjectHistory | None = None


class ObjectTracker:
    """
    Maintains a set of WorldObjects across frames.

    Each call to update():
      1. Converts blob centroids from local patch coords → world coords.
      2. Matches blobs to existing objects by nearest-centroid (toroidal).
         On match: records a centroid-anchored binary frame into the object's
         ObjectHistory and refreshes the RGB snapshot.
      3. Spawns new objects for unmatched blobs unless a near-duplicate exists.

    Objects are never evicted.  Frame crops are anchored to the tracked centroid
    (not the per-frame bbox) so the history sequence is spatially stable.
    """

    def __init__(self, world_width: int, world_height: int,
                 match_radius: float = 5.0,
                 dedup_radius: float = 10.0,
                 area_tolerance: float = 0.5):
        self.world_width    = world_width
        self.world_height   = world_height
        self.match_radius   = match_radius
        self.dedup_radius   = dedup_radius   # wider search for duplicate suppression
        self.area_tolerance = area_tolerance # min(a,b)/max(a,b) threshold for "same size"
        self._objects: dict[int, WorldObject] = {}
        self._next_id = 0

    @property
    def objects(self) -> dict[int, WorldObject]:
        return self._objects

    def _to_world(self, cx_local: float, cy_local: float,
                  agent_pos: list, radius: int) -> np.ndarray:
        """Convert a patch-local (x, y) to world coordinates (with wrapping)."""
        wx = (agent_pos[0] - radius + cx_local) % self.world_width
        wy = (agent_pos[1] - radius + cy_local) % self.world_height
        return np.array([wx, wy], dtype=np.float32)

    def _toroidal_dist(self, a: np.ndarray, b: np.ndarray) -> float:
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        dx = min(dx, self.world_width  - dx)
        dy = min(dy, self.world_height - dy)
        return float(np.sqrt(dx * dx + dy * dy))

    def _blob_to_world_bbox(self, bbox_local: tuple, agent_pos: list,
                            radius: int) -> tuple:
        bx, by, bw, bh = bbox_local
        wx = int((agent_pos[0] - radius + bx) % self.world_width)
        wy = int((agent_pos[1] - radius + by) % self.world_height)
        return (wx, wy, bw, bh)

    def _extract_frame(self, world_values: np.ndarray, centroid: np.ndarray,
                       half_size: int) -> np.ndarray:
        """Square binary crop (uint8) centred on centroid with toroidal wrapping."""
        cx = int(round(float(centroid[0])))
        cy = int(round(float(centroid[1])))
        H, W = world_values.shape
        ys = np.arange(cy - half_size, cy + half_size + 1) % H
        xs = np.arange(cx - half_size, cx + half_size + 1) % W
        return world_values[np.ix_(ys, xs)].copy()

    @staticmethod
    def _frame_to_rgb(frame: np.ndarray) -> np.ndarray:
        rgb = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        rgb[frame == 1] = (255, 255, 255)
        return rgb

    def update(self, blobs: list[dict], agent_pos: list,
               radius: int, world_time: int,
               world_values: np.ndarray | None = None) -> dict[int, WorldObject]:

        # Convert blobs to world centroids
        new_centroids = [
            self._to_world(*b['centroid_local'], agent_pos, radius)
            for b in blobs
        ]

        matched_ids   = set()
        matched_blobs = set()

        # Greedy nearest-centroid matching
        for obj_id, obj in self._objects.items():
            best_blob, best_dist = None, self.match_radius
            for i, centroid in enumerate(new_centroids):
                if i in matched_blobs:
                    continue
                d = self._toroidal_dist(obj.centroid, centroid)
                if d < best_dist:
                    best_dist, best_blob = d, i
            if best_blob is not None:
                b = blobs[best_blob]
                obj.centroid     = new_centroids[best_blob]
                obj.area         = b['area']
                obj.bbox         = self._blob_to_world_bbox(b['bbox_local'], agent_pos, radius)
                obj.motion_score = b['motion_score']
                obj.last_seen    = world_time
                if world_values is not None and obj.history is not None:
                    frame = self._extract_frame(world_values, obj.centroid, obj.history.half_size)
                    obj.history.record(frame)
                    obj.snapshot = self._frame_to_rgb(frame)
                matched_ids.add(obj_id)
                matched_blobs.add(best_blob)

        # Spawn new objects for unmatched blobs (with duplicate suppression)
        for i, b in enumerate(blobs):
            if i not in matched_blobs:
                centroid = new_centroids[i]
                new_area = b['area']

                # Dedup: skip if a stored object is within dedup_radius AND similar area.
                # This handles the case where the agent revisits an area and the same
                # physical object is detected again after GoL evolution has drifted it
                # slightly beyond match_radius.
                is_duplicate = any(
                    self._toroidal_dist(obj.centroid, centroid) < self.dedup_radius
                    and obj.area > 0
                    and min(new_area, obj.area) / max(new_area, obj.area) >= self.area_tolerance
                    for obj in self._objects.values()
                )
                if is_duplicate:
                    continue

                world_bbox = self._blob_to_world_bbox(b['bbox_local'], agent_pos, radius)
                _, _, bw, bh = world_bbox
                half_size   = max(bw, bh) // 2 + 3
                history     = ObjectHistory(half_size=half_size)
                snap = None
                if world_values is not None:
                    frame = self._extract_frame(world_values, centroid, half_size)
                    history.record(frame)
                    snap = self._frame_to_rgb(frame)
                obj = WorldObject(
                    id           = self._next_id,
                    centroid     = centroid,
                    area         = new_area,
                    bbox         = world_bbox,
                    motion_score = b['motion_score'],
                    first_seen   = world_time,
                    last_seen    = world_time,
                    snapshot     = snap,
                    history      = history,
                )
                self._objects[self._next_id] = obj
                self._next_id += 1

        return self._objects
