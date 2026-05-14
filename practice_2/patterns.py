import json
import numpy as np


_METRICS = ('hamming', 'cosine')


def load_pattern(path: str) -> tuple[np.ndarray, str]:
    """Load a pattern from a JSON file. Returns (pattern_array, name)."""
    with open(path) as f:
        data = json.load(f)
    return np.array(data['pattern'], dtype=np.uint8), data.get('name', path)


class PatternMatcher:
    """
    Matches a binary pattern against WorldObjects using a sliding-window metric.

    metric='hamming'  – sum of differing bits (lower = better, 0 = perfect match)
    metric='cosine'   – 1 - cosine_similarity of flattened window and pattern
                        (lower = better, 0 = identical direction in bit-space)

    For each object, the pattern is slid over an expanded bbox region so that
    surrounding dead-cell context is included in the comparison.
    The minimum score across all window positions is the object's match score.
    """

    def __init__(self, pattern: np.ndarray, pattern_name: str = '',
                 metric: str = 'cosine'):
        if metric not in _METRICS:
            raise ValueError(f'metric must be one of {_METRICS}, got {metric!r}')
        self.pattern      = pattern.astype(np.float32)
        self.pattern_name = pattern_name
        self.metric       = metric
        self._ph, self._pw = pattern.shape
        self._p_flat      = self.pattern.ravel()
        self._p_norm      = float(np.linalg.norm(self._p_flat))

    def _dist(self, window: np.ndarray) -> float:
        """Compute distance between window and pattern (lower = more similar)."""
        if self.metric == 'hamming':
            return float(np.abs(window - self.pattern).sum())
        # cosine: 1 - cos_sim; treat all-zero window as maximally dissimilar
        w_flat = window.ravel()
        w_norm = float(np.linalg.norm(w_flat))
        if w_norm == 0 or self._p_norm == 0:
            return 1.0
        return 1.0 - float(np.dot(w_flat, self._p_flat) / (w_norm * self._p_norm))

    def _extract_region(self, values: np.ndarray, bbox: tuple) -> np.ndarray:
        """Extract bbox region from the world grid with toroidal wrapping."""
        x, y, w, h = bbox
        H, W = values.shape
        ys = np.arange(y, y + h) % H
        xs = np.arange(x, x + w) % W
        return values[np.ix_(ys, xs)].astype(np.float32)

    def _score_object(self, region: np.ndarray) -> float:
        """Slide pattern over region, return minimum distance score."""
        rh, rw = region.shape
        ph, pw = self._ph, self._pw
        if rh < ph or rw < pw:
            return 1.0 if self.metric == 'cosine' else float(self.pattern.size)

        best = float('inf')
        for dy in range(rh - ph + 1):
            for dx in range(rw - pw + 1):
                d = self._dist(region[dy:dy + ph, dx:dx + pw])
                if d < best:
                    best = d
                    if best == 0.0:
                        return 0.0
        return best

    def find_best_matches(self, model, top_k: int = 5, debug: bool = False) -> list[dict]:
        """
        Score every tracked object and return top_k sorted by score ascending.
        Each result dict: {obj_id, centroid, hamming}  (key kept as 'hamming' for compat)
        """
        if not model.objects:
            if debug:
                print('[PatternMatcher] find_best_matches: model.objects is empty')
            return []

        ph, pw = self._ph, self._pw
        pad_y, pad_x = ph // 2, pw // 2

        if debug:
            print(f'[PatternMatcher] metric={self.metric}  pattern shape={self.pattern.shape}  '
                  f'scoring {len(model.objects)} objects')

        scores = []
        for obj in model.objects.values():
            if obj.history is not None and obj.history.cycle is not None:
                # Phase-invariant: score every frame in the detected cycle, take min
                score = min(
                    self._score_object(f.astype(np.float32))
                    for f in obj.history.cycle
                )
                if debug:
                    print(f'  obj_id={obj.id}  cycle_len={len(obj.history.cycle)}  score={score:.4f}')
            else:
                bx, by, bw, bh = obj.bbox
                ex, ey = bx - pad_x, by - pad_y
                ew, eh = bw + pw,    bh + ph
                region = self._extract_region(model._values, (ex, ey, ew, eh))
                score  = self._score_object(region)
                if debug:
                    print(f'  obj_id={obj.id}  bbox={obj.bbox}  '
                          f'region_shape={region.shape}  score={score:.4f}')
            scores.append({'obj_id': obj.id, 'centroid': obj.centroid, 'hamming': score})

        scores.sort(key=lambda s: s['hamming'])
        return scores[:top_k]

