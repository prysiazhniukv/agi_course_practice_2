import numpy as np
import cv2


class SaliencyMap:
    """
    Motion and static saliency from the agent's perception patch.

    Motion saliency  – exponential moving average of frame-to-frame cell changes.
    Static saliency  – Gaussian density blur of live cells.
    """

    def __init__(self, perception_size, motion_decay=0.6, static_sigma=1.2):
        self.motion_decay = motion_decay
        self.static_sigma = static_sigma
        self._prev_frame: np.ndarray | None = None
        self._motion_ema = np.zeros((perception_size, perception_size), dtype=np.float32)
        self._static_map = np.zeros((perception_size, perception_size), dtype=np.float32)

    def update(self, perception: np.ndarray):
        frame = perception.astype(np.float32)
        if self._prev_frame is not None:
            diff = np.abs(frame - self._prev_frame)
            self._motion_ema = self.motion_decay * self._motion_ema + (1.0 - self.motion_decay) * diff
        self._prev_frame = frame.copy()

        k = max(3, int(self.static_sigma * 6) | 1)
        self._static_map = cv2.GaussianBlur(frame, (k, k), self.static_sigma)

    def get_motion_saliency(self) -> np.ndarray:
        return self._motion_ema.copy()

    def get_static_saliency(self) -> np.ndarray:
        return self._static_map.copy()

    def segment(self, threshold: float = 0.15) -> list[dict]:
        """
        Find blobs in the static saliency map via connected components.

        Returns a list of dicts (one per blob, background excluded):
            centroid_local  – (x, y) float in patch coordinates
            area            – number of pixels above threshold
            bbox_local      – (x, y, w, h) in patch coordinates
            motion_score    – mean motion EMA inside the blob mask
        """
        mx = self._static_map.max()
        if mx == 0:
            return []

        binary = (self._static_map >= threshold * mx).astype(np.uint8)
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

        blobs = []
        for label in range(1, n_labels):          # 0 is background
            mask = labels == label
            blobs.append({
                'centroid_local': (float(centroids[label][0]), float(centroids[label][1])),
                'area':           int(stats[label, cv2.CC_STAT_AREA]),
                'bbox_local':     (
                    int(stats[label, cv2.CC_STAT_LEFT]),
                    int(stats[label, cv2.CC_STAT_TOP]),
                    int(stats[label, cv2.CC_STAT_WIDTH]),
                    int(stats[label, cv2.CC_STAT_HEIGHT]),
                ),
                'motion_score':   float(self._motion_ema[mask].mean()),
            })
        return blobs

    @staticmethod
    def _to_heatmap(arr: np.ndarray, colormap: int) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            normalised = ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            normalised = np.zeros_like(arr, dtype=np.uint8)
        return cv2.applyColorMap(normalised, colormap)

    def visualize(self, cell_size: int = 20) -> np.ndarray:
        motion_img = self._to_heatmap(self._motion_ema, cv2.COLORMAP_HOT)
        static_img = self._to_heatmap(self._static_map, cv2.COLORMAP_WINTER)

        def scale_up(img):
            return np.repeat(np.repeat(img, cell_size, axis=0), cell_size, axis=1)

        motion_big = scale_up(motion_img)
        static_big = scale_up(static_img)

        bar_h, w = 22, motion_big.shape[1]

        def label_bar(text, color):
            bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
            cv2.putText(bar, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            return bar

        return np.hstack([
            np.vstack([label_bar("Motion Saliency", (160, 160, 255)), motion_big]),
            np.vstack([label_bar("Static Saliency",  (160, 255, 160)), static_big]),
        ])
