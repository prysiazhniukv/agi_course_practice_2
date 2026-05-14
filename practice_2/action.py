import numpy as np


_DELTAS = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}


class Policy:
    """Base class for agent policies. Subclasses implement __call__."""

    def __call__(self, agent, model) -> str | None:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return type(self).__name__


class GreedyExplorerPolicy(Policy):
    """
    One-step greedy lookahead: picks the direction that reveals the most
    unknown cells in the next perception patch.
    Freezes when no immediate step uncovers anything new.
    """

    def __call__(self, agent, model) -> str | None:
        r = agent.perception_radius
        w, h = model.width, model.height
        x, y = agent.position

        best_dir, best_score = None, 0
        for direction, (dx, dy) in _DELTAS.items():
            nx, ny = (x + dx) % w, (y + dy) % h
            ys = np.arange(ny - r, ny + r + 1) % h
            xs = np.arange(nx - r, nx + r + 1) % w
            score = int((~model._known[np.ix_(ys, xs)]).sum())
            if score > best_score:
                best_score, best_dir = score, direction

        return best_dir  # None = no gain in any direction → stay


class NearestUnknownPolicy(Policy):
    """
    Navigates toward the globally nearest undiscovered cell
    using toroidal Manhattan distance. Works even when unknowns
    are far outside the immediate perception window.
    Returns None only when the entire map has been observed.
    """

    def __call__(self, agent, model) -> str | None:
        w, h = model.width, model.height
        x, y = agent.position

        unknown_yx = np.argwhere(~model._known)
        if len(unknown_yx) == 0:
            return None

        dy = unknown_yx[:, 0] - y
        dx = unknown_yx[:, 1] - x
        dy = np.where(np.abs(dy) > h / 2, dy - np.sign(dy) * h, dy)
        dx = np.where(np.abs(dx) > w / 2, dx - np.sign(dx) * w, dx)

        nearest = np.argmin(np.abs(dx) + np.abs(dy))
        target_dy, target_dx = int(dy[nearest]), int(dx[nearest])

        if abs(target_dx) >= abs(target_dy):
            return 'right' if target_dx > 0 else 'left'
        else:
            return 'down' if target_dy > 0 else 'up'


POLICIES = [GreedyExplorerPolicy(), NearestUnknownPolicy()]


class PatternSeekPolicy(Policy):
    """
    Two-phase policy:
      Phase A — world not fully explored: delegates to NearestUnknownPolicy.
      Phase B — world fully known: uses PatternMatcher to find the best-matching
                object and navigates to it. Among objects with equal (minimum)
                Hamming distance, picks the one toroidally nearest to the agent.
    """

    def __init__(self, matcher, debug_every: int = 60):
        self._matcher     = matcher
        self._explorer    = NearestUnknownPolicy()
        self._debug_every = debug_every
        self._call_count  = 0

    @property
    def name(self) -> str:
        return f"PatternSeek({self._matcher.pattern_name})"

    def _navigate_to(self, agent, target_centroid, w, h) -> str | None:
        """Return a cardinal direction that reduces toroidal distance to target."""
        x, y  = agent.position
        tx, ty = float(target_centroid[0]), float(target_centroid[1])
        dx = tx - x
        dy = ty - y
        if abs(dx) > w / 2:
            dx -= np.sign(dx) * w
        if abs(dy) > h / 2:
            dy -= np.sign(dy) * h
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            return None   # already there
        if abs(dx) >= abs(dy):
            return 'right' if dx > 0 else 'left'
        else:
            return 'down' if dy > 0 else 'up'

    def _debug(self, msg: str):
        print(f'[PatternSeek] {msg}')

    def __call__(self, agent, model) -> str | None:
        self._call_count += 1
        verbose = (self._call_count % self._debug_every == 1)

        # Phase A: explore until everything is known
        if not model._known.all():
            unknown_count = int((~model._known).sum())
            coverage = model._known.mean() * 100
            if verbose:
                self._debug(f'Phase A — exploring. coverage={coverage:.1f}%  unknown={unknown_count}')
            return self._explorer(agent, model)

        if verbose:
            self._debug(f'Phase B — world fully known. tracked objects={len(model.objects)}')

        # Phase B: find best-matching object
        matches = self._matcher.find_best_matches(model, debug=verbose)
        if not matches:
            if verbose:
                self._debug('Phase B — no matches returned (no tracked objects?). Staying.')
            return None

        best_hamming = matches[0]['hamming']
        candidates   = [m for m in matches if m['hamming'] == best_hamming]

        w, h = model.width, model.height
        x, y = agent.position

        def toroidal_dist(c):
            dx = abs(c[0] - x); dy = abs(c[1] - y)
            return min(dx, w - dx) + min(dy, h - dy)

        target    = min(candidates, key=lambda m: toroidal_dist(m['centroid']))
        direction = self._navigate_to(agent, target['centroid'], w, h)

        if verbose:
            self._debug(
                f'Phase B — best hamming={best_hamming}  candidates={len(candidates)}  '
                f'target_centroid={target["centroid"]}  agent={agent.position}  '
                f'direction={direction}'
            )
        return direction

