import numpy as np
import cv2
import rerun as rr
import rerun.blueprint as rrb


class RerunViz:
    """
    Rerun-based inspection panel for the AGI simulation.

    Runs alongside OpenCV — call log_frame() each simulation step.
    Opens the Rerun viewer automatically as a companion window.

    Panels provided:
      world/state          — full world grid (image)
      agent/sensory_input  — agent perception patch (image)
      model/map            — internal world model: known/unknown/alive/dead (image)
      saliency/motion      — motion EMA heatmap (image)
      saliency/static      — static Gaussian saliency heatmap (image)
      objects/tracked      — 2-D bounding boxes of all tracked objects
      metrics/coverage     — % of world cells observed (time series)
      metrics/object_count — number of live tracked objects (time series)
      policy/state         — policy name, phase, coverage as a text log
      pattern/match_*      — per-object pattern match scores (time series)
    """

    def __init__(self, app_id: str = 'agi_sim'):
        rr.init(app_id)

        blueprint = rrb.Blueprint(
            rrb.Vertical(
                # Top row: three main views at equal width
                rrb.Horizontal(
                    rrb.Spatial2DView(name='World State',    origin='world/state'),
                    rrb.Spatial2DView(name='Observation',   origin='agent/sensory_input'),
                    rrb.Spatial2DView(name='World Map',     origin='model/map'),
                    column_shares=[1, 1, 1],
                ),
                # Middle row: saliency pair + object details tabs
                rrb.Horizontal(
                    rrb.Spatial2DView(name='Motion Saliency', origin='saliency/motion'),
                    rrb.Spatial2DView(name='Static Saliency', origin='saliency/static'),
                    rrb.Tabs(
                        rrb.TextDocumentView(name='Debug',    origin='debug/info'),
                        rrb.TextLogView(name='Policy Log',   origin='policy/state'),
                        rrb.TextDocumentView(name='Objects', origin='objects'),
                        name='Info',
                    ),
                    column_shares=[1, 1, 1],
                ),
                # Bottom row: time-series metrics
                rrb.Horizontal(
                    rrb.TimeSeriesView(name='Coverage %',     origin='metrics/coverage'),
                    rrb.TimeSeriesView(name='Object Count',   origin='metrics/object_count'),
                    column_shares=[1, 1],
                ),
                row_shares=[3, 2, 1],
            ),
            collapse_panels=True,
        )

        rr.spawn()
        rr.send_blueprint(blueprint, make_active=True, make_default=True)

        # Configure time-series display (logged once as static metadata)
        rr.log('metrics/coverage',     rr.SeriesLine(name='Coverage %',   color=(100, 200, 100)), static=True)
        rr.log('metrics/object_count', rr.SeriesLine(name='Object count', color=(100, 180, 255)), static=True)

    # ------------------------------------------------------------------
    # Main entry point called each simulation step
    # ------------------------------------------------------------------

    def log_frame(self, world_time: int, world, agent, model,
                  saliency=None, policy=None, matches: list | None = None):
        rr.set_time_sequence('step', world_time)

        self._log_world(world, agent)
        self._log_sensory(agent)
        self._log_model(model, agent)
        self._log_scalars(model)

        if saliency is not None:
            self._log_saliency(saliency)

        if model.objects:
            self._log_objects(model)

        if policy is not None:
            self._log_policy(policy, model, agent, world_time)

        if matches:
            self._log_matches(matches)

    # ------------------------------------------------------------------
    # Per-panel helpers
    # ------------------------------------------------------------------

    def _log_world(self, world, agent):
        rgb = np.zeros((world.height, world.width, 3), dtype=np.uint8)
        rgb[world.state == 1] = (255, 255, 255)
        if agent is not None:
            ay = agent.position[1] % world.height
            ax = agent.position[0] % world.width
            rgb[ay, ax] = (255, 0, 0)
        rr.log('world/state', rr.Image(rgb))

    def _log_sensory(self, agent):
        perception = agent.perceive()
        r = agent.perception_radius
        rgb = np.zeros((perception.shape[0], perception.shape[1], 3), dtype=np.uint8)
        rgb[perception == 1] = (255, 255, 255)
        rgb[r, r] = (255, 0, 0)  # agent centre
        rr.log('agent/sensory_input', rr.Image(rgb))

    def _log_model(self, model, agent):
        rgb = np.full((model.height, model.width, 3), (128, 128, 128), dtype=np.uint8)
        rgb[model._known & (model._values == 0)] = (20, 20, 20)
        rgb[model._known & (model._values == 1)] = (255, 255, 255)
        if agent is not None:
            ay = agent.position[1] % model.height
            ax = agent.position[0] % model.width
            rgb[ay, ax] = (255, 0, 0)
        rr.log('model/map', rr.Image(rgb))

    def _log_scalars(self, model):
        rr.log('metrics/coverage',     rr.Scalar(model.coverage * 100))
        rr.log('metrics/object_count', rr.Scalar(len(model.objects)))

    def _log_saliency(self, saliency):
        motion = saliency.get_motion_saliency()
        static = saliency.get_static_saliency()
        rr.log('saliency/motion', rr.Image(self._heatmap_rgb(motion, cv2.COLORMAP_HOT)))
        rr.log('saliency/static', rr.Image(self._heatmap_rgb(static, cv2.COLORMAP_WINTER)))

    def _log_objects(self, model):
        mins, sizes, labels, colors = [], [], [], []
        for obj in model.objects.values():
            bx, by, bw, bh = obj.bbox
            mins.append([float(bx), float(by)])
            sizes.append([float(bw), float(bh)])
            labels.append(f'id={obj.id} a={obj.area} m={obj.motion_score:.2f}')
            colors.append((0, 200, 80, 255))

            # Per-object snapshot image, cycle tensor, and metadata
            if obj.snapshot is not None:
                rr.log(f'objects/obj_{obj.id}/snapshot', rr.Image(obj.snapshot))

            h = obj.history
            if h is not None and h.cycle is not None:
                cycle_arr = np.stack([f.astype(np.uint8) * 255 for f in h.cycle])  # [P, H, W]
                rr.log(f'objects/obj_{obj.id}/cycle', rr.Tensor(cycle_arr))

            period_str = str(h.period) if (h is not None and h.period is not None) else 'unknown'
            n_frames   = len(h.frames)  if h is not None else 0
            rr.log(f'objects/obj_{obj.id}/info', rr.TextDocument(
                f'id={obj.id}\narea={obj.area}\n'
                f'centroid=({obj.centroid[0]:.1f}, {obj.centroid[1]:.1f})\n'
                f'bbox={obj.bbox}\nmotion={obj.motion_score:.3f}\n'
                f'first_seen={obj.first_seen}  last_seen={obj.last_seen}\n'
                f'period={period_str}  frames_recorded={n_frames}'
            ))

        rr.log('objects/tracked', rr.Boxes2D(mins=mins, sizes=sizes,
                                              labels=labels, colors=colors))

    def _log_policy(self, policy, model, agent=None, step: int = 0):
        unknown  = int((~model._known).sum())
        coverage = model._known.mean() * 100
        phase    = 'A – explore' if unknown > 0 else 'B – seek'
        rr.log('policy/state', rr.TextLog(
            f'policy={policy.name}  phase={phase}  '
            f'coverage={coverage:.1f}%  unknown={unknown}  objects={len(model.objects)}',
            level=rr.TextLogLevel.INFO,
        ))

        pos_str = f'({agent.position[0]}, {agent.position[1]})' if agent is not None else 'n/a'
        lines = [
            f'## Debug Info',
            f'',
            f'| Field       | Value |',
            f'|-------------|-------|',
            f'| **Step**    | {step} |',
            f'| **Policy**  | `{policy.name}` |',
            f'| **Phase**   | {phase} |',
            f'| **Position**| {pos_str} |',
            f'| **Coverage**| {coverage:.1f} % |',
            f'| **Unknown** | {unknown} cells |',
            f'| **Objects** | {len(model.objects)} |',
        ]
        rr.log('debug/info', rr.TextDocument('\n'.join(lines), media_type='text/markdown'))

    def _log_matches(self, matches: list):
        for m in matches:
            rr.log(f'pattern/match_obj{m["obj_id"]}',
                   rr.Scalar(float(m['hamming'])))

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _heatmap_rgb(arr: np.ndarray, colormap: int) -> np.ndarray:
        """Apply an OpenCV colormap and return an RGB uint8 image."""
        mn, mx = arr.min(), arr.max()
        norm = ((arr - mn) / (mx - mn) * 255).astype(np.uint8) if mx > mn \
               else np.zeros_like(arr, dtype=np.uint8)
        return cv2.cvtColor(cv2.applyColorMap(norm, colormap), cv2.COLOR_BGR2RGB)
