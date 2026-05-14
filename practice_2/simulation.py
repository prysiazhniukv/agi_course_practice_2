import time
import queue
from pynput import keyboard as kb

from world import World
from agent import Agent
from saliency import SaliencyMap
from model import WorldModel
from action import POLICIES, PatternSeekPolicy
from patterns import load_pattern, PatternMatcher
from viz_rerun import RerunViz


_CONTROLS = """\
Controls (type anywhere — global keyboard capture):
  w/s/a/d   - Move Up / Down / Left / Right  [keyboard mode]
  SPACE     - Pause / Resume
  + / -     - Speed up / slow down
  e         - Toggle cell at agent position (0 <-> 1)
  z / x     - Kill / Revive cell at agent position
  t         - Toggle keyboard / auto policy
  p         - Cycle policy  (auto mode only)
  q / ESC   - Quit"""


class Simulation:
    def __init__(self, world_size=(50, 50), start_pos=None, perception_radius=9,
                 warmup=400, pattern_file=None, pattern_metric='cosine',
                 use_rerun=True):
        self.world = World(*world_size)
        if start_pos is None:
            start_pos = (world_size[0] // 2, world_size[1] // 2)
        self.agent = Agent(self.world, start_pos, perception_radius)
        saliency          = SaliencyMap(2 * perception_radius + 1)
        self.model        = WorldModel(*world_size, saliency=saliency)
        self.saliency     = saliency
        self._policies    = list(POLICIES)
        self._policy_idx  = 0
        self.policy       = self._policies[self._policy_idx]
        self.auto_control = False
        self._quit        = False

        if pattern_file:
            try:
                pattern, name = load_pattern(pattern_file)
                seek_policy = PatternSeekPolicy(PatternMatcher(pattern, name, metric=pattern_metric))
                self._policies.append(seek_policy)
                self._policy_idx  = len(self._policies) - 1
                # self.policy       = seek_policy
                # self.auto_control = True
                print(f'Pattern loaded: {name}  ({pattern.shape[0]}x{pattern.shape[1]})')
                print(f'Control mode: AUTO ({self.policy.name})')
            except FileNotFoundError:
                print(f'Warning: pattern file not found: {pattern_file}')

        self._rr = RerunViz() if use_rerun else None
        self.paused = False
        self.frames_per_step = 1
        self._frame_count = 0

        print(f"Running {warmup} warm-up steps...")
        for _ in range(warmup):
            self.world.step()
        print(f"Done. Starting at step {self.world.time_step}.")

    def _handle_key(self, key: str) -> bool:
        """Process a single key string. Returns False to request quit."""
        if key in ('q', '\x1b'):   # q or ESC
            return False
        elif key == ' ':
            self.paused = not self.paused
        elif key in ('+', '='):
            self.frames_per_step = max(1, self.frames_per_step - 1)
        elif key == '-':
            self.frames_per_step = min(30, self.frames_per_step + 1)
        elif key == 't':
            self.auto_control = not self.auto_control
            mode = f'AUTO ({self.policy.name})' if self.auto_control else 'KEYBOARD'
            print(f'Control mode: {mode}')
        elif key == 'p':
            self._policy_idx = (self._policy_idx + 1) % len(self._policies)
            self.policy = self._policies[self._policy_idx]
            print(f'Policy: {self.policy.name}')
        elif key in ('w', 's', 'a', 'd') and not self.auto_control:
            self.agent.move({'w': 'up', 's': 'down', 'a': 'left', 'd': 'right'}[key])
        elif key == 'e':
            self.agent.toggle_cell()
        elif key == 'z':
            self.agent.set_cell(0)
        elif key == 'x':
            self.agent.set_cell(1)
        return True

    def run(self):
        print(_CONTROLS)
        key_q: queue.SimpleQueue = queue.SimpleQueue()

        def on_press(key):
            try:
                key_q.put(key.char)
            except AttributeError:
                if key == kb.Key.esc:
                    key_q.put('\x1b')
                elif key == kb.Key.space:
                    key_q.put(' ')

        listener = kb.Listener(on_press=on_press)
        listener.daemon = True
        listener.start()

        try:
            while not self._quit:
                # Drain pending keypresses
                while True:
                    try:
                        k = key_q.get_nowait()
                        if not self._handle_key(k):
                            self._quit = True
                            break
                    except queue.Empty:
                        break

                if self._quit:
                    break

                if not self.paused:
                    self._frame_count += 1
                    if self._frame_count >= self.frames_per_step:
                        self.world.step()
                        self._frame_count = 0

                self.model.update(self.agent.perceive(), self.agent.position,
                                  self.agent.perception_radius, self.world.time_step)

                if self.auto_control and not self.paused:
                    direction = self.policy(self.agent, self.model)
                    if direction:
                        self.agent.move(direction)

                if self._rr is not None:
                    self._rr.log_frame(self.world.time_step, self.world, self.agent,
                                       self.model, saliency=self.saliency,
                                       policy=self.policy)

                time.sleep(0.05)
        finally:
            listener.stop()
