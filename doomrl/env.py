"""One Gymnasium environment that serves all eight scenarios.

Wraps a ViZDoom game. Reads the button list and episode timeout from the
scenario's bundled .cfg rather than hardcoding them, turns each frame into a
small greyscale image, and maps a MultiDiscrete action onto Doom's button array.

The flow per step: agent picks an action -> `_to_buttons` turns it into Doom's
on/off array -> the game advances `frame_skip` tics -> `_observe` shrinks the
new frame to 100x160 grey.
"""

from __future__ import annotations

import os

import cv2
import gymnasium as gym
import numpy as np
import vizdoom as vzd
from gymnasium import spaces

from doomrl.scenarios import SCENARIOS

# Each SubprocVecEnv worker imports this module. OpenCV otherwise spins up a
# thread pool per process (12 on this box), so N workers oversubscribe the CPU
# by ~Nx and throughput collapses into scheduler thrash. The per-frame resize is
# tiny; single-threaded is strictly better here.
cv2.setNumThreads(0)

# Opposed buttons that should never be pressed together. Each becomes one
# MultiDiscrete dimension with a "neither" option at index 0.
BUTTON_GROUPS = (
    (vzd.Button.MOVE_FORWARD, vzd.Button.MOVE_BACKWARD),
    (vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT),
    (vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT),
)


TURN_GROUP = BUTTON_GROUPS[2]


def _build_action_map(buttons, exclusive_attack=False):
    """Map MultiDiscrete dimensions to indices into the scenario's button list.

    Returns a list of lists: entry `d` holds the button indices selectable by
    dimension `d`, where choice 0 always means "press nothing".

    `exclusive_attack` folds the leftover buttons (ATTACK) into the turn group
    instead of giving them their own toggle, so firing and turning become
    mutually exclusive -- the agent has to stop sweeping to shoot. With the two
    independent, defend_center fired 99.9% of its shots mid-turn at 36.7%
    accuracy. Nothing in the reward penalises a miss, so there is no gradient
    discouraging the spray.
    """
    action_map, grouped, turn_dim = [], set(), None
    for group in BUTTON_GROUPS:
        idxs = [buttons.index(b) for b in group if b in buttons]
        if idxs:
            if group is TURN_GROUP:
                turn_dim = len(action_map)
            action_map.append(idxs)
            grouped.update(idxs)
    leftover = [i for i in range(len(buttons)) if i not in grouped]
    if exclusive_attack and turn_dim is not None:
        action_map[turn_dim] = action_map[turn_dim] + leftover
    else:
        # any button not part of an opposed pair is its own on/off toggle
        action_map.extend([i] for i in leftover)
    return action_map


class DoomEnv(gym.Env):
    # "human" lets ViZDoom draw its own window (see set_window_visible
    # below); "rgb_array" hands frames back for the GIF recorder
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 35}

    def __init__(
        self,
        scenario: str,
        render_mode: str | None = None,
        frame_skip: int | None = None,
        width: int = 160,
        height: int = 100,
    ):
        super().__init__()
        spec = SCENARIOS[scenario]

        # ViZDoom setup order matters: load the config, flip any per-scenario
        # toggles, then init(). Nothing can change after init().
        self.game = vzd.DoomGame()
        # the scenario .cfg/.wad files ship inside the vizdoom package itself
        self.game.load_config(os.path.join(vzd.scenarios_path, spec.config))
        self.game.set_window_visible(render_mode == "human")
        self.game.set_render_hud(False)
        self.aux_enemy = spec.aux_enemy
        if self.aux_enemy:
            # per-pixel object ids; used only to derive a supervised target, never
            # fed to the policy -- see doomrl/auxiliary.py
            self.game.set_labels_buffer_enabled(True)
        self.game.init()

        self.render_mode = render_mode
        # per-scenario by default; the argument is an override for experiments
        self.frame_skip = spec.frame_skip if frame_skip is None else frame_skip
        self.width, self.height = width, height

        # each .cfg declares its own buttons; the action space is built from them
        buttons = self.game.get_available_buttons()
        self._n_buttons = len(buttons)
        self._action_map = _build_action_map(buttons, spec.exclusive_attack)

        # episode_timeout is in tics; 0 means "no timeout"
        self._timeout_tics = self.game.get_episode_timeout()
        self._tics = 0
        self._last_frame = np.zeros((height, width, 1), dtype=np.uint8)

        self._var_names = [
            str(v) for v in self.game.get_available_game_variables()
        ]

        # what the policy sees: a single 100x160 grey frame. vec.py stacks four
        # of these before they reach the network.
        screen_space = spaces.Box(
            low=0, high=255, shape=(height, width, 1), dtype=np.uint8
        )
        if self.aux_enemy:
            # "enemy" rides along in the observation dict purely so SB3's rollout
            # buffer carries it to the update; the feature extractor consumes
            # only "screen", so the policy never sees it at inference
            self.observation_space = spaces.Dict(
                {
                    "screen": screen_space,
                    "enemy": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
                }
            )
        else:
            self.observation_space = screen_space
        self._last_enemy = np.zeros((1,), dtype=np.uint8)
        # one dimension per button group, each sized +1 for "press nothing"
        self.action_space = spaces.MultiDiscrete(
            [len(group) + 1 for group in self._action_map]
        )

    # --- Gymnasium API ----------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.game.set_seed(seed)   # must precede new_episode()
        self.game.new_episode()
        self._tics = 0
        obs = self._observe()
        return obs, self._info()

    def step(self, action):
        reward = self.game.make_action(self._to_buttons(action), self.frame_skip)
        self._tics += self.frame_skip

        done = self.game.is_episode_finished()
        # timeout is truncation, not termination — see "why this matters" below
        truncated = done and self._timeout_tics > 0 and self._tics >= self._timeout_tics
        terminated = done and not truncated

        return self._observe(), reward, terminated, truncated, self._info()

    def render(self):
        state = self.game.get_state()
        if state is None:
            return None
        return np.moveaxis(state.screen_buffer, 0, -1)

    def close(self):
        self.game.close()

    # --- internals --------------------------------------------------------

    def _observe(self):
        state = self.game.get_state()
        if state is not None:
            # episode over: reuse the last real frame rather than zeros, so a
            # truncated episode bootstraps from something meaningful
            self._last_frame = self._preprocess(state.screen_buffer)
            if self.aux_enemy:
                self._last_enemy = self._enemy_aimed(state)
        if self.aux_enemy:
            return {"screen": self._last_frame, "enemy": self._last_enemy}
        return self._last_frame

    # crosshair sits at screen centre; how wide a band counts as "lined up"
    AIM_BAND = 0.04

    @classmethod
    def _enemy_aimed(cls, state):
        """1 if an enemy occupies the central band of the screen.

        Arnold's auxiliary target is "is an enemy visible", which works in
        deathmatch where enemies are often off-screen. In defend_the_center a
        monster is visible on 100% of steps, so that target is constant and
        teaches nothing. "Is an enemy under the crosshair" is positive on ~54%
        of steps under a random policy -- well balanced, and it is the signal
        the firing decision actually needs.
        """
        values = [
            label.value
            for label in (state.labels or ())
            if label.object_name != "DoomPlayer"
        ]
        if not values or state.labels_buffer is None:
            return np.zeros((1,), dtype=np.uint8)
        mask = np.isin(state.labels_buffer, values)
        width = mask.shape[1]
        lo = int(width * (0.5 - cls.AIM_BAND))
        hi = int(width * (0.5 + cls.AIM_BAND))
        return np.array([int(mask[:, lo:hi].any())], dtype=np.uint8)

    def _preprocess(self, screen):
        grey = cv2.cvtColor(np.moveaxis(screen, 0, -1), cv2.COLOR_BGR2GRAY)
        # INTER_AREA is the correct filter for downscaling (INTER_CUBIC rings)
        resized = cv2.resize(
            grey, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        return resized.reshape(self.height, self.width, 1).astype(np.uint8)

    def _info(self):
        state = self.game.get_state()
        if state is None or state.game_variables is None:
            # predict_position declares no available_game_variables, and ViZDoom
            # returns None there rather than an empty array
            return {}
        return dict(zip(self._var_names, state.game_variables))

    def _to_buttons(self, action):
        """MultiDiscrete choice -> the 0/1 array ViZDoom expects."""
        buttons = [0] * self._n_buttons
        for dim, idxs in enumerate(self._action_map):
            choice = int(action[dim])
            if choice > 0:  # 0 == press nothing
                buttons[idxs[choice - 1]] = 1
        return buttons

def make_env(scenario: str, render_mode=None, seed=None):
    """Zero-argument constructor for one env, which is what SubprocVecEnv
    needs to build the game inside each worker process."""
    def _init():
        env = DoomEnv(scenario, render_mode=render_mode)
        env.reset(seed=seed)
        return env
    return _init