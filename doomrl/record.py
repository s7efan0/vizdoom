"""GIF capture: `doomrl record <scenario>`.

Runs the policy with the game rendering off-screen, collects frames, and writes
`media/<scenario>.gif` at real-time speed. Use `play` instead to just watch it.
"""

from pathlib import Path

import cv2
import imageio
import numpy as np
from doomrl.auxiliary import algo_for

from doomrl.scenarios import SCENARIOS
from doomrl.vec import make_vec


# Doom runs at 35 tics/s and one agent step advances frame_skip tics, so
# real-time playback is 35/frame_skip fps -- 9 at skip 4, 18 at skip 2.
DOOM_TICS_PER_SECOND = 35
# Frames, not episodes, is the real limit: health_gathering survives ~525 steps
# per episode while a solved `basic` finishes in 2. So `episodes` is a generous
# upper bound and MAX_FRAMES does the actual capping -- ~13s of playback.
MAX_FRAMES = 120
MAX_EPISODES = 50
# Doom renders through a native 256-colour palette, which is why full-res frames
# compress so well as GIF. Downscale with INTER_NEAREST: any interpolating
# filter invents off-palette colours and the file gets *bigger* -- measured
# 240x180 at 4820KB via INTER_AREA vs 2508KB via INTER_NEAREST, against 4058KB
# for untouched 320x240.
SCALE = 0.75


def record(
    scenario: str,
    model_path: str | None = None,
    episodes: int = MAX_EPISODES,
    fps: int | None = None,
    max_frames: int = MAX_FRAMES,
    scale: float = SCALE,
):
    fps = fps or round(DOOM_TICS_PER_SECOND / SCENARIOS[scenario].frame_skip)
    model_path = model_path or str(Path("runs") / scenario / "best_model.zip")
    venv = make_vec(scenario, n_envs=1, subproc=False, render_mode="rgb_array")
    model = algo_for(scenario).load(model_path)

    frames, finished = [], 0
    obs = venv.reset()
    while finished < episodes and len(frames) < max_frames:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = venv.step(action)
        frame = venv.env_method("render")[0]
        if frame is not None:
            frames.append(frame)
        if dones[0]:
            finished += 1

    # one flat media/ dir, one GIF per scenario, so the readme can link them
    # by name without a directory walk
    out = Path("media") / f"{scenario}.gif"
    out.parent.mkdir(exist_ok=True)
    # no frame subsampling: a solved agent finishes `basic` in ~2 steps, and
    # dropping every other frame left a 3-frame GIF
    if scale != 1.0:
        h, w = frames[0].shape[:2]
        size = (int(w * scale), int(h * scale))
        frames = [
            cv2.resize(f, size, interpolation=cv2.INTER_NEAREST) for f in frames
        ]
    imageio.mimsave(out, [np.asarray(f) for f in frames], fps=fps)
    kb = out.stat().st_size / 1024
    print(f"wrote {out} ({len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]}, {kb:.0f} KB)")
    venv.close()