"""Watch a trained policy play in ViZDoom's own window, at real speed.

`record` renders to a GIF; this renders to the screen and paces itself so the
game runs at the speed a person would play it. Nothing is written to disk.
"""
import random
import time
from pathlib import Path

from doomrl.auxiliary import algo_for
from doomrl.scenarios import SCENARIOS
from doomrl.vec import make_vec

DOOM_TICS_PER_SECOND = 35
EPISODES = 5


def play(
    scenario: str,
    model_path: str | None = None,
    episodes: int = EPISODES,
    speed: float = 1.0,
    seed: int | None = None,
    stochastic: bool = False,
):
    spec = SCENARIOS[scenario]
    # A demo is a handful of episodes, and ViZDoom replays the same one for
    # the first several at seed 0 -- `basic` opens with seven identical 95s.
    # Draw a seed instead and print it, so a good run can be replayed.
    if seed is None:
        seed = random.randrange(2**31)
    model_path = model_path or str(Path("runs") / scenario / "best_model.zip")

    # subproc=False keeps the game in this process, so the window belongs to us.
    # normalize stays off: reward scaling is a training-time device, and the
    # scores printed here should be the ones in the readme.
    venv = make_vec(
        scenario, n_envs=1, subproc=False, render_mode="human", seed=seed
    )
    model = algo_for(scenario).load(model_path)

    # One agent step advances frame_skip tics, and Doom runs at 35 tics/s. In
    # PLAYER mode the game only advances when we ask it to, so without this it
    # plays back as fast as the policy can predict -- several times real speed.
    step_seconds = spec.frame_skip / DOOM_TICS_PER_SECOND / speed

    print(f"{scenario}: {model_path}, {episodes} episodes at {speed}x speed, seed {seed}")
    print("ctrl-c to stop\n")

    scores, finished = [], 0
    obs = venv.reset()
    deadline = time.perf_counter()
    try:
        while finished < episodes:
            action, _ = model.predict(obs, deterministic=not stochastic)
            obs, _, dones, infos = venv.step(action)

            if dones[0]:
                finished += 1
                # VecMonitor sits below the frame stack and records the raw
                # episode return, which is what `eval` reports too
                episode = infos[0].get("episode")
                if episode is not None:
                    scores.append(episode["r"])
                    print(
                        f"  episode {finished}/{episodes}: "
                        f"{episode['r']:.2f} in {episode['l']} steps"
                    )

            deadline += step_seconds
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            else:
                # fell behind (slow predict, or the window was dragged); reset
                # the clock rather than sprinting to catch up
                deadline = time.perf_counter()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        venv.close()

    if scores:
        mean = sum(scores) / len(scores)
        print(f"\n{scenario}: {mean:.2f} over {len(scores)} episodes")
    return scores
