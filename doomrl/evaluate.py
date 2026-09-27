"""Scoring: `doomrl eval <scenario>`.

Replays a saved model for N episodes with deterministic actions and prints the
mean and standard deviation. These are the numbers in the readme -- deliberately
re-measured here rather than read off the training logs, because `best_model` is
picked as the max over many noisy evaluations and so is biased upward.
"""

from pathlib import Path

from stable_baselines3.common.evaluation import evaluate_policy

from doomrl.auxiliary import algo_for
from doomrl.vec import make_vec


def evaluate(scenario: str, model_path: str | None = None, episodes: int = 20):
    model_path = model_path or str(Path("runs") / scenario / "best_model.zip")
    venv = make_vec(scenario, n_envs=1, subproc=False)
    model = algo_for(scenario).load(model_path)
    mean, std = evaluate_policy(
        model, venv, n_eval_episodes=episodes, deterministic=True
    )
    print(f"{scenario}: {mean:.2f} +/- {std:.2f} over {episodes} episodes")
    venv.close()
    return mean, std