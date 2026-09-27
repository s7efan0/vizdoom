"""Training entry point: `doomrl train <scenario>`.

Builds two env stacks (one for learning, one for periodic scoring), constructs
PPO with the hyperparameters below, and runs it. Two callbacks ride along: one
saves checkpoints, the other scores the policy every EVAL_EVERY steps, keeps the
best as `best_model.zip`, and can stop early once scores stop improving.

Everything lands in `runs/<scenario>/`.
"""

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)

from doomrl.scenarios import SCENARIOS
from doomrl.vec import make_vec


def linear_schedule(initial: float):
    """SB3 passes progress_remaining: 1.0 at start -> 0.0 at end."""
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial
    return schedule


# Evaluation cadence, in env steps (divided by n_envs where SB3 counts calls).
EVAL_EVERY = 50_000
# Stop after this many consecutive evals with no new best. At EVAL_EVERY=50k
# that tolerates 400k unproductive steps (~45 min at ~150 steps/s) before
# giving up -- loose enough to ride out a noisy plateau, tight enough that a
# genuinely finished run doesn't burn hours. Lower it to be more aggressive.
NO_IMPROVEMENT_EVALS = 8
MIN_EVALS = 8

PPO_KWARGS = dict(
    learning_rate=linear_schedule(2.5e-4),
    n_steps=128,             # per env; 8 envs -> 1024-step rollout
    batch_size=256,          # 4 minibatches per rollout
    n_epochs=4,              # 10 overfits correlated pixel rollouts
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=linear_schedule(0.1),
    ent_coef=0.01,           # SB3 default is 0.0; the main guard against policy collapse
    vf_coef=0.5,
    max_grad_norm=0.5,
)



def train(scenario: str, timesteps: int | None = None, seed: int = 0, n_envs: int = 8):
    spec = SCENARIOS[scenario]
    timesteps = timesteps or spec.timesteps
    run_dir = Path("runs") / scenario
    run_dir.mkdir(parents=True, exist_ok=True)

    venv = make_vec(scenario, n_envs=n_envs, seed=seed, normalize=True)
    # eval env MUST match the training wrapper stack exactly -- including
    # VecNormalize -- but score raw reward and never update the statistics.
    eval_venv = make_vec(
        scenario,
        n_envs=1,
        seed=seed + 1000,
        subproc=False,
        normalize=True,
        norm_reward=False,
        training=False,
    )

    # checkpoints on a fixed cadence; evaluation keeps best_model.zip and, where
    # the scenario allows it, stops the run once scores stop improving
    callbacks = [
        CheckpointCallback(
            save_freq=max(100_000 // n_envs, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="model",
        ),
        EvalCallback(
            eval_venv,
            best_model_save_path=str(run_dir),
            log_path=str(run_dir / "eval"),
            eval_freq=EVAL_EVERY // n_envs,
            n_eval_episodes=spec.eval_episodes,
            deterministic=True,
            # health_gathering peaked at 100k then spent 500k steps getting
            # worse; this caps that waste at NO_IMPROVEMENT_EVALS * EVAL_EVERY.
            # Off where improvement is slower than the eval noise -- see
            # Scenario.early_stop.
            callback_after_eval=(
                StopTrainingOnNoModelImprovement(
                    max_no_improvement_evals=NO_IMPROVEMENT_EVALS,
                    min_evals=MIN_EVALS,
                    verbose=1,
                )
                if spec.early_stop
                else None
            ),
        ),
    ]

    # aux scenarios need the Dict observation space, a feature extractor that
    # ignores the label, and the PPO subclass that adds the extra loss term
    if spec.aux_enemy:
        from doomrl.auxiliary import AuxPPO, ScreenOnlyExtractor

        model = AuxPPO(
            "MultiInputPolicy",
            venv,
            policy_kwargs=dict(features_extractor_class=ScreenOnlyExtractor),
            tensorboard_log=str(run_dir / "tb"),
            verbose=1,
            seed=seed,
            **PPO_KWARGS,
        )
    else:
        model = PPO(
            "CnnPolicy",
            venv,
            tensorboard_log=str(run_dir / "tb"),
            verbose=1,
            seed=seed,
            **PPO_KWARGS,
        )
    model.learn(timesteps, callback=callbacks)
    model.save(run_dir / "final")
    venv.close()
    eval_venv.close()
