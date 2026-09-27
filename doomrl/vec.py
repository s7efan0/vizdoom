"""Builds the wrapper stack that PPO actually trains against.

Eight copies of the game run in parallel processes, and a chain of wrappers sits
on top. Order matters and is explained inline; the short version is that each
layer adds one thing:

    SubprocVecEnv    8 games in 8 processes
    VecMonitor       records raw episode return/length
    VecFrameStack    (100,160,1) -> (100,160,4), so motion is visible
    VecNormalize     scales rewards during training only
    VecTransposeImage (100,160,4) -> (4,100,160) for PyTorch

`train` and `eval` must build the *same* stack or SB3 asserts when it syncs them.
"""

from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecMonitor,
    VecNormalize,
    VecTransposeImage,
)

from doomrl.env import make_env
from doomrl.scenarios import SCENARIOS

N_ENVS = 8       # 6 physical cores; leaves headroom for the learner


def make_vec(
    scenario: str,
    n_envs: int = N_ENVS,
    n_stack: int | None = None,
    seed: int = 0,
    subproc: bool = True,
    render_mode: str | None = None,
    normalize: bool = False,
    norm_reward: bool = True,
    training: bool = True,
):
    # per-scenario by default; the argument is an override for experiments
    if n_stack is None:
        n_stack = SCENARIOS[scenario].n_stack
    # one factory per worker; each gets its own seed so the 8 games diverge
    fns = [
        make_env(scenario, render_mode=render_mode, seed=seed + i)
        for i in range(n_envs)
    ]
    venv = (
        SubprocVecEnv(fns, start_method="spawn")
        if subproc and n_envs > 1
        else DummyVecEnv(fns)
    )
    venv = VecMonitor(venv)                          # episode stats for tensorboard
    venv = VecFrameStack(venv, n_stack=n_stack)      # (100,160,1) -> (100,160,4)
    if normalize:
        # Required for training, not optional. ViZDoom returns run to +/-100s;
        # CnnPolicy shares its CNN trunk between the policy and value heads, so
        # unscaled value gradients wreck the policy features. VecMonitor sits
        # *inside* this, so tensorboard still logs raw episode returns.
        #
        # The eval env must ALSO be VecNormalize or EvalCallback dies in
        # sync_envs_normalization() -- but with norm_reward=False so it scores
        # raw, and training=False so it never updates the statistics.
        venv = VecNormalize(
            venv,
            norm_obs=False,
            norm_reward=norm_reward,
            clip_reward=10.0,
            training=training,
        )
    # Applied explicitly rather than left to SB3's auto-wrap. SB3 only wraps the
    # *training* env, which leaves the eval env one layer shallower --
    # sync_envs_normalization() walks both stacks in lockstep and then asserts on
    # the mismatch, killing the run at the first eval. Doing it here keeps both
    # stacks identical; SB3 detects it and skips its own wrap.
    return VecTransposeImage(venv)