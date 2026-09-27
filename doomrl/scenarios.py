"""Everything that differs between the eight scenarios, in one table.

`env.py`, `vec.py` and `train.py` all read from here and contain no per-scenario
branches, so adding a scenario means adding one line at the bottom and nothing
else. Each field below is a knob that some scenario needed and the rest didn't.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    name: str
    config: str          # filename inside vizdoom's bundled scenarios dir
    timesteps: int
    # Tics advanced per agent step. This is the aiming knob: one TURN action
    # rotates 1.8 deg per tic, so frame_skip=4 moves the crosshair 7.0 deg at a
    # time while the 100x160 observation resolves ~0.56 deg/pixel -- the action
    # is ~12x coarser than what the agent can see. defend_center fires all 26
    # rounds every episode and lands 31.5% of them; every hit is a kill, so its
    # score is just accuracy x 26. Halving the skip halves the overshoot.
    frame_skip: int = 4
    # Fold ATTACK into the turn group so firing and turning are mutually
    # exclusive. See _build_action_map.
    exclusive_attack: bool = False
    # Stacked frames. predict_position gives the agent one rocket and it fires
    # at exactly the step its stack first fills, so the stack depth and the
    # frame skip together decide how much motion it has seen -- and how many
    # aiming decisions it has made -- before it commits.
    n_stack: int = 4
    # Train an auxiliary "is an enemy on screen" head off the labels buffer,
    # after Lample & Chaplot's Arnold. Gives the CNN a dense per-step supervised
    # signal instead of only the sparse kill reward. Target only -- the policy
    # still sees pixels alone.
    aux_enemy: bool = False
    # Episodes per periodic evaluation. 10 is fine where reward is continuous,
    # but predict_position's is one bit per episode, so SE is ~0.155 at n=10 --
    # best_model then selects on noise (a 0.95 checkpoint evaluated at 0.65).
    # Raise it where episodes are short and the reward is near-Bernoulli.
    eval_episodes: int = 10
    # Stop once evaluation stops improving. Disable where the improvement rate
    # is slow relative to evaluation noise: predict_position gains ~0.157 per 1M
    # steps against an eval SE of 0.089, so no short-window rule can tell "still
    # climbing" from "plateaued" -- "no new best" truncated it at 1.5M of 4M
    # while the overall trend was significant at t=+4.2, and a slope-based rule
    # replayed against the saved curve stopped it even earlier, at 600k.
    early_stop: bool = True


SCENARIOS = {
    s.name: s
    for s in [
        Scenario("basic",            "basic.cfg",             200_000),
        Scenario("defend_line",      "defend_the_line.cfg",   600_000),
        Scenario("health_gathering", "health_gathering.cfg", 600_000),
        # finer turn granularity: 3.5 deg per action instead of 7.0, which took
        # accuracy 31.5% -> 36.7%. Tried on predict_position too and it made no
        # difference (0.58 -> 0.52, inside one standard error), so that one
        # stays at 4.
        Scenario("defend_center",    "defend_the_center.cfg", 1_000_000,
                 frame_skip=2, exclusive_attack=True, aux_enemy=True),
        Scenario("my_way_home",      "my_way_home.cfg",      1_000_000),
        # skip 2 doubles the aiming decisions (3.5 deg each, not 7.0) and
        # n_stack 8 doubles the motion samples, over the same 16-tic window it
        # already waits out before firing
        # same turn-and-shoot structure as defend_center, and it had the same
        # pathology: 95% of shots were initiated mid-sweep
        Scenario("predict_position", "predict_position.cfg",  4_000_000,
                 frame_skip=2, n_stack=8, exclusive_attack=True, aux_enemy=True,
                 eval_episodes=30, early_stop=False),
        Scenario("take_cover",       "take_cover.cfg",       1_500_000),
        Scenario("deadly_corridor",  "deadly_corridor.cfg",  1_500_000),
    ]
}