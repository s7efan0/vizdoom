# Notes — how this repo actually works

Informal notes for me. The readme is the write-up; this is the map.

## The one idea

There is **one** environment and **one** training script. All eight scenarios go
through the same code. Everything that differs between them is a row in
`scenarios.py`, and nothing anywhere else says `if scenario == "basic"`.

So: **want to change how a scenario behaves? Edit `scenarios.py`. That's it.**

```python
Scenario("defend_center", "defend_the_center.cfg", 1_000_000,
         frame_skip=2, exclusive_attack=True, aux_enemy=True)
#         ^ name      ^ vizdoom cfg    ^ how long to train    ^ the knobs
```

## What happens on one step

```
 policy picks an action        e.g. [2, 0, 1]  (MultiDiscrete)
        |
        v  env.py  _to_buttons()
 Doom button array             [0,1,0,0,1]     (which keys are held)
        |
        v  game.make_action(buttons, frame_skip)
 game advances frame_skip tics                 (4 tics = ~114 ms)
        |
        v  env.py  _observe() -> _preprocess()
 320x240 colour -> 100x160 grey
        |
        v  vec.py  VecFrameStack
 last 4 frames stacked -> (4,100,160)          so motion is visible
        |
        v
 CNN -> 512 features -> two heads: action probabilities, and state value
```

## Files, in the order they matter

| file | what it is |
|---|---|
| `scenarios.py` | the settings table. **Start here.** |
| `env.py` | one game = one Gymnasium env. Screen → observation, action → buttons. |
| `vec.py` | 8 parallel games + the wrapper chain |
| `train.py` | PPO hyperparameters + the training loop + callbacks |
| `auxiliary.py` | the extra supervised head (only 2 scenarios use it) |
| `evaluate.py` | score a saved model, deterministic |
| `record.py` | write a GIF |
| `play.py` | watch it live in a window |
| `cli.py` | argparse, maps subcommands to the above |

## The wrapper stack

This trips people up because order matters:

```
SubprocVecEnv       8 games, 8 OS processes
  VecMonitor        logs raw episode return + length   <- raw, before scaling
    VecFrameStack   1 frame -> 4 stacked
      VecNormalize  divides rewards by a running std   <- training only
        VecTransposeImage   (100,160,4) -> (4,100,160) for PyTorch
```

Two things to not get wrong:

- `VecMonitor` sits **below** `VecNormalize` on purpose. That's why tensorboard
  logs real game scores and not the scaled ones the optimiser sees.
- The **eval env must have the identical stack** to the training env, including
  `VecNormalize` — just with `norm_reward=False, training=False`. SB3 walks both
  stacks in lockstep and asserts if the depths differ. This is what crashed every
  run at the first eval until `VecTransposeImage` was applied explicitly in
  `make_vec` instead of being left to SB3's auto-wrap (which only wraps the
  training env).

## One training run, start to finish

`doomrl train defend_center` →

1. `train.py` reads the `Scenario` row
2. builds the training env (8 procs, normalized) and the eval env (1 proc, raw)
3. picks `AuxPPO` if `aux_enemy` else plain `PPO`
4. `model.learn(timesteps)` — collect 128 steps × 8 envs = 1024 transitions,
   then 4 passes over them in minibatches of 256, then collect again
5. every 50k steps: score on the eval env, save `best_model.zip` if it's a record
6. if 8 evals pass with no new record → stop early (unless `early_stop=False`)
7. output → `runs/defend_center/`: `best_model.zip`, `final.zip`, `checkpoints/`,
   `tb/` (tensorboard), `eval/`

Then `doomrl eval defend_center` re-scores `best_model.zip` from scratch. **The
readme numbers come from this, not from the training logs** — `best_model` is
the max over ~20 noisy evaluations, so it's biased upward. Measuring again is
the honest number.

## The action space, in one paragraph

Doom takes an array of on/off buttons and you can hold several at once. Opposed
buttons (forward/back, left/right, turn L/R) get grouped into one MultiDiscrete
dimension each, with choice 0 = "press nothing". Leftover buttons (usually
ATTACK) become their own on/off dimension. `exclusive_attack=True` instead folds
ATTACK *into the turn group*, so the agent physically cannot turn and shoot in
the same step — that's the fix that took `defend_center` from 8.97 to 16.13.

## The auxiliary head

Only `defend_center` and `predict_position`. A second output off the shared CNN
predicts "is an enemy under the crosshair", trained with BCE against ViZDoom's
labels buffer. The point is to give the convolutions a dense signal instead of
only a sparse +1 per kill.

The label travels inside the observation dict under `"enemy"` — **not** because
the policy sees it, but because that's the only way SB3's rollout buffer carries
per-step data to the update. `ScreenOnlyExtractor` reads `"screen"` and nothing
else.

Consequence: those two scenarios save with `AuxPPO`, so they must **load** with
`AuxPPO` too. That's what `algo_for(scenario)` is for. Loading an AuxPPO
checkpoint with plain PPO fails on the optimiser param-group count.


## Where to change what

| I want to… | edit |
|---|---|
| train longer / shorter | `timesteps` in `scenarios.py` |
| make aiming finer | `frame_skip` (2 = 3.5°/action instead of 7.0°) |
| stop turn-and-spray | `exclusive_attack=True` |
| give the CNN more help | `aux_enemy=True` |
| stop early-stopping killing a slow run | `early_stop=False` |
| reduce eval noise | `eval_episodes` |
| change PPO itself | `PPO_KWARGS` in `train.py` |
| change parallelism | `N_ENVS` in `vec.py` (8 is the measured optimum) |

## Running things

```bash
doomrl train defend_center          # learn
doomrl eval defend_center           # score it
doomrl play defend_center           # watch it
doomrl record defend_center         # save a gif
tensorboard --logdir runs           # curves
```
