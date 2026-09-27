"""PPO agents for ViZDoom scenarios.

    scenarios.py   the per-scenario settings table -- start here
    env.py         one Gymnasium env wrapping a ViZDoom game
    vec.py         the parallel-env + wrapper stack
    train.py       PPO setup and the training loop
    auxiliary.py   the extra supervised head (defend_center, predict_position)
    evaluate.py    scoring a saved model
    record.py      GIF capture
    play.py        live playback in a window
    cli.py         the `doomrl` command

See NOTES.md for how it fits together.
"""
