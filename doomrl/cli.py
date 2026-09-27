"""The `doomrl` command. Four subcommands, one scenario argument each:

    train   learn a policy         -> runs/<scenario>/
    eval    score a saved policy   -> prints mean +/- std
    record  save a GIF             -> media/<scenario>.gif
    play    watch it in a window   -> nothing written

Imports are inside the branches so `--help` doesn't have to load torch.
"""

import argparse

from doomrl.scenarios import SCENARIOS


def main():
    parser = argparse.ArgumentParser(prog="doomrl")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("train", "eval", "record", "play"):
        p = sub.add_parser(name)
        p.add_argument("scenario", choices=sorted(SCENARIOS))
        p.add_argument("--model", default=None)
        if name == "train":
            p.add_argument("--timesteps", type=int, default=None)
            p.add_argument("--seed", type=int, default=0)
        if name == "play":
            p.add_argument("--episodes", type=int, default=5)
            p.add_argument(
                "--speed", type=float, default=1.0,
                help="1.0 is real time; 0.5 is half speed, 2.0 is double",
            )
            p.add_argument(
                "--seed", type=int, default=None,
                help="fix the episodes; omit for a different run each time",
            )
            p.add_argument(
                "--stochastic", action="store_true",
                help="sample from the policy instead of taking the best action",
            )

    args = parser.parse_args()

    if args.command == "train":
        from doomrl.train import train
        train(args.scenario, args.timesteps, args.seed)
    elif args.command == "eval":
        from doomrl.evaluate import evaluate
        evaluate(args.scenario, args.model)
    elif args.command == "play":
        from doomrl.play import play
        play(
            args.scenario, args.model, args.episodes,
            args.speed, args.seed, args.stochastic,
        )
    else:
        from doomrl.record import record
        record(args.scenario, args.model)


if __name__ == "__main__":
    main()
