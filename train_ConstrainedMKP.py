"""
train_ConstrainedMKP.py
=======================
Minimal entry point: trains the constraint-aware hierarchical MKP policy and writes
checkpoints to checkpoints/ (best.pt / last.pt) and logs to runs/<timestamp>/.

    python train_ConstrainedMKP.py
    python train_ConstrainedMKP.py --train.epochs 200 --train.batch_size 32 --train.num_samples 32
    python train_ConstrainedMKP.py --train.resume checkpoints/last.pt
"""
import json
import sys

from mkp_config import get_config, get_device, build_arg_parser
from mkp_Trainer import MKPTrainer, set_seed


def main(argv=None):
    parser = build_arg_parser("Train the Constrained-Aware MKP policy.")
    args, overrides = parser.parse_known_args(argv)
    cfg = get_config(overrides)
    if args.print_config:
        print(json.dumps(cfg.to_dict(), indent=2))
        return
    set_seed(cfg.device.seed)
    device = get_device(cfg)          # cuda:1 by default (see mkp_config.DeviceConfig)
    MKPTrainer(cfg, device).fit()


if __name__ == "__main__":
    main(sys.argv[1:])
