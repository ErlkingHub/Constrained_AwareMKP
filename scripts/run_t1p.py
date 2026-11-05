"""Prepare T1-P's two seed-1234 runs; --train explicitly starts user-run training.

    python scripts/run_t1p.py
    python scripts/run_t1p.py --mode static --train --cuda_device 0
    python scripts/run_t1p.py --mode dynamic --train --cuda_device 1

The GPU override affects the executed run; the prepared mathematical configs inherit
the source device. Training records its actual complete config in the run directory.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mkp_artifacts import sha256_file, source_provenance, write_json
from mkp_config import config_from_dict, get_device, resolve_path
from mkp_Trainer import MKPTrainer, set_seed


def prepare(source_path):
    """Copy T1-R's selected settings; retain its optimizer, data and validation budget."""
    source = Path(resolve_path(source_path))
    original = config_from_dict(json.loads(source.read_text(encoding="utf-8")))
    if original.device.seed != 1234 or original.model.theta_groups != ["price", "alpha"]:
        raise ValueError("T1-P specifies seed 1234 and unchanged G0/G1 theta groups")
    if not original.model.score_normalization or original.model.temperature_mode != "fixed" or abs(original.model.temperature_init - 1/8) > 1e-12:
        raise ValueError("Inherit normalized scores and calibrated fixed beta=8; do not recalibrate")
    expected = {"epochs": 10, "steps_per_epoch": 250, "batch_size": 64, "num_samples": 16}
    if any(getattr(original.train, key) != value for key, value in expected.items()):
        raise ValueError("T1-P requires the inherited 10 epochs x 250 steps, B=64, K=16 budget")
    if not original.train.initial_validation or original.train.val_every != 1:
        raise ValueError("Keep epoch -1 initialization and validation at every epoch")
    if original.train.resume or original.train.init_from:
        raise ValueError("T1-P starts fresh; source resume/init_from must be empty")
    configs = {}
    for mode, interval, name in (("static", 1, "t1p_static_s1234"),
                                  ("step", 8, "t1p_dynamic_k8_s1234")):
        cfg = copy.deepcopy(original)
        cfg.model.theta_update, cfg.model.theta_interval = mode, interval
        cfg.train.signal = "theta_pg"
        cfg.train.theta_pg_sigma_u = .05
        cfg.train.theta_pg_sigma_alpha = .3
        cfg.train.theta_pg_antithetic = True
        cfg.train.gate_early_stop = True
        cfg.train.run_name = name
        configs["static" if mode == "static" else "dynamic_k8"] = cfg
    return configs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="runs/t1r_static_s1234/config.json")
    parser.add_argument("--output", default="results/t1p_configs")
    parser.add_argument("--mode", choices=("static", "dynamic", "both"), default="both")
    parser.add_argument("--cuda_device", type=int, default=None, help="GPU for --train only")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args(argv)
    if args.cuda_device is not None and (args.cuda_device < 0 or not args.train):
        parser.error("--cuda_device must be nonnegative and is used only with --train")
    configs = prepare(args.config)
    selected = configs if args.mode == "both" else {"static" if args.mode == "static" else "dynamic_k8":
                    configs["static" if args.mode == "static" else "dynamic_k8"]}
    dest = Path(resolve_path(args.output))
    dest.mkdir(parents=True, exist_ok=True)
    for name, cfg in selected.items():
        cfg_path = dest / (name + ".json")
        if cfg_path.exists() and json.loads(cfg_path.read_text(encoding="utf-8")) != cfg.to_dict():
            raise FileExistsError(f"Preserve a different prepared configuration: {cfg_path}")
        cfg.save(str(cfg_path))
        print(f"Prepared: {cfg_path}")
    write_json(dest / "meta.json", {"source_config": resolve_path(args.config),
        "source_config_sha256": sha256_file(resolve_path(args.config)), "source": source_provenance(),
        "seed": 1234, "training_started": args.train, "prepared_configs": {
            key: {"path": str(dest / (key + ".json")), "sha256": sha256_file(dest / (key + ".json"))}
            for key in configs if (dest / (key + ".json")).exists()},
        "changes": "theta-PG signal, prescribed fixed sigmas/antithetic noise, update mode/interval, run name; "
                   "gate early stopping enabled for both arms. Beta, optimizer and training budget inherited.",
        "training": "Only --train starts training. --mode both runs sequentially; use separate mode commands "
                    "on two GPUs to run in parallel. No benchmark or other diagnostic is started.",
        "evaluation": "After training, Benchmark_zeroShot.py --arm fixed,learned_static,learned_dynamic "
                      "--static_checkpoint checkpoints/t1p_static_s1234/best.pt "
                      "--checkpoint checkpoints/t1p_dynamic_k8_s1234/best.pt --stages greedy"})
    if args.train:
        for cfg in selected.values():
            run_path = Path(resolve_path(cfg.train.runs_dir)) / cfg.train.run_name
            checkpoint_path = Path(resolve_path(cfg.train.ckpt_dir)) / cfg.train.run_name
            if run_path.exists() or checkpoint_path.exists():
                raise FileExistsError(f"Preserve the existing run/checkpoint: {cfg.train.run_name}")
            if args.cuda_device is not None:
                cfg.device.cuda_device = args.cuda_device
            set_seed(cfg.device.seed)
            MKPTrainer(cfg, get_device(cfg)).fit()
    return dest


if __name__ == "__main__":
    main()
