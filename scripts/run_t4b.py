"""Prepare static max@16 proposal learning from the formal T4-a theta selection.

    python scripts/run_t4b.py
    python scripts/run_t4b.py --train --cuda_device 1

The default command writes configuration only. Claude/user decides the research
gate before explicitly using --train; no benchmark is started by this script.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mkp_artifacts import sha256_file, source_provenance, write_json
from mkp_config import config_from_dict, get_device, resolve_path


def prepare(source_path, selection_path):
    source, selected_path = Path(resolve_path(source_path)), Path(resolve_path(selection_path))
    if not source.is_file():
        raise FileNotFoundError(f"T4-b requires the actual T1-P static configuration: {source}")
    if not selected_path.is_file():
        raise FileNotFoundError(f"Run the formal T4-a scan first; no theta proposal may be invented without {selected_path}")
    selection = json.loads(selected_path.read_text(encoding="utf-8"))
    if (selection.get("schema") != "t4a_sms_selection_v1" or selection.get("status") != "complete" or
            selection.get("smoke_only") is not False or selection.get("selection_dataset") != "synthetic_N100" or
            selection.get("seed") != 1234 or selection.get("selection_instances") != 128):
        raise ValueError("T4-b requires a complete non-smoke t4a_sms_selection_v1 from 128 synthetic_N100 instances, seed 1234")
    if selection.get("source_config_sha256") != sha256_file(source):
        raise ValueError("T4-a selection/source configuration SHA mismatch")
    original = config_from_dict(json.loads(source.read_text(encoding="utf-8")))
    if (original.device.seed != 1234 or original.model.theta_update != "static" or
            original.model.theta_groups != ["price", "alpha"] or original.model.price_anchor != "root" or
            original.model.alpha_init != 1.0 or original.model.theta_sigma_head or
            original.train.signal != "theta_pg"):
        raise ValueError("Inherit the seed-1234 T1-P static F0 executor and its training configuration")
    expected = {"epochs": 10, "steps_per_epoch": 250, "batch_size": 64, "num_samples": 16}
    if any(getattr(original.train, name) != value for name, value in expected.items()):
        raise ValueError("Retain T1-P's 10 x 250 steps, B64 and K16 budget")
    if (not original.train.initial_validation or original.train.val_every != 1 or
            original.train.resume or original.train.init_from):
        raise ValueError("T4-b starts fresh with epoch -1 and per-epoch validation")
    if (not original.model.score_normalization or original.model.temperature_mode != "fixed" or
            abs(original.model.temperature_init - 1 / 8) > 1e-12):
        raise ValueError("Retain normalized scores and the source fixed beta=8 calibration")
    theta = selection.get("selected_theta")
    if not isinstance(theta, dict) or theta.get("dims") not in ("u", "a", "ua"):
        raise ValueError("T4-a must explicitly select theta dimensions u/a/ua")
    scale = theta.get("scale")
    if (isinstance(scale, bool) or not isinstance(scale, (float, int)) or not math.isfinite(scale) or
            scale not in (0.5, 1.0, 2.0, 4.0)):
        raise ValueError("T4-a must explicitly select one registered theta scale: 0.5, 1, 2, 4")
    cfg = copy.deepcopy(original)
    cfg.model.theta_update, cfg.model.theta_interval = "static", 1
    cfg.model.theta_sigma_head = True
    cfg.model.theta_sigma_dims = theta["dims"]
    cfg.model.theta_sigma_u0 = float(scale) * 0.05
    cfg.model.theta_sigma_alpha0 = float(scale) * 0.3
    cfg.model.untrained_random_heads = False
    cfg.train.signal = "theta_maxk"
    cfg.train.theta_pg_antithetic = False
    cfg.train.theta_pg_sigma_u = cfg.model.theta_sigma_u0
    cfg.train.theta_pg_sigma_alpha = cfg.model.theta_sigma_alpha0
    cfg.train.val_noise_seed = 20261106
    cfg.train.run_name = "t4b_static_maxk16_s1234"
    cfg.eval.sms_dims, cfg.eval.sms_scale = theta["dims"], float(scale)
    cfg.eval.sms_include_center = True
    cfg.eval.ls_enable = False
    return cfg, selection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="runs/t1p_static_s1234/config.json")
    parser.add_argument("--selection", default="results/t4a_sms/selection.json")
    parser.add_argument("--output", default="results/t4b_configs")
    parser.add_argument("--cuda_device", type=int, default=None, help="GPU for --train only")
    parser.add_argument("--train", action="store_true", help="Explicitly start training after the research gate decision")
    args = parser.parse_args(argv)
    if args.cuda_device is not None and (args.cuda_device < 0 or not args.train):
        parser.error("--cuda_device must be nonnegative and is used only with --train")
    cfg, selection = prepare(args.config, args.selection)
    dest = Path(resolve_path(args.output))
    config_path = dest / "t4b_static_maxk16_s1234.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != cfg.to_dict():
        raise FileExistsError(f"Preserve the different prepared configuration: {config_path}")
    dest.mkdir(parents=True, exist_ok=True)
    cfg.save(str(config_path))
    checkpoint = str(Path(cfg.train.ckpt_dir) / cfg.train.run_name / "best.pt")
    write_json(dest / "meta.json", {
        "source_config": resolve_path(args.config), "source_config_sha256": sha256_file(resolve_path(args.config)),
        "selection": resolve_path(args.selection), "selection_sha256": sha256_file(resolve_path(args.selection)),
        "selected_theta": selection["selected_theta"], "source": source_provenance(),
        "prepared_config": str(config_path), "prepared_config_sha256": sha256_file(config_path),
        "seed": 1234, "training_requested": args.train, "execution_cuda_override": args.cuda_device,
        "gate": "Claude/user decides whether to train; no metric starts training automatically",
        "training": "16 independent static theta draws, no forced center, no antithetic and no entropy term",
        "validation": "best of 16: one mean-theta greedy center plus 15 independent proposals; "
                      "fixed val_noise_seed=20261106, independent of the training RNG; includes epoch -1",
        "evaluation": f"python Benchmark_zeroShot.py --arm fixed,learned_static,learned_mu_only "
                      f"--static_checkpoint {checkpoint} --stages theta16,theta64,theta64+ILS "
                      f"--eval.sms_dims {cfg.eval.sms_dims} --eval.sms_scale {cfg.eval.sms_scale}",
        "scope": "F0 root executor; only mean/sigma proposal heads learn; no residual LP or block group",
    })
    print(f"Prepared: {config_path}")
    print("Training starts only with --train after the Claude/user research gate decision.")
    if args.train:
        run_path = Path(resolve_path(cfg.train.runs_dir)) / cfg.train.run_name
        checkpoint_path = Path(resolve_path(cfg.train.ckpt_dir)) / cfg.train.run_name
        if run_path.exists() or checkpoint_path.exists():
            raise FileExistsError(f"Preserve the existing run/checkpoint: {cfg.train.run_name}")
        if args.cuda_device is not None:
            cfg.device.cuda_device = args.cuda_device
        from mkp_Trainer import MKPTrainer, set_seed
        set_seed(cfg.device.seed)
        MKPTrainer(cfg, get_device(cfg)).fit()
    return dest


if __name__ == "__main__":
    main()
