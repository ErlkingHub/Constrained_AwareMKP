"""Prepare the gated T3-b dynamic theta-PG run; only --train starts training.

    python scripts/run_t3b.py
    python scripts/run_t3b.py --train --cuda_device 1

Requires a completed, non-smoke T3-a selection based only on synthetic N=100
instances. Claude and the user decide the research gate before using --train.
No training or benchmark is started by the default command.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
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
        raise FileNotFoundError(f"T3-b needs the actual T1-P dynamic source config: {source}")
    if not selected_path.is_file():
        raise FileNotFoundError(f"Run T3-a first; no global constants may be invented without {selected_path}")
    selection = json.loads(selected_path.read_text(encoding="utf-8"))
    if (selection.get("schema") != "t3a_selection_v1" or selection.get("status") != "complete" or
            selection.get("smoke_only") is not False or selection.get("selection_dataset") != "synthetic_N100" or
            selection.get("seed") != 1234):
        raise ValueError("T3-b requires a complete non-smoke t3a_selection_v1 from synthetic_N100, seed 1234")
    if selection.get("source_config_sha256") != sha256_file(source):
        raise ValueError("T3-a selection and T3-b source config SHA differ; preserve the actual scan source")
    original = config_from_dict(json.loads(source.read_text(encoding="utf-8")))
    if (original.device.seed != 1234 or original.train.signal != "theta_pg" or
            original.model.theta_update != "step" or original.model.theta_interval != 8 or
            original.model.theta_groups != ["price", "alpha"]):
        raise ValueError("Inherit the seed-1234 T1-P dynamic-k8 G0/G1 theta-PG configuration")
    expected = {"epochs": 10, "steps_per_epoch": 250, "batch_size": 64, "num_samples": 16}
    if any(getattr(original.train, name) != value for name, value in expected.items()):
        raise ValueError("Retain the T1-P 10 x 250, B64, K16 training budget")
    if (not original.train.initial_validation or original.train.val_every != 1 or
            original.train.resume or original.train.init_from):
        raise ValueError("T3-b starts fresh with epoch -1 validation and validation every epoch")
    if (not original.model.score_normalization or original.model.temperature_mode != "fixed" or
            abs(original.model.temperature_init - 1 / 8) > 1e-12):
        raise ValueError("Retain normalized scores and calibrated fixed beta=8")
    selected = selection.get("selected_global", {})
    model_values = selected.get("model")
    if not isinstance(model_values, dict):
        raise ValueError("T3-a selection lacks selected_global.model; do not infer missing constants")
    required = {"theta_groups", "price_anchor", "lp_source", "lp_interval", "lp_eta_init",
                "block_gamma_init", "alpha_init"}
    if not required.issubset(model_values):
        raise ValueError("T3-a selected model must explicitly record groups, LP paths and all global initial values")
    model_fields = {field.name for field in dataclasses.fields(original.model)}
    unknown = set(model_values) - model_fields
    if unknown:
        raise ValueError("Unknown selected model fields: " + ", ".join(sorted(unknown)))
    cfg = copy.deepcopy(original)
    tunable_model = {"theta_groups", "price_anchor", "lp_source", "lp_interval", "lp_eta_init",
                     "block_gamma_init", "alpha_init", "theta_update", "theta_interval"}
    for name, value in model_values.items():
        if name not in tunable_model and getattr(original.model, name) != value:
            raise ValueError(f"T3-a must preserve the source architecture/calibration model.{name}")
        setattr(cfg.model, name, value)
    groups = cfg.model.theta_groups
    if (not isinstance(groups, list) or len(groups) != len(set(groups)) or
            not {"price", "alpha"}.issubset(groups) or set(groups) - {"price", "alpha", "block", "lp"}):
        raise ValueError("The selected groups must retain price/alpha and may add only lp/block")
    if not ({"lp", "block"}.intersection(groups) or cfg.model.price_anchor == "residual"):
        raise ValueError("T3-b needs a retained new executor path; a G0/G1-only selection does not qualify")
    if cfg.model.price_anchor not in ("root", "residual") or cfg.model.lp_source not in ("root", "residual"):
        raise ValueError("The selected LP anchor/source must be root or residual")
    for name in ("alpha_init", "lp_eta_init", "block_gamma_init"):
        value = getattr(cfg.model, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"The selected model.{name} must be positive and finite")
    if not 0.5 < cfg.model.alpha_init < 2.5:
        raise ValueError("The selected alpha_init must lie strictly inside the sigmoid range (0.5, 2.5)")
    if isinstance(cfg.model.lp_interval, bool) or not isinstance(cfg.model.lp_interval, int) or cfg.model.lp_interval < 1:
        raise ValueError("The recorded scan lp_interval must be a positive integer")
    env_values = selected.get("env", {})
    if not isinstance(env_values, dict):
        raise ValueError("selected_global.env must be a dictionary when supplied")
    allowed_env = {"residual_lp_iters", "residual_lp_lr"}
    for name, value in env_values.items():
        if not hasattr(cfg.env, name):
            raise ValueError(f"Unknown selected environment field: {name}")
        if name in allowed_env:
            setattr(cfg.env, name, value)
        elif getattr(cfg.env, name) != value:
            raise ValueError(f"T3-a may change only the residual solver settings, not source env.{name}")
    if (isinstance(cfg.env.residual_lp_iters, bool) or not isinstance(cfg.env.residual_lp_iters, int) or
            cfg.env.residual_lp_iters < 1 or not math.isfinite(cfg.env.residual_lp_lr) or cfg.env.residual_lp_lr <= 0):
        raise ValueError("The selected residual solver settings must be positive and finite")
    # The method definition deliberately uses interval 8 even if T3-a's K1 arm won.
    cfg.model.theta_update, cfg.model.theta_interval, cfg.model.lp_interval = "step", 8, 8
    cfg.model.untrained_random_heads = False
    cfg.train.signal = "theta_pg"
    cfg.train.theta_pg_sigma_eta = 0.3
    cfg.train.run_name = "t3b_dynamic_k8_s1234"
    cfg.eval.ls_enable = False
    return cfg, selection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="runs/t1p_dynamic_k8_s1234/config.json")
    parser.add_argument("--selection", default="results/t3a_scan/selection.json")
    parser.add_argument("--output", default="results/t3b_configs")
    parser.add_argument("--cuda_device", type=int, default=None, help="GPU for --train only")
    parser.add_argument("--train", action="store_true", help="Explicitly start after Claude/user gate decision")
    args = parser.parse_args(argv)
    if args.cuda_device is not None and (args.cuda_device < 0 or not args.train):
        parser.error("--cuda_device must be nonnegative and is used only with --train")
    cfg, selection = prepare(args.config, args.selection)
    dest = Path(resolve_path(args.output))
    config_path = dest / "t3b_dynamic_k8_s1234.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != cfg.to_dict():
        raise FileExistsError(f"Preserve the different prepared config: {config_path}")
    dest.mkdir(parents=True, exist_ok=True)
    cfg.save(str(config_path))
    checkpoint = str(Path(cfg.train.ckpt_dir) / cfg.train.run_name / "best.pt")
    write_json(dest / "meta.json", {
        "source_config": resolve_path(args.config), "source_config_sha256": sha256_file(resolve_path(args.config)),
        "selection": resolve_path(args.selection), "selection_sha256": sha256_file(resolve_path(args.selection)),
        "selected_global": selection["selected_global"],
        "scan_gate": selection.get("scan_gate"), "crossfit_gate": selection.get("crossfit_gate"),
        "prepared_config": str(config_path), "prepared_config_sha256": sha256_file(config_path),
        "source": source_provenance(), "seed": 1234, "training_requested": args.train,
        "gate": "Claude/user decides whether to train; no metric starts training automatically",
        "changes": "Retained selected global executor paths/constants; theta_interval=lp_interval=8; "
                   "fresh deterministic head initialization; eta latent sigma=0.3; source budget/optimizer retained",
        "evaluation": f"python Benchmark_zeroShot.py --arm fixed,learned_dynamic --checkpoint {checkpoint} --stages greedy",
        "interval_note": "T3-b uses LP interval 8 by definition even if the scan selected K1 at interval 1",
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
