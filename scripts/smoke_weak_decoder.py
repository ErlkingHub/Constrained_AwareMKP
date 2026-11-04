"""Small T1 engineering check; results are not a formal performance experiment.

Run from the repository: python scripts/smoke_weak_decoder.py
Two separately trained step/static policies use the same seed and 2 x 20 steps.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from mkp_artifacts import write_json
from mkp_config import Config, resolve_path
from mkp_Trainer import MKPTrainer, set_seed


def run(mode):
    cfg = Config()
    cfg.device.amp = False
    cfg.device.seed = 1234
    cfg.env.train_sizes, cfg.env.train_dims = [20, 30], [3, 5]
    cfg.env.dual_iters = 30
    cfg.model.embed_dim, cfg.model.num_heads = 32, 4
    cfg.model.encoder_layers, cfg.model.controller_layers = 1, 1
    cfg.model.edge_hidden = 8
    cfg.model.theta_update = mode
    cfg.train.batch_size, cfg.train.num_samples = 2, 16
    cfg.train.epochs, cfg.train.steps_per_epoch = 2, 20
    cfg.train.val_sizes, cfg.train.val_dims, cfg.train.val_instances = [20], [3, 5], 2
    cfg.train.val_milp_time_limit = 2.0
    cfg.train.log_every = 5
    cfg.train.run_name = "weak_decoder_t1_smoke_" + ("dynamic" if mode == "step" else "static")
    cfg.train.ckpt_dir = "checkpoints/weak_decoder_t1_smoke"
    expected_csv = resolve_path(os.path.join(cfg.train.runs_dir, cfg.train.run_name, "train_log.csv"))
    if os.path.exists(expected_csv):
        raise FileExistsError(f"Smoke output exists; preserve it or choose a fresh run name: {expected_csv}")
    set_seed(cfg.device.seed)
    started = time.perf_counter()
    trainer = MKPTrainer(cfg, torch.device("cpu"))
    trainer.log("T1 ENGINEERING SMOKE ONLY: reduced N/M/model/batch; not a formal gap experiment")
    trainer.fit()
    elapsed = time.perf_counter() - started
    with open(expected_csv, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    keys = ("loss", "grad_norm", "reward", "u_abs_mean", "alpha_mean", "temperature")
    if len(rows) != 40 or any(not math.isfinite(float(row[key])) for row in rows for key in keys):
        raise AssertionError("Smoke needs 40 finite training steps")
    if any(int(row["finite_gradients"]) != 1 for row in rows):
        raise AssertionError("Smoke encountered a non-finite gradient")
    checks = []
    with torch.no_grad():
        trainer.model.eval()
        for validation in trainer.val_sets:
            inst = validation["inst"]
            out = trainer.model(trainer.env, inst, decode="greedy", pre=validation["pre"], return_trace=True)
            solution = out["solution"].bool()
            remaining = inst.c - torch.einsum("bmn,bn->bm", inst.r, solution.float())
            feasible = bool((remaining >= -1e-5).all())
            fits = (inst.r <= remaining[:, :, None] + 1e-6).all(1) & ~solution
            maximal = not bool(fits.any())
            if not feasible or not maximal:
                raise AssertionError("Smoke produced an infeasible or nonmaximal solution")
            checks.append({"set": validation["name"], "all_feasible": feasible, "all_maximal": maximal,
                           "solution": solution.int().tolist(),
                           "theta_trace": [{k: v.tolist() if torch.is_tensor(v) else v for k, v in step.items()}
                                           for step in out["trace"]]})
    result = {"formal_experiment": False, "theta_update": mode, "seed": cfg.device.seed,
              "steps": len(rows), "epochs": 2, "num_samples": 16, "device": "cpu",
              "elapsed_s": elapsed, "all_losses_and_gradients_finite": True,
              "stats": {key: {"min": min(float(r[key]) for r in rows),
                               "max": max(float(r[key]) for r in rows)} for key in keys},
              "validation": checks, "run_dir": trainer.run_dir, "checkpoint_dir": trainer.ckpt_dir}
    write_json(os.path.join(trainer.run_dir, "smoke_report.json"), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("both", "step", "static"), default="both")
    args = parser.parse_args()
    torch.set_num_threads(2)
    for mode in (("step", "static") if args.mode == "both" else (args.mode,)):
        result = run(mode)
        print(f"SMOKE PASSED {mode}: {result['steps']} steps, {result['elapsed_s']:.1f}s, "
              f"finite losses/gradients, feasible maximal validation solutions", flush=True)


if __name__ == "__main__":
    main()
