"""Frozen-policy MKP evaluation; Chu data never participate in model selection.

Run ``python Benchmark_zeroShot.py`` after training, or pass ``--checkpoint``.
The default reports five matched single-greedy arms. ``--best-of-k K`` adds a
separate stage consisting of exactly K samples (it does not include greedy).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mkp_config import ExperimentConfig, load_config
from mkp_data import generate_instance, load_chu, solve_root_lp
from mkp_env import MKPState
from mkp_Model import ResourceController, run_episode
from mkp_utils import (
    load_checkpoint, resolve_device, resolve_path, runtime_manifest,
    save_source_snapshot, seed_everything, sha256_file, timestamp, write_json,
)


ARMS = ("dynamic", "static", "fixed", "untrained", "lp_dual")


def signed_gap(objective: float, reference: float | None) -> float | None:
    """Percent gap, retaining negative values; no claim of proven optimality."""
    if reference is None or not math.isfinite(reference) or reference <= 0:
        return None
    return 100.0 * (reference - objective) / reference


def paired_seed(seed: int, instance_id: str, stage: str, trial: int) -> int:
    text = f"{seed}|{instance_id}|{stage}|{trial}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") % (2**63 - 1)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty result table: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _resolve_checkpoint(config: ExperimentConfig) -> Path:
    if config.eval.checkpoint:
        path = resolve_path(config.eval.checkpoint)
    else:
        latest = (resolve_path(config.train.checkpoint_root) /
                  config.train.version / "latest.json")
        if not latest.is_file():
            raise FileNotFoundError(
                f"No trained checkpoint pointer at {latest}. Run "
                "train_ConstrainedMKP.py first, or supply --checkpoint."
            )
        pointer = json.loads(latest.read_text(encoding="utf-8"))
        if not pointer.get("best_checkpoint"):
            raise ValueError(f"Missing best_checkpoint in {latest}")
        path = resolve_path(pointer["best_checkpoint"])
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def _instances(config: ExperimentConfig, trained: ExperimentConfig):
    cfg = config.eval
    if cfg.synthetic:
        if cfg.seed in (trained.train.seed, trained.data.validation_seed):
            raise ValueError("Synthetic test seed must differ from training and validation seeds")
        instances = []
        for n, m, family, ratio in product(
            cfg.synthetic_n, cfg.synthetic_m, cfg.synthetic_distributions,
            cfg.synthetic_ratios,
        ):
            for replicate in range(cfg.synthetic_instances_per_group):
                instance_id = f"test_N{n}_M{m}_{family}_a{ratio:g}_{replicate:04d}"
                # A test instance is independent of training settings and of
                # other cells included in the evaluation grid.
                rng = np.random.default_rng(paired_seed(cfg.seed, instance_id, "synthetic_data", 0))
                instances.append(generate_instance(
                    rng, n, m, family, ratio, instance_id,
                    coefficient_max=cfg.synthetic_coefficient_max,
                    correlation_noise=cfg.synthetic_correlation_noise,
                    correlation_offset=cfg.synthetic_correlation_offset,
                ))
        dataset_path = None
        dataset_sha = hashlib.sha256(
            "\n".join(instance.fingerprint() for instance in instances).encode("ascii")
        ).hexdigest()
    else:
        dataset_path = resolve_path(cfg.dataset_path)
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Chu dataset not found: {dataset_path}")
        instances = load_chu(dataset_path)
        dataset_sha = sha256_file(dataset_path)
    original_count = len(instances)
    instances = [instance for instance in instances
                 if (not cfg.filter_n or instance.n in cfg.filter_n)
                 and (not cfg.filter_m or instance.m in cfg.filter_m)]
    if cfg.limit is not None:
        instances = instances[:cfg.limit]
    if not instances:
        raise ValueError("No test instances remain after applying N/M/limit filters")
    ids = [instance.id for instance in instances]
    if len(ids) != len(set(ids)):
        raise ValueError("Test instance IDs must be unique for paired comparisons")
    return instances, dataset_path, dataset_sha, original_count


def audit_solution(instance, result: dict[str, Any]) -> dict[str, Any]:
    """Recompute raw, float64 feasibility and objective independently of the policy."""
    selected = [int(index) for index in result["selected"]]
    if len(selected) != len(set(selected)):
        raise RuntimeError(f"Duplicate selected item in {instance.id}")
    if any(index < 0 or index >= instance.n for index in selected):
        raise RuntimeError(f"Out-of-range selected item in {instance.id}")
    used = instance.w[selected].sum(axis=0) if selected else np.zeros(instance.m)
    objective = float(instance.p[selected].sum()) if selected else 0.0
    feasible = bool(np.all(used <= instance.c))
    if not feasible:
        raise RuntimeError(f"Policy returned an infeasible solution for {instance.id}")
    if not np.isclose(objective, result["objective"], rtol=1e-12, atol=1e-8):
        raise RuntimeError(f"Objective/sequence mismatch for {instance.id}")
    left = np.ones(instance.n, dtype=bool)
    left[selected] = False
    maximal = not bool(np.any(np.all(instance.w[left] <= instance.c - used, axis=1)))
    if not maximal:
        raise RuntimeError(f"Policy stopped before a maximal solution for {instance.id}")
    # Zero-capacity resources contribute 0; their strict feasibility was checked.
    utilization = np.divide(used, instance.c, out=np.zeros_like(used), where=instance.c > 0)
    return {"selected": selected, "objective": objective, "feasible": feasible,
            "maximal": maximal, "utilization_mean": float(utilization.mean()),
            "utilization_min": float(utilization.min()),
            "utilization_max": float(utilization.max()),
            "utilization_by_resource": utilization.tolist()}


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Independent instances are the statistical unit, not trajectories/prefixes."""
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        for grouping, keys in (
            ("overall", ()), ("N", ("n",)), ("M", ("m",)),
            ("capacity_ratio", ("capacity_ratio_group",)),
            ("distribution", ("distribution",)),
            ("N_M_capacity_distribution", ("n", "m", "capacity_ratio_group", "distribution")),
            ("generalization", ("seen_n", "seen_m", "seen_distribution")),
        ):
            dimensions = tuple((key, row[key]) for key in keys)
            buckets[(row["arm"], row["stage"], row["budget"], grouping, dimensions)].append(row)
    output = []
    for (arm, stage, budget, grouping, dimensions), group in buckets.items():
        item = {"arm": arm, "stage": stage, "budget": budget, "grouping": grouping,
                **dict(dimensions), "instances": len(group),
                "feasible_count": sum(int(row["feasible"]) for row in group),
                "maximal_count": sum(int(row["maximal"]) for row in group),
                "best_known_match_or_better_count": sum(
                    row.get("best_known_gap_pct") is not None and row["best_known_gap_pct"] <= 0
                    for row in group)}
        for metric in ("best_known_gap_pct", "lp_gap_pct", "objective", "steps",
                       "utilization_mean", "lp_seconds", "policy_seconds", "total_seconds",
                       "policy_seconds_per_decision", "theta_change_mean"):
            values = np.asarray([row[metric] for row in group
                                 if row.get(metric) is not None and math.isfinite(row[metric])],
                                dtype=float)
            item[f"{metric}_count"] = int(values.size)
            for stat in ("mean", "median", "p95", "max", "std"):
                if not values.size:
                    value = None
                elif stat == "mean":
                    value = float(values.mean())
                elif stat == "median":
                    value = float(np.median(values))
                elif stat == "p95":
                    value = float(np.quantile(values, 0.95))
                elif stat == "max":
                    value = float(values.max())
                else:
                    value = float(values.std(ddof=1)) if values.size > 1 else None
                item[f"{metric}_{stat}"] = value
        output.append(item)
    return output


def paired_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-instance dynamic-minus-control differences; gap deltas are points."""
    lookup = {(row["instance_id"], row["stage"], row["budget"], row["arm"]): row for row in rows}
    output = []
    for row in rows:
        if row["arm"] != "dynamic":
            continue
        for control in ARMS[1:]:
            other = lookup.get((row["instance_id"], row["stage"], row["budget"], control))
            if other is None:
                continue
            output.append({
                "instance_id": row["instance_id"], "n": row["n"], "m": row["m"],
                "distribution": row["distribution"],
                "capacity_ratio_group": row["capacity_ratio_group"],
                "stage": row["stage"], "budget": row["budget"], "control": control,
                "objective_gain_dynamic_minus_control": row["objective"] - other["objective"],
                "best_known_gap_delta_pp": (
                    row["best_known_gap_pct"] - other["best_known_gap_pct"]
                    if row["best_known_gap_pct"] is not None else None),
                "lp_gap_delta_pp": (
                    row["lp_gap_pct"] - other["lp_gap_pct"]
                    if row["lp_gap_pct"] is not None else None),
                "time_delta_seconds": row["total_seconds"] - other["total_seconds"],
            })
    return output


def evaluate(config: ExperimentConfig) -> Path:
    config.validate()
    cfg = config.eval
    if not cfg.arms or any(arm not in ARMS for arm in cfg.arms) or len(set(cfg.arms)) != len(cfg.arms):
        raise ValueError(f"arms must be a nonempty, unique subset of {ARMS}")
    if cfg.best_of_k < 0 or cfg.torch_threads < 1 or (cfg.limit is not None and cfg.limit < 1):
        raise ValueError("best_of_k must be >= 0; threads and limit must be positive")
    seed_everything(cfg.seed)
    torch.set_num_threads(cfg.torch_threads)
    device = torch.device(resolve_device(cfg.device))
    checkpoint_path = _resolve_checkpoint(config)
    checkpoint = load_checkpoint(checkpoint_path, device="cpu")
    for required in ("config", "actor", "initial_actor"):
        if required not in checkpoint:
            raise ValueError(f"Checkpoint is missing {required}; cannot perform matched evaluation")
    trained = ExperimentConfig.from_dict(checkpoint["config"])
    # The checkpoint, not a fresh config/default, defines the architecture/executor.
    actor = ResourceController(trained.model).to(device)
    actor.load_state_dict(checkpoint["actor"], strict=True)
    actor.eval().requires_grad_(False)
    initial_actor = ResourceController(trained.model).to(device)
    initial_actor.load_state_dict(checkpoint["initial_actor"], strict=True)
    initial_actor.eval().requires_grad_(False)
    checkpoint_sha = sha256_file(checkpoint_path)
    instances, dataset_path, dataset_sha, original_count = _instances(config, trained)
    output_dir = resolve_path(cfg.output_root) / f"{timestamp()}_{trained.train.version}"
    output_dir.mkdir(parents=True, exist_ok=False)
    save_source_snapshot(output_dir / "source")
    manifest = runtime_manifest(config.to_dict())
    manifest.update({
        "status": "running", "evaluation_config": config.to_dict(),
        "checkpoint_config": checkpoint["config"], "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha, "checkpoint_update": checkpoint.get("update"),
        "training_seed": checkpoint.get("seed", trained.train.seed),
        "checkpoint_selection": {
            "resolved_via": "explicit checkpoint" if cfg.checkpoint else "latest synthetic-best pointer",
            "recorded_training_rule": checkpoint.get("selection_rule"),
            "evaluated_update": checkpoint.get("update"),
            "recorded_best_update": checkpoint.get("best_update"),
            "evaluation_changes_or_selects_weights": False,
        },
        "training_source_manifest": checkpoint.get("source_manifest"),
        "dataset": str(dataset_path) if dataset_path else "generated synthetic final-test suite",
        "dataset_sha256": dataset_sha, "dataset_instances_before_filters": original_count,
        "evaluated_instances": len(instances), "evaluation_seed": cfg.seed,
        "device": str(device), "instance_sha256": {x.id: x.fingerprint() for x in instances},
        "timing_boundary": "One root LP charged to each arm/stage plus all policy decisions and "
                           "state construction; GPU synchronized. Loading, file output and "
                           "independent solution audit excluded. Shared LP computed once per instance.",
        "sampling_protocol": "K independent samples with matched per-instance/trial seeds across "
                             "arms; greedy excluded from best-of-K; no local search or repair.",
        "statistical_unit": "independent instance within this one training seed",
        "reference_note": "best-known is not proven optimal; synthetic data have no best-known label; "
                          "LP gap and best-known gap are distinct; negative gaps are retained.",
        "heldout_note": "Repeated inspection of this suite makes it research feedback. Reserve a "
                        "separate unseen final-test seed for publication claims.",
    })
    write_json(output_dir / "manifest.json", manifest)
    rows: list[dict[str, Any]] = []
    suite_start = time.perf_counter()
    try:
        with (output_dir / "solutions.jsonl").open("w", encoding="utf-8") as solutions:
            for number, instance in enumerate(instances, 1):
                lp = solve_root_lp(instance)
                for arm in cfg.arms:
                    active_actor = initial_actor if arm == "untrained" else actor
                    mode = "dynamic" if arm == "untrained" else arm
                    stages = [("greedy", 1, False)]
                    if cfg.best_of_k:
                        stages.append(("sample_best_of_k", cfg.best_of_k, True))
                    for stage, budget, sample in stages:
                        _synchronize(device)
                        start = time.perf_counter()
                        results = []
                        seeds = []
                        with torch.inference_mode():
                            for trial in range(budget):
                                trial_seed = paired_seed(cfg.seed, instance.id, stage, trial)
                                generator = torch.Generator(device=device).manual_seed(trial_seed)
                                result = run_episode(active_actor,
                                    MKPState(instance, lp, device=device), mode=mode,
                                    sample=sample, generator=generator)
                                results.append(result)
                                seeds.append(trial_seed)
                        _synchronize(device)
                        policy_seconds = time.perf_counter() - start
                        audited = [audit_solution(instance, result) for result in results]
                        winner = max(range(budget), key=lambda i: audited[i]["objective"])
                        best, audit = results[winner], audited[winner]
                        decisions = sum(len(result["selected"]) for result in results)
                        row = {
                            "instance_id": instance.id, "instance_sha256": instance.fingerprint(),
                            "n": instance.n, "m": instance.m, "distribution": instance.distribution,
                            "capacity_ratio": instance.capacity_ratio,
                            "capacity_ratio_group": round(instance.capacity_ratio, 2),
                            "seen_n": instance.n in trained.data.train_n,
                            "seen_m": instance.m in trained.data.train_m,
                            "seen_distribution": instance.distribution in trained.data.train_distributions,
                            "arm": arm, "stage": stage, "budget": budget,
                            "objective": audit["objective"], "best_known": instance.best_known,
                            "lp_upper_bound": lp.upper_bound,
                            "best_known_gap_pct": signed_gap(audit["objective"], instance.best_known),
                            "lp_gap_pct": signed_gap(audit["objective"], lp.upper_bound),
                            "feasible": audit["feasible"], "maximal": audit["maximal"],
                            "steps": len(audit["selected"]), "total_decisions": decisions,
                            "utilization_mean": audit["utilization_mean"],
                            "utilization_min": audit["utilization_min"],
                            "utilization_max": audit["utilization_max"],
                            "theta_change_mean": float(best.get("theta_change_mean", 0.0)),
                            "lp_seconds": lp.solve_seconds, "policy_seconds": policy_seconds,
                            "total_seconds": lp.solve_seconds + policy_seconds,
                            "policy_seconds_per_decision": policy_seconds / max(decisions, 1),
                            "winning_trial": winner, "evaluation_seed": cfg.seed,
                            "training_seed": trained.train.seed,
                            "checkpoint_update": checkpoint.get("update"),
                            "checkpoint_sha256": checkpoint_sha, "dataset_sha256": dataset_sha,
                        }
                        rows.append(row)
                        solutions.write(json.dumps({
                            "instance_id": instance.id, "arm": arm, "stage": stage,
                            "budget": budget, "winning_trial": winner, "trial_seeds": seeds,
                            "selected": audit["selected"], "objective": audit["objective"],
                            "utilization_by_resource": audit["utilization_by_resource"],
                            "theta_trace": best.get("theta_trace", []),
                            "trials": [{"selected": item["selected"], "objective": item["objective"]}
                                       for item in audited],
                        }, ensure_ascii=False, allow_nan=False) + "\n")
                        solutions.flush()
                write_csv(output_dir / "instances.csv", rows)
                print(f"[{number}/{len(instances)}] {instance.id} N={instance.n} M={instance.m}", flush=True)
        write_csv(output_dir / "summary.csv", summarize(rows))
        paired = paired_comparisons(rows)
        if paired:
            write_csv(output_dir / "paired.csv", paired)
        manifest.update(status="complete", total_wall_seconds=time.perf_counter() - suite_start,
                        result_rows=len(rows), result_files_sha256={
                            file.name: sha256_file(file) for file in output_dir.iterdir()
                            if file.suffix in (".csv", ".jsonl")})
        write_json(output_dir / "manifest.json", manifest)
    except BaseException as error:
        if rows:
            write_csv(output_dir / "instances.csv", rows)
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=f"{type(error).__name__}: {error}", completed_result_rows=len(rows))
        write_json(output_dir / "manifest.json", manifest)
        raise
    print(f"Zero-shot results: {output_dir}", flush=True)
    return output_dir


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, help="Optional JSON experiment config")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--limit", type=int, help="Limit instances for a bounded smoke test")
    parser.add_argument("--n", type=int, nargs="+", dest="filter_n")
    parser.add_argument("--m", type=int, nargs="+", dest="filter_m")
    parser.add_argument("--arms", choices=ARMS, nargs="+")
    parser.add_argument("--best-of-k", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--synthetic", action="store_true", default=None)
    parser.add_argument("--output-root", type=str)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        setattr(config.eval, key, tuple(value) if isinstance(value, list) else value)
    return evaluate(config)


if __name__ == "__main__":
    main()
