"""User-run T3-a fixed greedy scan with synthetic-only selection and two-fold cross-fitting.

    python scripts/run_t3a_scan.py --device cuda:0
    python scripts/run_t3a_scan.py --limit 2 --smoke-items 12 --output results/t3_code_checks/scan_smoke

The second command checks the complete data flow on small fixtures. Its selection
is marked smoke_only and cannot authorize T3-b. No training is started here.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from scipy.stats import wilcoxon

from Benchmark_zeroShot import _audit
from mkp_artifacts import instance_sha, sha256_file, source_provenance, write_json
from mkp_config import config_from_dict, resolve_path
from mkp_env import MKPEnv, MKPInstances, generate_instances, instances_from_dicts
from mkp_evaluation import paired_seed, write_csv
from mkp_Model import build_model
from mkp_reference import milp_reference
from read_mat import read_mat

ETAS, GAMMAS, ALPHAS = (.25, .5, 1., 2.), (.1, .3, 1.), (1., 1.25)
N100, EXTRA, CHU = "synthetic_N100", "synthetic_extrapolation", "chu"
EXTRA_SEEDS = {250: 20261105250, 500: 20261105500}
FOLD_SEED = 20261105
DEFAULT_F0_REFERENCE = "results/t1_fixed180/20261102_140130_251675/per_instance.csv"


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _stats(values):
    a = np.asarray(values, dtype=float)
    p = 1. if not np.any(np.abs(a) > 1e-12) else float(wilcoxon(a, zero_method="wilcox").pvalue)
    return {"count": len(a), "mean_delta_pp": float(a.mean()), "median_delta_pp": float(np.median(a)),
            "wins": int((a < -1e-9).sum()), "ties": int((np.abs(a) <= 1e-9).sum()),
            "losses": int((a > 1e-9).sum()), "wilcoxon_p": p}


def _cell(record):
    return record["M"], record["tightness"]


def _record(dataset, key, inst, reference=None, kind="none", proven=False, reference_sha=None):
    ratio = float(inst.c.sum() / inst.r.sum())
    tightness = min((.25, .5, .75), key=lambda x: abs(x - ratio))
    return {"dataset": dataset, "id": str(key), "N": inst.N, "M": inst.M, "tightness": tightness,
            "reference": reference, "reference_kind": kind, "reference_proven": bool(proven),
            "reference_sha256": reference_sha, "instance_sha256": instance_sha(inst), "inst": inst}


def load_data(cfg, limit=0, smoke_items=0):
    """Reproduce t1r_common.synthetic's generator sequence and MILP cache keys.

    Root LPs are computed separately per instance by Scanner, so their timings
    exclude reference solving/cache IO. Larger N uses separate fixed data seeds.
    """
    data, manifests = {N100: [], EXTRA: [], CHU: []}, []
    g = torch.Generator().manual_seed(cfg.train.val_seed)
    for m in (5, 10):
        n = smoke_items or 100
        batch = generate_instances(64, n, m, cfg.env.train_tightness, cfg.env.weight_max,
                                   cfg.env.profit_noise, generator=g)
        count = min(limit, 64) if limit else 64
        ref_batch = (MKPInstances(batch.p[:count], batch.r[:count], batch.c[:count]) if limit else batch)
        refs = milp_reference(ref_batch, cfg.train.val_milp_time_limit, cfg.train.val_mip_rel_gap,
                              cfg.train.reference_cache)
        manifests.append({"dataset": N100, "M": m, "N": n, "seed": cfg.train.val_seed,
                          "data_sha256": instance_sha(ref_batch), "reference_sha256": refs["sha256"],
                          "reference_path": refs["path"], "reference_proven_count": int(refs["optimal"].sum())})
        for b in range(count):
            inst = MKPInstances(batch.p[b:b+1], batch.r[b:b+1], batch.c[b:b+1])
            data[N100].append(_record(N100, f"syn100_M{m}_{b:03d}", inst, float(refs["obj"][b]),
                                      "MILP optimum" if refs["optimal"][b] else "MILP incumbent",
                                      refs["optimal"][b], refs["sha256"]))
    for nominal_n, seed in EXTRA_SEEDS.items():
        g = torch.Generator().manual_seed(seed)
        for m in (5, 10):
            batch = generate_instances(32, smoke_items or nominal_n, m, cfg.env.train_tightness,
                                       cfg.env.weight_max, cfg.env.profit_noise, generator=g)
            count = min(limit, 32) if limit else 32
            manifests.append({"dataset": EXTRA, "nominal_N": nominal_n, "M": m, "seed": seed,
                              "data_sha256": instance_sha(batch), "count_evaluated": count})
            for b in range(count):
                inst = MKPInstances(batch.p[b:b+1], batch.r[b:b+1], batch.c[b:b+1])
                rec = _record(EXTRA, f"syn{nominal_n}_M{m}_{b:03d}", inst)
                rec["nominal_N"] = nominal_n
                data[EXTRA].append(rec)
    path = resolve_path(cfg.eval.mat_file)
    chu = [d for d in read_mat(path) if d["M"] in (5, 10) and d["N"] in (100, 250, 500)]
    chu = chu[:limit] if limit else chu
    for d in chu:
        inst = instances_from_dicts([d])
        if smoke_items:
            # A fixture is no longer a Chu benchmark instance and has no usable
            # best-known value. Keep that distinction in every row and manifest.
            inst = MKPInstances(inst.p[:, :smoke_items], inst.r[:, :, :smoke_items],
                                torch.ceil(.5 * inst.r[:, :, :smoke_items].sum(2)))
        rec = _record(CHU, d["id"], inst, None if smoke_items else float(d["best"]),
                      "smoke fixture; no benchmark reference" if smoke_items else "Chu best known")
        rec["nominal_N"] = d["N"]
        data[CHU].append(rec)
    manifests.append({"dataset": CHU, "path": path, "sha256": sha256_file(path),
                      "count_evaluated": len(chu), "modified_smoke_fixture": bool(smoke_items)})
    if not data[CHU]:
        raise ValueError("No selected Chu instances")
    return data, manifests


class Scanner:
    def __init__(self, cfg, data, device, output):
        self.cfg, self.data, self.device, self.output = cfg, data, device, output
        self.cache, self.pre, self.catalog, self.primary, self.auxiliary = {}, {}, {}, {}, {}
        self.root_env = MKPEnv(cfg.env)
        self.solution_stream = (output / "solutions.jsonl").open("w", encoding="utf-8")
        self.progress_stream = (output / "progress.jsonl").open("w", encoding="utf-8")
        (output / "completed_configs").mkdir(exist_ok=True)

    def close(self):
        self.solution_stream.close()
        self.progress_stream.close()

    def spec(self, *, anchor="root", source="root", eta=None, gamma=None, interval=8, alpha=1.):
        cfg = copy.deepcopy(self.cfg)
        cfg.model.theta_groups = ["price", "alpha"]
        if gamma is not None:
            cfg.model.theta_groups.append("block")
            cfg.model.block_gamma_init = float(gamma)
        if eta is not None:
            cfg.model.theta_groups.append("lp")
            cfg.model.lp_eta_init = float(eta)
        cfg.model.price_anchor, cfg.model.lp_source = anchor, source
        cfg.model.lp_interval, cfg.model.alpha_init = interval, float(alpha)
        cfg.model.theta_update, cfg.model.theta_interval = "step", 1
        cfg.model.untrained_random_heads = False
        return cfg

    def key(self, cfg):
        return _hash({"model": cfg.to_dict()["model"], "env": cfg.to_dict()["env"]})

    def register(self, name, cfg, primary=False):
        key = self.key(cfg)
        self.catalog[key] = {"model": cfg.to_dict()["model"], "env": cfg.to_dict()["env"]}
        (self.primary if primary else self.auxiliary)[name] = key
        return cfg

    @torch.no_grad()
    def evaluate(self, cfg, dataset=N100):
        key = self.key(cfg)
        slot = (key, dataset)
        if slot in self.cache:
            return self.cache[slot]
        label = next((name for name, value in {**self.primary, **self.auxiliary}.items() if value == key), key[:10])
        print(f"[scan] {label}: {dataset}, {len(self.data[dataset])} instances", flush=True)
        definition = {"model": cfg.to_dict()["model"], "env": cfg.to_dict()["env"]}
        self.catalog.setdefault(key, definition)
        write_json(self.output / "configs" / (key + ".json"), definition)
        model, env = build_model(cfg).to(self.device).eval(), MKPEnv(cfg.env)
        rows = {}
        for rec in self.data[dataset]:
            rid = (dataset, rec["id"])
            if rid not in self.pre:
                inst = rec["inst"].to(self.device)
                _sync(self.device)
                started = time.perf_counter()
                pre = self.root_env.precompute(inst)
                _sync(self.device)
                self.pre[rid] = (inst, pre, time.perf_counter() - started)
            inst, pre, root_time = self.pre[rid]
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
                baseline_memory = torch.cuda.memory_allocated(self.device)
            _sync(self.device)
            started = time.perf_counter()
            out = model(env, inst, num_samples=1, decode="greedy", pre=pre, theta_override="fixed")
            _sync(self.device)
            wall = time.perf_counter() - started
            lp = float(out["lp_time_s"])
            if float(out["root_lp_time_s"]) != 0:
                raise RuntimeError("Shared root precompute was unexpectedly charged again")
            peak = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else ""
            peak_extra = peak - baseline_memory if self.device.type == "cuda" else ""
            _audit(out["solution"], inst)
            objective, ub = float(out["profit"][0]), float(pre[1][0])
            if abs(objective - float((out["solution"] * inst.p).sum())) > 1e-4:
                raise RuntimeError("Solution vector and reported objective disagree")
            row = {k: v for k, v in rec.items() if k != "inst"}
            row.update({"config_sha256": key, "objective": objective, "root_lp_ub": ub,
                        "gap_pct": 100 * (1 - objective / rec["reference"]) if rec["reference"] else "",
                        "root_lp_time_s": root_time, "residual_lp_time_s": lp,
                        "lp_time_s": root_time + lp, "construct_time_s": max(0., wall - lp),
                        "policy_time_s": wall, "total_time_s": root_time + wall,
                        "lp_refreshes": int(out["lp_refreshes"]), "peak_allocated_bytes": peak,
                        "peak_extra_allocated_bytes": peak_extra, "memory_device": str(self.device),
                        "feasible": True, "maximal": True})
            rows[rec["id"]] = row
            self.progress_stream.write(json.dumps(row) + "\n")
            self.progress_stream.flush()
            self.solution_stream.write(json.dumps({"dataset": dataset, "id": rec["id"],
                "config_sha256": key, "objective": objective,
                "solution": out["solution"][0].to(torch.int8).cpu().tolist()}) + "\n")
        self.solution_stream.flush()
        write_csv(self.output / "completed_configs" / (key + "_" + dataset + ".csv"), list(rows.values()))
        self.cache[slot] = rows
        return rows

    def score(self, cfg, ids=None):
        rows = self.evaluate(cfg, N100)
        ids = ids if ids is not None else rows.keys()
        return float(np.mean([rows[i]["gap_pct"] for i in ids]))

    def best(self, candidates, ids=None):
        # Python min is stable: ties retain the first declared grid candidate.
        return min(candidates, key=lambda pair: self.score(pair[1], ids))

    def evaluate_from_key(self, key, dataset=N100):
        cfg = copy.deepcopy(self.cfg)
        for field, value in self.catalog[key]["model"].items():
            setattr(cfg.model, field, value)
        return self.evaluate(cfg, dataset)


def choose_chain(scan, ids=None, prefix="main", primary=False):
    """The entire dependent selection chain may be refit on a cross-fit half."""
    f0, anchor = scan.spec(), scan.spec(anchor="residual")
    roots = [(f"R{eta:g}", scan.spec(eta=eta)) for eta in ETAS]
    residuals = [(f"S{eta:g}", scan.spec(source="residual", eta=eta)) for eta in ETAS]
    if primary:
        for name, cfg in [("F0", f0), ("A", anchor), *roots, *residuals]:
            scan.register(name, cfg, primary=True)
            scan.evaluate(cfg)
    s_name, s_best = scan.best(residuals, ids)
    as_cfg = copy.deepcopy(s_best)
    as_cfg.model.price_anchor = "residual"
    scan.register("AS" if primary else prefix + "_AS", as_cfg, primary)
    scan.evaluate(as_cfg)
    k1_base_name, k1_base = scan.best([("A", anchor), (s_name, s_best), ("AS", as_cfg)], ids)
    k1 = copy.deepcopy(k1_base)
    k1.model.lp_interval = 1
    scan.register("K1" if primary else prefix + "_K1", k1, primary)
    scan.evaluate(k1)
    g3_name, g3_best = scan.best([("A", anchor), *roots, *residuals, ("AS", as_cfg), ("K1", k1)], ids)
    blocks = [(f"G{gamma:g}", scan.spec(gamma=gamma)) for gamma in GAMMAS]
    if primary:
        for name, cfg in blocks:
            scan.register(name, cfg, primary=True)
            scan.evaluate(cfg)
    gamma_name, gamma_best = scan.best(blocks, ids)
    best = copy.deepcopy(g3_best)
    if "block" not in best.model.theta_groups:
        best.model.theta_groups.append("block")
    best.model.block_gamma_init = gamma_best.model.block_gamma_init
    scan.register("BEST" if primary else prefix + "_BEST", best, primary)
    scan.evaluate(best)
    return best, {"s_eta": s_best.model.lp_eta_init, "s_arm": s_name, "k1_base": k1_base_name,
                  "g3_best_arm": g3_name, "g3_best_config_sha256": scan.key(g3_best),
                  "gamma": gamma_best.model.block_gamma_init, "gamma_arm": gamma_name,
                  "selection_dataset": N100, "selection_count": len(ids) if ids is not None else len(scan.data[N100])}


def _strata(row):
    d, n, m, a = row["dataset"], row.get("nominal_N", row["N"]), row["M"], row["tightness"]
    return ((d, "all", "all", "all"), (d, n, "all", "all"),
            (d, "all", m, a), (d, n, m, a))


def scan_report(scan):
    rows, groups = [], defaultdict(list)
    for name, key in scan.primary.items():
        cfg = copy.deepcopy(scan.cfg)
        for field, value in scan.catalog[key]["model"].items():
            setattr(cfg.model, field, value)
        for dataset in (N100, EXTRA, CHU):
            control = scan.evaluate_from_key(scan.primary["F0"], dataset)
            for rid, original in scan.evaluate(cfg, dataset).items():
                base = control[rid]
                denom = original["reference"] or original["root_lp_ub"]
                row = {"arm": name, **original,
                       "paired_delta_pp": 100 * (base["objective"] - original["objective"]) / denom,
                       "paired_metric": "reference-gap difference (pp)" if original["reference"] else
                                        "F0 minus arm objective / root LP UB (pp)",
                       "objective_gain_vs_f0": original["objective"] - base["objective"],
                       "objective_gain_ub_pp": 100 * (original["objective"] - base["objective"]) / original["root_lp_ub"]}
                rows.append(row)
                for stratum in _strata(row):
                    groups[(name, *stratum)].append(row)
    summary = []
    for (arm, dataset, n, m, a), values in groups.items():
        gaps = [r["gap_pct"] for r in values if r["gap_pct"] != ""]
        rec = {"arm": arm, "dataset": dataset, "N": n, "M": m, "tightness": a,
               "reference": "MILP reference; optimum/incumbent flagged per instance" if dataset == N100 else values[0]["reference_kind"],
               "reference_proven_count": sum(r["reference_proven"] for r in values),
               "paired_metric": values[0]["paired_metric"],
               **_stats([r["paired_delta_pp"] for r in values]),
               "mean_gap_pct": float(np.mean(gaps)) if gaps else "",
               "p90_gap_pct": float(np.quantile(gaps, .9)) if gaps else "",
               "worst_gap_pct": float(np.max(gaps)) if gaps else "",
               "mean_objective_gain_ub_pp": float(np.mean([r["objective_gain_ub_pp"] for r in values]))}
        for field in ("root_lp_time_s", "residual_lp_time_s", "lp_time_s", "construct_time_s", "total_time_s"):
            rec["mean_" + field] = float(np.mean([r[field] for r in values]))
        mem = [r["peak_allocated_bytes"] for r in values if r["peak_allocated_bytes"] != ""]
        rec["max_peak_allocated_bytes"] = max(mem) if mem else ""
        summary.append(rec)
    return rows, summary


def crossfit(scan):
    records = scan.data[N100]
    strata = defaultdict(list)
    for rec in records:
        strata[_cell(rec)].append(rec["id"])
    assignments, fold_counts = {}, [0, 0]
    for cell, ids in sorted(strata.items()):
        order = np.random.default_rng(paired_seed(FOLD_SEED, str(cell), "t3a_crossfit")).permutation(len(ids))
        start_fold = 0 if fold_counts[0] <= fold_counts[1] else 1
        for pos, index in enumerate(order):
            fold = (pos + start_fold) % 2
            assignments[ids[int(index)]] = fold
            fold_counts[fold] += 1
    # Singleton smoke strata may all land in fold 0. Alternate their assignment
    # only in smoke-sized data to exercise both halves; full strata retain balance.
    if len(set(assignments.values())) < 2 and len(records) >= 2:
        assignments = {r["id"]: i % 2 for i, r in enumerate(records)}
    if len(set(assignments.values())) < 2:
        return [], [], {"status": "insufficient_data", "fold_seed": FOLD_SEED, "assignments": assignments}
    output, fitted = [], []
    id_record = {r["id"]: r for r in records}
    for heldout in (0, 1):
        train = [rid for rid, fold in assignments.items() if fold != heldout]
        test = [rid for rid, fold in assignments.items() if fold == heldout]
        families = {family: [(eta, scan.spec(source="root" if family == "R" else "residual", eta=eta))
                             for eta in ETAS] for family in ("R", "S")}
        family = min(families, key=lambda name: scan.score(scan.best(families[name], train)[1], train))
        fold_best, choices = choose_chain(scan, train, prefix=f"fold{heldout}")
        alpha_best = []
        for alpha in ALPHAS:
            cfg = copy.deepcopy(fold_best)
            cfg.model.alpha_init = alpha
            scan.register(f"fold{heldout}_BEST_alpha{alpha:g}", cfg)
            alpha_best.append((alpha, cfg))
        grids = {"eta": families[family], "gamma": [(g, scan.spec(gamma=g)) for g in GAMMAS],
                 "alpha_F0": [(a, scan.spec(alpha=a)) for a in ALPHAS], "alpha_BEST": alpha_best}
        for coefficient, grid in grids.items():
            for value, cfg in grid:
                scan.register(f"fold{heldout}_{coefficient}_{value:g}", cfg)
            global_value, global_cfg = scan.best(grid, train)
            cell_choices = {}
            for cell in strata:
                fit_ids = [rid for rid in train if _cell(id_record[rid]) == cell]
                cell_choices[cell] = scan.best(grid, fit_ids) if fit_ids else (global_value, global_cfg)
            fitted.append({"heldout_fold": heldout, "coefficient": coefficient, "eta_family": family,
                           "global_value": global_value, "training_ids": train, "heldout_ids": test,
                           "chain_choices": choices if coefficient == "alpha_BEST" else None,
                           "per_stratum": [{"M": c[0], "tightness": c[1], "value": v[0],
                              "global_fallback": not any(_cell(id_record[rid]) == c for rid in train)}
                              for c, v in cell_choices.items()]})
            global_rows = scan.evaluate(global_cfg)
            for rid in test:
                rec = id_record[rid]
                value, cfg = cell_choices[_cell(rec)]
                conditional = scan.evaluate(cfg)[rid]
                global_row = global_rows[rid]
                output.append({"coefficient": coefficient, "heldout_fold": heldout, "id": rid,
                    "M": rec["M"], "N": rec["N"], "tightness": rec["tightness"], "eta_family": family,
                    "global_value": global_value, "stratified_value": value,
                    "global_config_sha256": scan.key(global_cfg), "stratified_config_sha256": scan.key(cfg),
                    "global_objective": global_row["objective"], "stratified_objective": conditional["objective"],
                    "global_gap_pct": global_row["gap_pct"], "stratified_gap_pct": conditional["gap_pct"],
                    "delta_pp": conditional["gap_pct"] - global_row["gap_pct"]})
    groups = defaultdict(list)
    for row in output:
        for cell in (("all", "all"), (row["M"], row["tightness"])):
            groups[(row["coefficient"], *cell)].append(row)
    summary = []
    for (coefficient, m, a), values in groups.items():
        fold_means = [float(np.mean([r["delta_pp"] for r in values if r["heldout_fold"] == fold]))
                      for fold in (0, 1) if any(r["heldout_fold"] == fold for r in values)]
        equal_fold_mean = float(np.mean(fold_means))
        summary.append({"coefficient": coefficient, "M": m, "tightness": a,
                        **_stats([r["delta_pp"] for r in values]),
                        "equal_fold_mean_delta_pp": equal_fold_mean,
                        "amortizable_gain_pp": -equal_fold_mean,
                        "mean_global_gap_pct": float(np.mean([r["global_gap_pct"] for r in values])),
                        "mean_stratified_gap_pct": float(np.mean([r["stratified_gap_pct"] for r in values]))})
    return output, summary, {"status": "complete", "fold_seed": FOLD_SEED, "assignments": assignments,
        "fits": fitted, "protocol": "Two fixed folds stratified by M x tightness. All family, chain and "
        "constant selection uses fit-half only. Both controls use the same coefficient grid and fit half. "
        "eta R/S family is chosen on fit-half, shared by global and stratified controls. alpha_BEST rebuilds "
        "S/AS/K1/G3/gamma/BEST on fit-half; auxiliary configurations are not part of the 16 primary arms. "
        "Missing fit strata in smoke fixtures fall back to the fit-global constant. Negative pp is better."}


def check_f0(scan, path, smoke_items):
    if smoke_items:
        return {"status": "not_checked_smoke_fixture"}
    reference = Path(resolve_path(path))
    if not reference.is_file():
        return {"status": "not_checked_reference_missing", "path": str(reference)}
    with reference.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    stored = {}
    for row in rows:
        if row.get("arm") == "fixed" and row.get("stage") == "greedy":
            stored[str(row["id"])] = float(row.get("objective") or row.get("greedy_obj"))
    current = scan.evaluate_from_key(scan.primary["F0"], CHU)
    missing, mismatches = [], []
    for rid, row in current.items():
        if rid not in stored:
            missing.append(rid)
        elif abs(row["objective"] - stored[rid]) > 1e-5:
            mismatches.append(rid)
    return {"status": "checked" if not missing else "partially_checked", "path": str(reference),
            "sha256": sha256_file(reference), "count": len(current), "mismatch_count": len(mismatches),
            "mismatch_ids": mismatches, "missing_reference_ids": missing,
            "mean_gap_pct": float(np.mean([r["gap_pct"] for r in current.values()])),
            "historical_full180_mean_gap_pct": .61070129}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="runs/t1p_dynamic_k8_s1234/config.json")
    parser.add_argument("--output", default="results/t3a_scan")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0, help="smoke only: examples per synthetic M/N group and total Chu")
    parser.add_argument("--smoke-items", type=int, default=0, help="with --limit, replace all N by small fixture size")
    parser.add_argument("--f0_reference", default=DEFAULT_F0_REFERENCE)
    args = parser.parse_args(argv)
    if min(args.limit, args.smoke_items) < 0 or args.smoke_items and not args.limit:
        parser.error("Limits must be nonnegative; --smoke-items requires --limit")
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("--device must be cpu or cuda:<index>")
    cfg_path = Path(resolve_path(args.config))
    cfg = config_from_dict(json.loads(cfg_path.read_text(encoding="utf-8")))
    if cfg.device.seed != 1234 or cfg.train.val_sizes != [100] or cfg.train.val_dims != [5, 10] or cfg.train.val_instances != 64:
        raise ValueError("T3-a requires seed1234 and the existing N100/M5,10 x64 synthetic validation set")
    if cfg.env.feasibility_mode != "hard_mask" or not cfg.model.score_normalization:
        raise ValueError("T3-a requires the existing normalized hard-mask executor")
    dest = Path(resolve_path(args.output))
    if dest.exists():
        raise FileExistsError(f"Preserve existing results: {dest}")
    dest.mkdir(parents=True)
    (dest / "configs").mkdir()
    meta = {"schema": "t3a_scan_v1", "status": "running", "smoke_only": bool(args.limit or args.smoke_items),
            "arguments": vars(args), "source_config": str(cfg_path), "source_config_sha256": sha256_file(cfg_path),
            "source": source_provenance(), "config": cfg.to_dict(), "selection_dataset": N100,
            "grid_order": ["F0", "A", *[f"R{e:g}" for e in ETAS], *[f"S{e:g}" for e in ETAS],
                           "AS", "K1", *[f"G{g:g}" for g in GAMMAS], "BEST"],
            "selection": "All dependent choices use synthetic N100 only. Ties retain grid order. "
                         "G3 candidates are A, R, S, AS, K1; no Chu or extrapolation selection.",
            "residual_lp": {"iters": cfg.env.residual_lp_iters, "lr": cfg.env.residual_lp_lr,
                            "note": "Default raised from 30 to 300 to pass required HiGHS accuracy checks."},
            "metrics": "N100 uses MILP reference (incumbents identified); Chu uses best-known. Larger N "
                       "has no MILP/gap reference: paired differences are (F0 objective - arm objective)/rootLP UB x100. "
                       "Negative pp means improvement. LP upper bounds are not integer optima.",
            "statistics": "Two-sided Wilcoxon, unadjusted p-values; pair unit is an instance. No training-seed replication.",
            "progress": "Per-instance progress.jsonl and completed_configs CSVs preserve partial results; no automatic resume.",
            "timing": "Root LP shared per instance but charged once per configuration. Residual LP measured "
                      "inside the policy; construction excludes residual LP. Audits, data/reference IO and output excluded. "
                      "CUDA peak allocation and extra allocation are reported; CPU GPU-memory fields are blank."}
    write_json(dest / "meta.json", meta)
    scan = None
    try:
        data, manifests = load_data(cfg, args.limit, args.smoke_items)
        meta["data"] = manifests
        write_json(dest / "meta.json", meta)
        scan = Scanner(cfg, data, device, dest)
        best, choices = choose_chain(scan, primary=True)
        if list(scan.primary) != meta["grid_order"] or len(scan.primary) != 16:
            raise RuntimeError("Primary scan order/count changed")
        print("Synthetic N100 choices: " + json.dumps(choices), flush=True)
        # Freeze primary choices before looking at any extrapolation/Chu result.
        write_json(dest / "configs" / "primary_choices.json", choices)
        for name, key in scan.primary.items():
            for dataset in (EXTRA, CHU):
                scan.evaluate_from_key(key, dataset)
            print(f"Completed {name}: synthetic N100, extrapolation and Chu", flush=True)
        rows, summary = scan_report(scan)
        write_csv(dest / "per_instance.csv", rows)
        write_csv(dest / "summary.csv", summary)
        write_csv(dest / "paired_vs_f0.csv", [{k: r[k] for k in ("arm", "dataset", "id", "N", "M", "tightness",
            "paired_metric", "paired_delta_pp", "objective_gain_vs_f0", "objective_gain_ub_pp")} for r in rows])
        cross_rows, cross_summary, folds = crossfit(scan)
        write_csv(dest / "crossfit_per_instance.csv", cross_rows)
        write_csv(dest / "crossfit_summary.csv", cross_summary)
        write_json(dest / "crossfit_folds.json", folds)
        # Initial alpha for future T3-b is calibrated globally, separately from
        # the 16 primary arms, all of which retain alpha=1.
        alpha_variants = []
        for value in ALPHAS:
            variant = copy.deepcopy(best)
            variant.model.alpha_init = value
            scan.register(f"BEST_alpha_global_{value:g}", variant)
            alpha_variants.append((value, variant))
        alpha, initial = scan.best(alpha_variants)
        initial_rows = [{"arm": f"BEST_alpha_global_{value:g}", **r} for value, variant in alpha_variants
                        for r in scan.evaluate(variant).values()]
        write_csv(dest / "alpha_global_diagnostic.csv", initial_rows)
        f0_check = check_f0(scan, args.f0_reference, args.smoke_items)
        if f0_check.get("mismatch_count", 0):
            meta["f0_regression"] = f0_check
            raise RuntimeError("F0 historical objectives changed; inspect f0_regression before interpreting results")
        best_syn = next(r for r in summary if r["arm"] == "BEST" and r["dataset"] == N100 and r["N"] == r["M"] == r["tightness"] == "all")
        ext = [r for r in summary if r["arm"] == "BEST" and r["dataset"] == EXTRA and r["N"] != "all" and r["M"] == r["tightness"] == "all"]
        selection = {"schema": "t3a_selection_v1", "status": "complete", "smoke_only": meta["smoke_only"],
            "selection_dataset": N100, "seed": 1234, "source_config": str(cfg_path),
            "data": manifests,
            "source_config_sha256": meta["source_config_sha256"], "choices": choices,
            "selected_global": {"scan_arm": "BEST", "model": initial.to_dict()["model"], "env": initial.to_dict()["env"],
                "alpha_calibration": "global BEST alpha in {1,1.25}, synthetic N100 only; auxiliary diagnostic. "
                                     "Primary 16-arm BEST remains alpha=1.", "alpha_global_value": alpha},
            "scan_gate": {"best_vs_f0_pp": best_syn["mean_delta_pp"], "wilcoxon_p": best_syn["wilcoxon_p"],
                          "extrapolation_by_N": ext, "extrapolation_non_worse": bool(ext) and all(r["mean_delta_pp"] <= 0 for r in ext)},
            "crossfit_gate": {r["coefficient"]: r for r in cross_summary if r["M"] == r["tightness"] == "all"},
            "f0_regression": f0_check,
            "decision": "Measurement only; Claude/user decide adoption and training. This script never starts T3-b."}
        write_json(dest / "selection.json", selection)
        write_json(dest / "configs" / "selection.json", selection)
        write_json(dest / "configs" / "catalog.json", {"primary": scan.primary, "auxiliary": scan.auxiliary,
                                                         "configurations": scan.catalog})
        for name, key in scan.primary.items():
            write_json(dest / "configs" / (name + ".json"), {"model": scan.catalog[key]["model"], "env": scan.catalog[key]["env"]})
        auxiliary = [{"dataset": dataset, **r} for (key, dataset), entries in scan.cache.items()
                     if key not in set(scan.primary.values()) for r in entries.values()]
        write_csv(dest / "auxiliary_per_instance.csv", auxiliary)
        meta.update({"status": "complete", "data": manifests, "f0_regression": f0_check,
                     "primary_config_count": 16, "auxiliary_unique_config_count": len(scan.catalog) - len(set(scan.primary.values())),
                     "selection_sha256": sha256_file(dest / "selection.json")})
    except Exception as exc:
        meta.update({"status": "failed", "error": repr(exc)})
        raise
    finally:
        if scan is not None:
            scan.close()
        write_json(dest / "meta.json", meta)
    print(f"T3-a results: {dest}")
    return dest


if __name__ == "__main__":
    main()
