"""User-run T4-a SMS grid. Full runs select on synthetic N100 only; no training.

    python scripts/run_t4a_sms.py --device cuda:0
    python scripts/run_t4a_sms.py --limit 2 --device cuda:0 --output results/t4_code_checks/sms_smoke

--limit 2 uses two real N100 validation instances and two real Chu instances,
one of each M=5/10. Every limited run is smoke_only and unusable by T4-b.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from scipy.stats import wilcoxon

from Benchmark_zeroShot import evaluate_stage
from mkp_artifacts import instance_sha, sha256_file, source_provenance, write_json
from mkp_config import config_from_dict, get_device, resolve_path
from mkp_env import MKPEnv, MKPInstances, generate_instances, instances_from_dicts
from mkp_evaluation import write_csv
from mkp_Model import build_model
from mkp_reference import milp_reference
from read_mat import read_mat

PREFIXES = (1, 4, 16, 64, 256)
N100, CHU = "synthetic_N100", "chu"
HYBRID_RULE = {
    "user_confirmed_rule": True,
    "origin": "User-approved clarification; the guidance does not define H* selection.",
    "selection": "Minimum mean gap delta against S* at the hybrid's native total K; "
                 "ties choose smaller total K, then declared grid order.",
    "chu_extension": "Keep selected Ks; increase Ktheta to 256/Ks and report the resulting "
                     "theta-major trajectory prefixes. This changes Ktheta from the native selection arm.",
}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _record(dataset, key, inst, reference, kind, proven=False):
    ratio = float(inst.c.sum() / inst.r.sum())
    return {"dataset": dataset, "id": str(key), "M": inst.M, "N": inst.N,
            "tightness": min((.25, .5, .75), key=lambda a: abs(a - ratio)),
            "reference": float(reference), "reference_kind": kind,
            "reference_proven": bool(proven), "instance_sha256": instance_sha(inst), "inst": inst}


def load_data(cfg, limit=0, smoke_items=0):
    """Same two generated batches/cache keys as T3-a in a full run."""
    datasets, manifests = {N100: [], CHU: []}, []
    rng = torch.Generator().manual_seed(cfg.train.val_seed)
    count = 128 if not limit else min(limit, 128)
    per_dim = ((count + 1) // 2, count // 2)
    for m, take in zip((5, 10), per_dim):
        batch = generate_instances(64, smoke_items or 100, m, cfg.env.train_tightness,
                                   cfg.env.weight_max, cfg.env.profit_noise, generator=rng)
        selected = MKPInstances(batch.p[:take], batch.r[:take], batch.c[:take])
        refs = milp_reference(selected, cfg.train.val_milp_time_limit, cfg.train.val_mip_rel_gap,
                              cfg.train.reference_cache)
        manifests.append({"dataset": N100, "M": m, "N": selected.N, "seed": cfg.train.val_seed,
                          "count": take, "data_sha256": instance_sha(selected),
                          "reference_path": refs["path"], "reference_sha256": refs["sha256"],
                          "reference_proven_count": int(refs["optimal"].sum())})
        for b in range(take):
            inst = MKPInstances(batch.p[b:b+1], batch.r[b:b+1], batch.c[b:b+1])
            datasets[N100].append(_record(N100, f"syn100_M{m}_{b:03d}", inst, refs["obj"][b],
                                          "MILP optimum" if refs["optimal"][b] else "MILP incumbent",
                                          refs["optimal"][b]))
    path = resolve_path(cfg.eval.mat_file)
    chu = [d for d in read_mat(path) if d["M"] in (5, 10) and d["N"] in (100, 250, 500)]
    if not limit:
        counts = {(m, n): sum(d["M"] == m and d["N"] == n for d in chu)
                  for m in (5, 10) for n in (100, 250, 500)}
        if any(count != 30 for count in counts.values()) or len({d["id"] for d in chu}) != 180:
            raise ValueError(f"T4-a requires 180 distinct Chu instances, 30 per M/N group; found {counts}")
    if limit:
        # Pick both dimensions first instead of silently testing only M=5.
        first = [next(d for d in chu if d["M"] == m and d["N"] == 100) for m in (5, 10)]
        chosen = {d["id"] for d in first}
        chu = (first + [d for d in chu if d["id"] not in chosen])[:limit]
    for d in chu:
        inst = instances_from_dicts([d])
        if smoke_items:
            inst = MKPInstances(inst.p[:, :smoke_items], inst.r[:, :, :smoke_items],
                                torch.ceil(.5 * inst.r[:, :, :smoke_items].sum(2)))
            # A cut fixture has no Chu best-known reference; use its root LP only
            # as an explicitly marked smoke denominator, never as a benchmark gap.
            reference = float(MKPEnv(cfg.env).precompute(inst)[1][0])
            kind = "smoke fixture LP bound; not a Chu benchmark gap"
        else:
            reference, kind = float(d["best"]), "Chu best known; not proven optimum"
        datasets[CHU].append(_record(CHU, d["id"], inst, reference, kind))
    manifests.append({"dataset": CHU, "path": path, "sha256": sha256_file(path),
                      "count": len(chu), "modified_fixture": bool(smoke_items)})
    if not datasets[CHU]:
        raise ValueError("No Chu instances selected")
    return datasets, manifests


def theta_spec(dims, scale):
    return {"name": f"theta_{dims}_s{scale:g}", "family": "theta", "dims": dims,
            "scale": float(scale), "temperature": .125, "stage": "theta256", "budget": 256}


def sample_spec(temperature):
    return {"name": f"sample_T{temperature:g}", "family": "sample", "dims": "ua",
            "scale": 1., "temperature": float(temperature), "stage": "sample256", "budget": 256}


class SMSRunner:
    def __init__(self, cfg, data, device, dest, resume=False):
        self.cfg, self.data, self.device, self.dest = cfg, data, device, dest
        self.env = MKPEnv(cfg.env)
        self.model = build_model(cfg).to(device).eval()
        self.pre, self.results, self.catalog, self.centers = {}, {}, {}, {}
        self.reported = set()
        self.progress = (dest / "progress.jsonl").open("a" if resume else "w", encoding="utf-8")
        self.candidates = (dest / "candidates.jsonl").open("a" if resume else "w", encoding="utf-8")
        self.rows, self.primary_names = [], []
        self.partial = defaultdict(dict)
        if resume:
            self._recover_smoke()

    def close(self):
        self.progress.close()
        self.candidates.close()

    def key(self, spec):
        evaluation = self.cfg.to_dict()["eval"]
        if "+ILS" not in spec["stage"]:
            # Inactive search options do not change a construction-only pool.
            evaluation = {k: v for k, v in evaluation.items() if not k.startswith(("ls_", "ils_"))}
        return _hash({"spec": {k: v for k, v in spec.items() if k != "name"},
                      "model": self.cfg.to_dict()["model"], "env": self.cfg.to_dict()["env"],
                      "eval": evaluation})

    def _recover_smoke(self):
        """Explicit limited-run recovery; never resumes or authorizes a full experiment."""
        records = {}
        for line in (self.dest / "progress.jsonl").read_text(encoding="utf-8").splitlines():
            saved = json.loads(line)
            first = saved["prefixes"][0]
            spec = saved.get("spec", {"name": saved["arm"], "family": first["family"],
                "dims": first["sms_dims"], "scale": first["sms_scale"],
                "temperature": first["effective_temperature"], "stage": saved["stage"],
                "budget": saved["pipeline_budget"]})
            if spec["family"] == "hybrid" and "item_k" not in spec:
                kt, ki = re.match(r"hyb([0-9]+)x([0-9]+)", spec["stage"]).groups()
                spec.update(theta_k=int(kt), item_k=int(ki))
            key = self.key(spec)
            expected = next((r for r in self.data[saved["dataset"]] if r["id"] == saved["id"]), None)
            if expected is None or first["instance_sha256"] != expected["instance_sha256"]:
                raise ValueError("Smoke recovery instance identity changed")
            rows = [{**r, "config_sha256": key} for r in saved["prefixes"]]
            self.partial[(saved["dataset"], key)][saved["id"]] = rows
            self.catalog[saved["arm"]] = spec
            records[(saved["dataset"], key, saved["arm"], saved["id"])] = rows
        for (dataset, key), by_id in self.partial.items():
            expected_ids = {r["id"] for r in self.data[dataset]}
            if set(by_id) == expected_ids:
                rows = [r for rec in self.data[dataset] for r in by_id[rec["id"]]]
                self.results[(dataset, key)] = rows
                self.reported.add((dataset, key, rows[0]["arm"]))
        self.rows = [r for rows in records.values() for r in rows]
        for line in (self.dest / "candidates.jsonl").read_text(encoding="utf-8").splitlines():
            saved = json.loads(line)
            n = saved["shape"][1]
            first = np.frombuffer(base64.b64decode(saved["candidate_solutions"]), dtype=np.uint8)[:(n+7)//8]
            bits = torch.from_numpy(np.unpackbits(first, bitorder="little")[:n].copy())
            self.centers.setdefault((saved["dataset"], saved["id"]), bits)
        print(f"[t4a smoke recovery] reused {len(records)} completed configuration/instance pools", flush=True)

    def evaluate(self, spec, dataset, alias=None, prefixes=None):
        key = self.key(spec)
        name = alias or spec["name"]
        self.catalog[name] = spec
        cache_key = (dataset, key)
        if cache_key in self.results:
            rows = [{**r, "arm": name} for r in self.results[cache_key]]
            if (dataset, key, name) not in self.reported:
                self.rows.extend(rows)
                self.reported.add((dataset, key, name))
            return rows
        prefixes = prefixes or (PREFIXES if spec["family"] != "hybrid" or dataset == CHU else (spec["budget"],))
        print(f"[t4a] {dataset}: {name}, {len(self.data[dataset])} instances, {spec['stage']}", flush=True)
        all_rows = []
        for rec in self.data[dataset]:
            if rec["id"] in self.partial.get(cache_key, {}):
                all_rows.extend(self.partial[cache_key][rec["id"]])
                continue
            rid = (dataset, rec["id"])
            if rid not in self.pre:
                inst = rec["inst"].to(self.device)
                _sync(self.device)
                start = time.perf_counter()
                pre = self.env.precompute(inst)
                _sync(self.device)
                self.pre[rid] = (inst, pre, time.perf_counter() - start)
            inst, pre, root_time = self.pre[rid]
            ec = copy.deepcopy(self.cfg.eval)
            ec.sms_dims, ec.sms_scale, ec.sms_include_center = spec["dims"], spec["scale"], True
            ec.sample_temperature = spec["temperature"] / self.cfg.model.temperature_init
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            result = evaluate_stage(self.model, self.env, inst, ec, self.device,
                                    self.cfg.device.amp and self.device.type == "cuda", pre,
                                    spec["stage"], "fixed", self.cfg.device.seed, rec["id"], root_time,
                                    return_candidates=True, centered_samples=True, sms_tag=0)
            solutions = result.pop("candidate_solutions").detach().cpu().to(torch.uint8)
            objectives = result.pop("candidate_objectives").detach().cpu().double()
            if solutions.shape != (spec["budget"], inst.N) or objectives.shape != (spec["budget"],):
                raise RuntimeError("Evaluator returned an unexpected candidate budget or ordering")
            if not bool(torch.isfinite(objectives).all()) or not bool((objectives >= 0).all()):
                raise RuntimeError("Invalid candidate objectives")
            if rid in self.centers and not torch.equal(solutions[0], self.centers[rid]):
                raise RuntimeError("Included center solution changed between inference families")
            self.centers.setdefault(rid, solutions[0].clone())
            # All inference families use center greedy at trajectory zero.
            if float(objectives.max()) + 1e-6 < float(objectives[0]):
                raise RuntimeError("Best-of-K is below the included center")
            pipeline = {k: result[k] for k in ("time_lp_s", "time_construct_s", "time_refine_s", "time_total_s")}
            peak = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None
            rows = []
            report_prefixes = list(prefixes)
            if dataset == N100 and spec["family"] == "theta" and spec["dims"] == "ua" and spec["scale"] == 1.:
                report_prefixes.append(32)
            for k in sorted(set(report_prefixes)):
                best_index = int(objectives[:k].argmax())
                objective = float(objectives[best_index])
                diversity = float(torch.unique(solutions[:k], dim=0).shape[0] / k) if k in (16, 64) else ""
                row = {key: val for key, val in rec.items() if key != "inst"}
                row.update({"arm": name, "family": spec["family"], "K": k,
                            "objective": objective, "gap_pct": 100 * (1 - objective / rec["reference"]),
                            "center_objective": float(objectives[0]), "unique_solution_ratio": diversity,
                            "pipeline_stage": spec["stage"], "pipeline_budget": spec["budget"],
                            "prefix_runtime_measured": k == spec["budget"],
                            "prefix_theta_starts": (1 if spec["family"] == "sample" else
                                                    math.ceil(k / spec.get("item_k", 1))),
                            "item_samples_per_theta": spec.get("item_k", 1),
                            "sms_dims": spec["dims"], "sms_scale": spec["scale"],
                            "effective_temperature": spec["temperature"], "seed": self.cfg.device.seed,
                            "config_sha256": key, "peak_allocated_bytes": peak,
                            "time_pipeline_lp_s": pipeline["time_lp_s"],
                            "time_pipeline_construct_s": pipeline["time_construct_s"],
                            "time_pipeline_refine_s": pipeline["time_refine_s"],
                            "time_pipeline_total_s": pipeline["time_total_s"],
                            "feasible": result["feasible"], "maximal": result["maximal"]})
                if "+ILS" in spec["stage"]:
                    row["objective"] = float(result["objective"])
                    row["gap_pct"] = 100 * (1 - row["objective"] / rec["reference"])
                    row["unique_solution_ratio"] = ""
                rows.append(row)
            encoded = base64.b64encode(np.packbits(solutions.numpy(), axis=1, bitorder="little").tobytes()).decode()
            self.candidates.write(json.dumps({"dataset": dataset, "id": rec["id"], "arm": name,
                "stage": spec["stage"], "shape": list(solutions.shape), "encoding": "base64-packbits-little-rowwise",
                "candidate_solutions": encoded, "candidate_objectives": objectives.tolist(),
                "best_pipeline_solution": result["solution"]}) + "\n")
            self.candidates.flush()
            progress = {"dataset": dataset, "id": rec["id"], "arm": name, "config_sha256": key,
                        "stage": spec["stage"], "spec": spec, "pipeline_budget": spec["budget"], "prefixes": rows,
                        "pipeline_timing": pipeline, "smoke_only": bool(self.cfg._sms_smoke_only)}
            self.progress.write(json.dumps(progress) + "\n")
            self.progress.flush()
            all_rows.extend(rows)
            print(f"  {rec['id']}: best@{spec['budget']}={rows[-1]['gap_pct']:.4f}%", flush=True)
        self.results[cache_key] = all_rows
        self.reported.add((dataset, key, name))
        self.rows.extend(all_rows)
        write_csv(self.dest / "per_instance.csv", self.rows)
        return all_rows


def _mean_at(rows, k):
    values = [r["gap_pct"] for r in rows if r["K"] == k]
    if not values:
        raise ValueError(f"Missing K={k} measurements")
    return float(np.mean(values))


def choose_theta_sample(runner, specs):
    scores = []
    for spec in specs:
        rows = runner.evaluate(spec, N100)
        scores.append({**spec, "mean_gap16": _mean_at(rows, 16), "mean_gap64": _mean_at(rows, 64)})
    selected = min(scores, key=lambda r: r["mean_gap16"] + r["mean_gap64"])
    return next(s for s in specs if s["name"] == selected["name"]), scores


def _paired(rows, lhs, rhs, dataset, ks=None):
    left = {(r["id"], r["K"]): r for r in rows if r["arm"] == lhs and r["dataset"] == dataset}
    right = {(r["id"], r["K"]): r for r in rows if r["arm"] == rhs and r["dataset"] == dataset}
    out = []
    for key in sorted(left.keys() & right.keys()):
        l, r = left[key], right[key]
        if ks is not None and l["K"] not in ks:
            continue
        out.append({"dataset": dataset, "arm": lhs, "control": rhs, "id": key[0], "K": key[1],
                    "M": l["M"], "N": l["N"], "tightness": l["tightness"],
                    "gap_delta_pp": l["gap_pct"] - r["gap_pct"],
                    "objective_delta": l["objective"] - r["objective"],
                    "arm_pipeline_budget": l["pipeline_budget"], "control_pipeline_budget": r["pipeline_budget"],
                    "pipeline_time_is_budget_matched": l["pipeline_budget"] == r["pipeline_budget"],
                    "pipeline_time_delta_s": l["time_pipeline_total_s"] - r["time_pipeline_total_s"]})
    return out


def reports(rows, theta_name, sample_name, hybrid_name):
    groups = defaultdict(list)
    for r in rows:
        for m, n in (("all", "all"), (r["M"], r["N"])):
            groups[(r["dataset"], r["arm"], r["K"], m, n)].append(r)
    summary = []
    for (dataset, arm, k, m, n), values in groups.items():
        gaps = [r["gap_pct"] for r in values]
        item = {"dataset": dataset, "arm": arm, "K": k, "M": m, "N": n, "count": len(values),
                "mean_gap_pct": float(np.mean(gaps)), "p90_gap_pct": float(np.quantile(gaps, .9)),
                "worst_gap_pct": float(max(gaps)), "pipeline_budget": values[0]["pipeline_budget"],
                "prefix_runtime_measured": values[0]["prefix_runtime_measured"]}
        for field in ("time_pipeline_lp_s", "time_pipeline_construct_s", "time_pipeline_refine_s", "time_pipeline_total_s"):
            item["mean_" + field] = float(np.mean([r[field] for r in values]))
        diversity = [r["unique_solution_ratio"] for r in values if r["unique_solution_ratio"] != ""]
        item["mean_unique_solution_ratio"] = float(np.mean(diversity)) if diversity else ""
        summary.append(item)
    pairs = []
    for dataset, names in ((N100, (theta_name, hybrid_name)),
                            (CHU, ("Theta_star", "Theta_ua_s1_preregistered", "H_star_extended"))):
        control = sample_name if dataset == N100 else "S_star"
        for name in names:
            pairs.extend(_paired(rows, name, control, dataset))
    pairs.extend(_paired(rows, "Theta_star_64+ILS", "S_star_64+ILS", CHU, (64,)))
    paired_summary, grouped = [], defaultdict(list)
    for r in pairs:
        strata = [("all", "all")]
        if r["K"] in (16, 64):
            strata.append((r["M"], r["N"]))
        for m, n in strata:
            grouped[(r["dataset"], r["arm"], r["control"], r["K"], m, n)].append(r)
    for (dataset, arm, control, k, m, n), values in grouped.items():
        delta = np.array([r["gap_delta_pp"] for r in values])
        p = 1. if not np.any(np.abs(delta) > 1e-12) else float(wilcoxon(delta, zero_method="wilcox").pvalue)
        paired_summary.append({"dataset": dataset, "arm": arm, "control": control, "K": k,
            "M": m, "N": n, "count": len(values), "mean_gap_delta_pp": float(delta.mean()),
            "median_gap_delta_pp": float(np.median(delta)), "wins": int((delta < -1e-9).sum()),
            "ties": int((np.abs(delta) <= 1e-9).sum()), "losses": int((delta > 1e-9).sum()),
            "wilcoxon_two_sided_p": p, "p_values_adjusted": False})
    return summary, pairs, paired_summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="runs/t1p_static_s1234/config.json")
    parser.add_argument("--output", default="results/t4a_sms")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0, help="0=full; >=2 means smoke-only total instances per dataset")
    parser.add_argument("--smoke-items", type=int, default=0, help="optional modified fixtures; requires --limit")
    parser.add_argument("--resume-smoke", action="store_true", help="reuse completed pools in an interrupted limited smoke only")
    args = parser.parse_args(argv)
    if args.limit < 0 or args.limit == 1 or args.limit > 128 or args.smoke_items < 0 or (args.smoke_items and not args.limit):
        parser.error("--limit is 0 or 2..128; --smoke-items requires a nonzero limit")
    torch.set_num_threads(2)
    source = Path(resolve_path(args.config))
    cfg = config_from_dict(json.loads(source.read_text(encoding="utf-8")))
    if cfg.device.seed != 1234 or cfg.train.val_instances != 64 or cfg.train.val_sizes != [100] or cfg.train.val_dims != [5, 10]:
        raise ValueError("T4-a requires the inherited seed1234 N100/M5,10 x64 synthetic validation set")
    if cfg.env.feasibility_mode != "hard_mask" or not cfg.model.score_normalization:
        raise ValueError("T4-a requires the normalized hard-mask F0 executor")
    cfg.model.theta_groups, cfg.model.price_anchor, cfg.model.lp_source = ["price", "alpha"], "root", "root"
    cfg.model.alpha_init, cfg.model.theta_sigma_head = 1., False
    cfg.model.untrained_random_heads = False
    cfg._sms_smoke_only = bool(args.limit or args.smoke_items)
    inherited_search = {k: getattr(cfg.eval, k) for k in ("ils_rounds", "ls_max_iters", "ls_pool", "ls_drop_k")}
    if cfg._sms_smoke_only:
        cfg.eval.ils_rounds, cfg.eval.ls_max_iters, cfg.eval.ls_pool, cfg.eval.ls_drop_k = 1, 1, 2, 2
    device = get_device(cfg) if args.device == "auto" else torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("--device must be auto, cpu or cuda:<index>")
    dest = Path(resolve_path(args.output))
    previous = None
    if args.resume_smoke:
        if not args.limit or not (dest / "meta.json").is_file():
            raise ValueError("--resume-smoke requires an existing limited smoke directory")
        previous = json.loads((dest / "meta.json").read_text(encoding="utf-8"))
        if not previous.get("smoke_only") or previous.get("source_config_sha256") != sha256_file(source):
            raise ValueError("Smoke recovery requires the same source config and smoke_only data")
        for field in ("limit", "smoke_items", "device"):
            if previous["arguments"].get(field) != getattr(args, field):
                raise ValueError(f"Smoke recovery argument changed: {field}")
    elif dest.exists():
        raise FileExistsError(f"Preserve existing T4-a results: {dest}")
    else:
        dest.mkdir(parents=True)
    meta = {"schema": "t4a_sms_v1", "status": "running", "smoke_only": cfg._sms_smoke_only,
            "arguments": vars(args), "source_config": str(source), "source_config_sha256": sha256_file(source),
            "source": source_provenance(), "config": cfg.to_dict(), "seed": 1234,
            "selection_dataset": N100, "prefixes": PREFIXES, "hybrid_selection": HYBRID_RULE,
            "smoke_search_override": {"applied": cfg._sms_smoke_only, "inherited": inherited_search,
                "actual": {k: getattr(cfg.eval, k) for k in inherited_search},
                "note": "Limited smoke checks use rounds1/iters1/pool2/drop2; full experiments retain inherited budgets."},
            "sampling": "Center greedy is candidate zero for every family; no antithetic. "
                        "All candidates are generated by Benchmark_zeroShot.evaluate_stage. sms_tag=0.",
            "grid": "12 theta and 2 sample configurations each generate K256 once per instance. "
                    "Three hybrid configurations report only their native budgets16/64/256, so a complete "
                    "17x5 rectangular table would invent unmeasured hybrid points and is not produced.",
            "timing": "Prefix rows carry the measured full-generation pipeline time, not inferred standalone "
                      "small-K time. Root LP is shared once per instance and charged per pipeline. ILS "
                      "pipelines independently generate exactly64 starts and use the inherited equal search budget.",
            "statistics": "Instance-paired two-sided Wilcoxon, unadjusted; one fixed seed, no training seed inference.",
            "artifacts": "progress.jsonl flushes one record per actual configuration/instance; candidates.jsonl "
                         "stores lossless bit-packed solution vectors/objectives. Identical Chu Theta* and "
                         "preregistered configurations share one actual evaluation, with aliases in tables. "
                         "Explicit --resume-smoke can recover only a limited smoke. No full-run resume or training."}
    if previous is not None:
        meta["resumed_smoke"] = {"previous_status": previous["status"], "previous_source": previous["source"]}
    write_json(dest / "meta.json", meta)
    runner = None
    try:
        data, manifest = load_data(cfg, args.limit, args.smoke_items)
        meta["data"] = manifest
        write_json(dest / "meta.json", meta)
        runner = SMSRunner(cfg, data, device, dest, resume=args.resume_smoke)
        theta_grid = [theta_spec(d, s) for d in ("u", "a", "ua") for s in (.5, 1., 2., 4.)]
        samples = [sample_spec(t) for t in (.125, .333)]
        theta, theta_scores = choose_theta_sample(runner, theta_grid)
        sample, sample_scores = choose_theta_sample(runner, samples)
        hybrids, hybrid_scores = [], []
        sample_rows = runner.results[(N100, runner.key(sample))]
        for kt, ki in ((4, 4), (8, 8), (16, 16)):
            spec = {**theta, "name": f"hyb_{kt}x{ki}", "family": "hybrid", "temperature": sample["temperature"],
                    "stage": f"hyb{kt}x{ki}", "budget": kt*ki, "theta_k": kt, "item_k": ki}
            hybrids.append(spec)
            rows = runner.evaluate(spec, N100)
            delta = _mean_at(rows, kt*ki) - _mean_at(sample_rows, kt*ki)
            hybrid_scores.append({**spec, "mean_gap_pct": _mean_at(rows, kt*ki), "matched_S_delta_pp": delta})
        winner = min(hybrid_scores, key=lambda r: (r["matched_S_delta_pp"], r["budget"]))
        hybrid = next(s for s in hybrids if s["name"] == winner["name"])
        theta_rows = runner.results[(N100, runner.key(theta_spec("ua", 1.)))]
        sample_anchor = next(s for s in samples if s["temperature"] == .125)
        sample_anchor_rows = runner.results[(N100, runner.key(sample_anchor))]
        anchors = {"theta_ua_s1_K32": {"measured": _mean_at(theta_rows, 32), "expected": .321},
                   "sample_T0.125_K16": {"measured": _mean_at(sample_anchor_rows, 16), "expected": .380}}
        for check in anchors.values():
            check["absolute_difference_pp"] = abs(check["measured"] - check["expected"])
        anchors["status"] = "not_assessed_smoke" if cfg._sms_smoke_only else (
            "review_required" if any(v["absolute_difference_pp"] > .1 for v in anchors.values() if isinstance(v, dict)) else "within_expected_tolerance")
        meta["anchors"] = anchors
        write_json(dest / "meta.json", meta)
        selection = {"schema": "t4a_sms_selection_v1", "status": "pending_chu_confirmation",
            "smoke_only": cfg._sms_smoke_only, "selection_dataset": N100,
            "selection_instances": len(data[N100]), "seed": 1234, "source_config": str(source),
            "source_config_sha256": sha256_file(source), "data": manifest,
            "selected_theta": {"dims": theta["dims"], "scale": theta["scale"], "arm": theta["name"]},
            "selected_sample": {"temperature": sample["temperature"], "arm": sample["name"]},
            "selected_hybrid": {**hybrid, "matched_S_delta_pp": winner["matched_S_delta_pp"]},
            "selection_rule": "Theta* and S*: minimum sum of mean gaps at K16 and K64; ties retain grid order.",
            "theta_scores": theta_scores, "sample_scores": sample_scores, "hybrid_scores": hybrid_scores,
            "hybrid_selection": HYBRID_RULE, "anchors": anchors}
        write_json(dest / "selection.json", selection)
        if anchors["status"] == "review_required":
            raise RuntimeError("An expected anchor differs by >0.1 pp; inspect artifacts before any Chu confirmation")
        for spec, label in ((sample, "S_star"), (theta_spec("ua", 1.), "Theta_ua_s1_preregistered"), (theta, "Theta_star")):
            runner.evaluate(spec, CHU, alias=label)
        chu_hybrid = {**hybrid, "stage": f"hyb{256//hybrid['item_k']}x{hybrid['item_k']}",
                      "budget": 256, "theta_k": 256//hybrid["item_k"]}
        runner.evaluate(chu_hybrid, CHU, alias="H_star_extended")
        for spec, label, stage in ((sample, "S_star_64+ILS", "sample64+ILS"),
                                    (theta, "Theta_star_64+ILS", "theta64+ILS")):
            runner.evaluate({**spec, "stage": stage, "budget": 64}, CHU, alias=label, prefixes=(64,))
        summary, pairs, pair_summary = reports(runner.rows, theta["name"], sample["name"], hybrid["name"])
        write_csv(dest / "per_instance.csv", runner.rows)
        write_csv(dest / "summary.csv", summary)
        write_csv(dest / "paired_per_instance.csv", pairs)
        write_csv(dest / "paired_summary.csv", pair_summary)
        meta.update(status="complete", primary_configuration_count=len(theta_grid)+len(samples)+len(hybrids),
                    primary_configurations=[s["name"] for s in theta_grid+samples+hybrids],
                    catalog=runner.catalog, row_count=len(runner.rows))
        selection.update(status="complete", chu_hybrid_actual={"theta_k": chu_hybrid["theta_k"],
                          "item_k": chu_hybrid["item_k"], "total_budget": 256})
        write_json(dest / "selection.json", selection)
    except Exception as exc:
        meta.update(status="failed", error=repr(exc))
        raise
    finally:
        if runner is not None:
            runner.close()
        write_json(dest / "meta.json", meta)
    print(f"T4-a artifacts: {dest}")
    return dest


if __name__ == "__main__":
    main()
