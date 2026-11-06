"""Budget-matched reports; combine seeds with --inputs per_instance.csv ... ."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ARMS = ("fixed", "fixed_dualgreedy", "untrained", "learned_static", "learned_dynamic", "learned_mu_only")
BUDGET_FIELDS = ("stage", "budget", "search_starts", "ils_rounds", "ls_max_iters", "ls_drop_k",
                 "ils_perturb_max", "sample_temperature", "evaluation_seed", "data_sha256", "protocol_sha256")
REPORT_FIELDS = tuple(k for k in BUDGET_FIELDS if k != "evaluation_seed")


def _comparison_budget(row):
    # Old reports stored only one (executor-level) SHA; their existing comparisons
    # remain valid. New reports carry a full method SHA plus an explicitly named
    # comparison signature allowing the specified static/dynamic schedule ablation.
    return tuple(str(row.get("comparison_protocol_sha256", row["protocol_sha256"]))
                 if key == "protocol_sha256" else str(row[key]) for key in BUDGET_FIELDS)


def paired_seed(seed, instance_id, stage, trial=0):
    raw = f"{seed}|{instance_id}|{stage}|{trial}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "little")


def parse_stages(names, sample_k=256):
    specs = []
    for name in names:
        name = name.strip().replace("sampleK", f"sample{sample_k}")
        sms = re.fullmatch(r"(theta([1-9][0-9]*)|hyb([1-9][0-9]*)x([1-9][0-9]*))(\+ILS)?", name)
        if sms:
            theta_k, item_k = (int(sms[2]), 1) if sms[2] else (int(sms[3]), int(sms[4]))
            specs.append({"name": name, "decode": "greedy" if sms[2] else "sample",
                          "budget": theta_k * item_k, "refine": bool(sms[5]),
                          "theta_k": theta_k, "item_k": item_k})
            continue
        match = re.fullmatch(r"(greedy|sample([1-9][0-9]*))(\+ILS)?", name)
        if not match:
            raise ValueError(f"Invalid stage {name!r}; use greedy, sampleK, thetaK or hybAxB, optionally +ILS")
        specs.append({"name": name, "decode": "greedy" if match[1] == "greedy" else "sample",
                      "budget": int(match[2]) if match[2] else 1, "refine": bool(match[3])})
    if not specs or len({s["name"] for s in specs}) != len(specs):
        raise ValueError("Stages must be nonempty and unique")
    return specs


def jsonable(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value.item() if isinstance(value, np.generic) else value


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def _strata(row):
    m, n, a = row["M"], row["N"], row["alpha"]
    return (("all", "all", "all"), (m, "all", "all"), ("all", n, "all"),
            ("all", "all", a), (m, n, "all"), (m, n, a))


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        for stratum in _strata(row):
            groups[(row["arm"], str(row["seed"]), *(str(row[k]) for k in BUDGET_FIELDS), *stratum)].append(row)
    output = []
    for key, values in groups.items():
        gap = np.asarray([float(r["gap_pct"]) for r in values])
        item = {"arm": key[0], "seed": key[1], **dict(zip(BUDGET_FIELDS, key[2:2 + len(BUDGET_FIELDS)])),
                "comparison_protocol_sha256": values[0].get("comparison_protocol_sha256", values[0]["protocol_sha256"]),
                "theta_update": values[0].get("theta_update", "unspecified"),
                "theta_interval": values[0].get("theta_interval", 1),
                "M": key[-3], "N": key[-2], "alpha": key[-1], "count": len(values),
                "mean_gap_pct": float(gap.mean()), "worst_gap_pct": float(gap.max()),
                "p90_gap_pct": float(np.quantile(gap, .9))}
        for field in ("time_lp_s", "time_construct_s", "time_refine_s", "time_total_s"):
            item["mean_" + field] = float(np.mean([float(r[field]) for r in values]))
        for field in ("theta_decisions", "decision_u_abs_mean", "decision_alpha_mean",
                      "decision_alpha_std", "decision_gamma_mean", "decision_eta_mean",
                      "time_root_lp_s", "time_residual_lp_s"):
            present = [float(r[field]) for r in values if r.get(field) not in (None, "")]
            if present:
                item["mean_" + field] = float(np.mean(present))
        output.append(item)
    return output


def _diff_stats(diffs):
    d = np.asarray(diffs)
    p = 1.0
    if any(abs(v) > 1e-12 for v in diffs):
        from scipy.stats import wilcoxon
        p = float(wilcoxon(diffs, zero_method="wilcox", alternative="two-sided").pvalue)
    return {"n_pairs": len(diffs), "mean_gap_delta_pp": float(d.mean()),
            "median_gap_delta_pp": float(np.median(d)), "wins": int((d < -1e-9).sum()),
            "ties": int((np.abs(d) <= 1e-9).sum()), "losses": int((d > 1e-9).sum()),
            "wilcoxon_two_sided_p": p, "p_values_adjusted": False}


def paired_report(rows):
    """Pair same instances/seeds/budgets; average seeds before cross-seed instance tests."""
    lookup, cohorts = {}, {}
    for row in rows:
        comparison_sha = row.get("comparison_protocol_sha256", row["protocol_sha256"])
        cohort = (str(row["seed"]), str(row["id"]), str(row["stage"]), row["data_sha256"], comparison_sha)
        budget = _comparison_budget(row)
        if cohort in cohorts and cohorts[cohort] != budget:
            raise ValueError(f"Mismatched inference budgets/evaluation seeds for seed={row['seed']}, "
                             f"id={row['id']}, stage={row['stage']}")
        cohorts[cohort] = budget
        key = (str(row["seed"]), str(row["id"]), *budget)
        bucket = lookup.setdefault(key, {})
        if row["arm"] in bucket:
            raise ValueError(f"Duplicate result: arm={row['arm']}, seed={row['seed']}, id={row['id']}, stage={row['stage']}")
        bucket[row["arm"]] = row
    pairs, groups = [], defaultdict(list)
    for key, arms in lookup.items():
        for left, right in combinations(sorted(arms), 2):
            if right == "learned_dynamic":
                left, right = right, left
            lhs, rhs = arms[left], arms[right]
            pair = {"id": lhs["id"], "seed": lhs["seed"], "M": lhs["M"], "N": lhs["N"],
                    "alpha": lhs["alpha"], "arm": left, "control": right,
                    **{k: lhs[k] for k in BUDGET_FIELDS},
                    "comparison_protocol_sha256": lhs.get("comparison_protocol_sha256", lhs["protocol_sha256"]),
                    "control_protocol_sha256": rhs["protocol_sha256"],
                    "theta_update": lhs.get("theta_update", "unspecified"),
                    "theta_interval": lhs.get("theta_interval", 1),
                    "control_theta_update": rhs.get("theta_update", "unspecified"),
                    "control_theta_interval": rhs.get("theta_interval", 1),
                    "matched_update_schedule": (lhs.get("theta_update", "unspecified"), str(lhs.get("theta_interval", 1))) ==
                                               (rhs.get("theta_update", "unspecified"), str(rhs.get("theta_interval", 1))),
                    "gap_delta_pp": float(lhs["gap_pct"]) - float(rhs["gap_pct"]),
                    "objective_delta": float(lhs["objective"]) - float(rhs["objective"]),
                    "time_delta_s": float(lhs["time_total_s"]) - float(rhs["time_total_s"]),
                    "matched_executor": "fixed_dualgreedy" not in (left, right)}
            pairs.append(pair)
            for stratum in _strata(pair):
                fields = dict(zip(BUDGET_FIELDS, _comparison_budget(lhs)))
                groups[(left, right, *(fields[k] for k in REPORT_FIELDS), *stratum)].append(pair)
    output = []
    for key, values in groups.items():
        base = {"arm": key[0], "control": key[1], **dict(zip(REPORT_FIELDS, key[2:2 + len(REPORT_FIELDS)])),
                "M": key[-3], "N": key[-2], "alpha": key[-1],
                "matched_executor": "fixed_dualgreedy" not in key[:2]}
        # The grouping key uses comparison SHA. Keep both full method identities
        # alongside it so differing theta intervals never disappear from reports.
        base["comparison_protocol_sha256"] = base.pop("protocol_sha256")
        for field in ("protocol_sha256", "control_protocol_sha256", "theta_update", "theta_interval",
                      "control_theta_update", "control_theta_interval", "matched_update_schedule"):
            variants = sorted({str(v[field]) for v in values})
            base[field] = variants[0] if len(variants) == 1 else json.dumps(variants)
        by_seed, by_instance = defaultdict(list), defaultdict(dict)
        for pair in values:
            by_seed[str(pair["seed"])].append(float(pair["gap_delta_pp"]))
            by_instance[str(pair["id"])][str(pair["seed"])] = float(pair["gap_delta_pp"])
        for seed, diffs in by_seed.items():
            output.append({**base, "seed": seed, "training_seeds": 1,
                           "evaluation_seed": next(p["evaluation_seed"] for p in values if str(p["seed"]) == seed),
                           "unit": "instance_within_training_seed", **_diff_stats(diffs)})
        if len(by_seed) > 1:
            seeds = set(by_seed)
            diffs = [float(np.mean(list(v.values()))) for v in by_instance.values() if set(v) == seeds]
            if diffs:
                output.append({**base, "seed": "mean_over_seeds", "training_seeds": len(seeds),
                               "evaluation_seed": ",".join(sorted({str(p["evaluation_seed"]) for p in values})),
                               "unit": "instance_mean_over_matched_training_seeds", **_diff_stats(diffs),
                               "meets_three_seed_requirement": len(seeds) >= 3})
    return pairs, output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", default="results/paired_report")
    args = parser.parse_args(argv)
    rows = []
    for path in args.inputs:
        with open(path, encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise ValueError("No input rows")
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=True)
    pairs, summary = paired_report(rows)
    write_csv(dest / "paired_per_instance.csv", pairs)
    write_csv(dest / "paired_summary.csv", summary)
    write_csv(dest / "summary.csv", summarize(rows))
    (dest / "meta.json").write_text(json.dumps({"inputs": args.inputs,
        "training_seeds": sorted({int(r["seed"]) for r in rows}),
        "note": "single seed" if len({r["seed"] for r in rows}) == 1 else "multiple training seeds",
        "gap_units": "percentage points; arm minus control, negative is better",
        "seed_aggregation": "instance differences averaged across all matched training seeds",
        "p_values": "two-sided Wilcoxon, unadjusted; report comparison and seed counts"}, indent=2), encoding="utf-8")
    print(dest)


if __name__ == "__main__":
    main()
