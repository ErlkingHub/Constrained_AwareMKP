"""Report T5 artifacts in a separate process, without importing either model.

The legacy inference process only uses its own modules, numpy and torch. This
postprocessor owns scipy statistics so those dependencies never enter inference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def paired_statistics(deltas):
    d = np.asarray(deltas, dtype=np.float64)
    if not len(d) or not np.isfinite(d).all():
        raise ValueError("Paired differences must be nonempty and finite")
    return {
        "count": len(d), "mean_gap_delta_pp": float(d.mean()),
        "median_gap_delta_pp": float(np.median(d)),
        "wins": int((d < -1e-9).sum()), "ties": int((np.abs(d) <= 1e-9).sum()),
        "losses": int((d > 1e-9).sum()),
        "wilcoxon_two_sided_p": (1.0 if np.all(d == 0) else
                                  float(wilcoxon(d, zero_method="wilcox").pvalue)),
        "p_values_adjusted": False,
    }


def outcome_tables(rows):
    index, groups = {}, defaultdict(list)
    for row in rows:
        key = (row["dataset"], row["arm"], str(row["id"]))
        if key in index:
            raise ValueError(f"Duplicate evaluated instance: {key}")
        if not row["feasible"] or not row["maximal"]:
            raise ValueError(f"Invalid construction: {key}")
        for field in ("gap_pct", "objective", "reference", "time_lp_s", "time_total_s"):
            if not np.isfinite(float(row[field])):
                raise ValueError(f"Non-finite {field}: {key}")
        if float(row["reference"]) <= 0:
            raise ValueError(f"Nonpositive reference: {key}")
        if float(row["time_lp_s"]) < 0 or float(row["time_total_s"]) < 0:
            raise ValueError(f"Negative measured time: {key}")
        computed = 100 * (1 - float(row["objective"]) / float(row["reference"]))
        if not np.isclose(float(row["gap_pct"]), computed, rtol=0, atol=1e-9):
            raise ValueError(f"Gap/reference mismatch: {key}")
        index[key] = row
        strata = [("all", "all")]
        if row["dataset"] == "chu":
            strata.append((row["M"], row["N"]))
        for m, n in strata:
            groups[(row["dataset"], row["arm"], m, n)].append(row)
    summary = []
    for (dataset, arm, m, n), values in groups.items():
        gaps = np.asarray([v["gap_pct"] for v in values])
        summary.append({"dataset": dataset, "arm": arm, "M": m, "N": n,
            "count": len(values), "mean_gap_pct": float(gaps.mean()),
            "p90_gap_pct": float(np.quantile(gaps, .9)), "worst_gap_pct": float(gaps.max()),
            "mean_time_lp_s": float(np.mean([v["time_lp_s"] for v in values])),
            "mean_time_total_s": float(np.mean([v["time_total_s"] for v in values])),
            "mean_time_with_required_prepass_s": float(np.mean([v["time_with_required_prepass_s"] for v in values]))})
    pairs, pair_groups = [], defaultdict(list)
    datasets = dict.fromkeys(r["dataset"] for r in rows)
    for dataset in datasets:
        control = "L" if dataset == "chu" else "G"
        arms = dict.fromkeys(r["arm"] for r in rows if r["dataset"] == dataset)
        control_ids = {key[2] for key in index if key[:2] == (dataset, control)}
        if not control_ids:
            raise ValueError(f"No control {control} for {dataset}")
        for arm in arms:
            if arm == control:
                continue
            arm_ids = {key[2] for key in index if key[:2] == (dataset, arm)}
            if arm_ids != control_ids:
                raise ValueError(f"Missing/extra paired instances for {dataset}/{arm}")
            for iid in sorted(control_ids):
                lhs, rhs = index[(dataset, arm, iid)], index[(dataset, control, iid)]
                for field in ("M", "N", "tightness", "reference", "reference_kind"):
                    if lhs[field] != rhs[field]:
                        raise ValueError(f"Paired {field} differs for {dataset}/{iid}")
                pair = {"dataset": dataset, "arm": arm, "control": control, "id": iid,
                    "M": lhs["M"], "N": lhs["N"], "tightness": lhs["tightness"],
                    "gap_delta_pp": lhs["gap_pct"] - rhs["gap_pct"],
                    "objective_delta": lhs["objective"] - rhs["objective"],
                    "time_lp_delta_s": lhs["time_lp_s"] - rhs["time_lp_s"],
                    "time_total_delta_s": lhs["time_total_s"] - rhs["time_total_s"]}
                pairs.append(pair)
                strata = [("all", "all")]
                if dataset == "chu":
                    strata.append((lhs["M"], lhs["N"]))
                for m, n in strata:
                    pair_groups[(dataset, arm, control, m, n)].append(pair["gap_delta_pp"])
    paired_summary = [{"dataset": d, "arm": a, "control": c, "M": m, "N": n,
                       **paired_statistics(values)}
                      for (d, a, c, m, n), values in pair_groups.items()]
    return summary, pairs, paired_summary


SCALAR_FIELDS = ("profit_exponent", "scarcity_exponent", "bottleneck_weight", "temperature",
                 "opportunity_weight", "reserve_weight", "lp_primal_weight", "lp_dual_weight")


def theta_statistics(theta):
    scalars = {}
    for field in SCALAR_FIELDS:
        value = np.asarray(theta[field], dtype=np.float64)
        if value.size != 1:
            raise ValueError(f"Expected one scalar per decision: {field}")
        scalars[field] = float(value.reshape(-1)[0])
    prices = np.asarray(theta["resource_prices"], dtype=np.float64).reshape(-1)
    interactions = np.asarray(theta["resource_interactions"], dtype=np.float64)
    if interactions.size != len(prices)**2:
        raise ValueError("Invalid interaction matrix in trace")
    interactions = interactions.reshape(len(prices), len(prices))
    scalars["price_std"] = float(prices.std(ddof=0))
    scalars["interactions_trace"] = float(np.trace(interactions))
    scalars["companion_sum"] = float(np.asarray(theta["companion_weights"], dtype=np.float64).sum())
    if not all(np.isfinite(v) for v in scalars.values()):
        raise ValueError("Non-finite theta trace")
    return scalars


def variance_table(traces):
    """Nested population sums of squares, weighting each actual decision equally."""
    fields, seen, steps = defaultdict(list), set(), defaultdict(list)
    for row in traces:
        if row["dataset"] != "chu" or row.get("arm", "L") != "L":
            continue
        iid = str(row["id"])
        key = (iid, row["step"])
        if key in seen:
            raise ValueError(f"Duplicate Chu decision: {key}")
        seen.add(key)
        category = (row["M"], row["N"], row["tightness"])
        steps[iid].append(row["step"])
        for field, value in theta_statistics(row["theta"]).items():
            fields[field].append((category, iid, value))
    if not fields:
        raise ValueError("No Chu L traces available for variance decomposition")
    for iid, values in steps.items():
        if sorted(values) != list(range(len(values))):
            raise ValueError(f"Noncontiguous decision trace: {iid}")
    result = []
    for field, observations in fields.items():
        categories, instances = defaultdict(list), defaultdict(list)
        for category, iid, value in observations:
            categories[category].append(value)
            instances[(category, iid)].append(value)
        grand = float(np.mean([v for _, _, v in observations]))
        total = float(sum((v - grand)**2 for _, _, v in observations))
        between_categories = sum(len(v) * (np.mean(v) - grand)**2 for v in categories.values())
        between_instances = sum(len(v) * (np.mean(v) - np.mean(categories[cat]))**2
                                for (cat, _), v in instances.items())
        within = sum(sum((x - np.mean(v))**2 for x in v) for v in instances.values())
        if not np.isclose(total, between_categories + between_instances + within, rtol=1e-8, atol=1e-12):
            raise ValueError(f"Variance decomposition failed: {field}")
        # A constant field has no defined share of nonzero variance; use blanks.
        fractions = (["", "", ""] if total == 0 else
                     [float(x / total) for x in (between_categories, between_instances, within)])
        result.append({"field": field, "decision_count": len(observations),
            "instance_count": len(instances), "category_count": len(categories),
            "mean": grand, "total_population_variance": total / len(observations),
            "between_category_fraction": fractions[0], "between_instance_fraction": fractions[1],
            "within_trajectory_fraction": fractions[2], "constant": total == 0,
            "weighting": "equal actual decision observations; nested population sums of squares"})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args(argv)
    dest = args.input.resolve()
    meta = json.loads((dest / "meta.json").read_text(encoding="utf-8"))
    # During a successful runner's finalization the status can still be running.
    # Require the full registered arm/instance coverage before calling it a report.
    rows, traces = read_jsonl(dest / "rows.jsonl"), read_jsonl(dest / "traces.jsonl")
    chu_arms = {r["arm"] for r in rows if r["dataset"] == "chu"}
    synthetic_arms = {r["arm"] for r in rows if r["dataset"] == "synthetic_N100"}
    if chu_arms != {"L", "D", "G", "C", "I"} or len(synthetic_arms) != 15 or not {"L", "D", "G", "G-min", "G-F0"} <= synthetic_arms:
        raise ValueError("T5 reporting requires all five Chu and fifteen synthetic arms")
    if not meta["smoke_only"]:
        for dataset, expected in (("chu", 180), ("synthetic_N100", 128)):
            actual = {str(r["id"]) for r in rows if r["dataset"] == dataset}
            if len(actual) != expected:
                raise ValueError(f"Formal {dataset} requires {expected} distinct instances")
    chu_ids = {str(r["id"]) for r in rows if r["dataset"] == "chu" and r["arm"] == "L"}
    trace_ids = {str(r["id"]) for r in traces if r["dataset"] == "chu" and r.get("arm", "L") == "L"}
    if chu_ids != trace_ids:
        raise ValueError("Chu L traces do not cover exactly the evaluated L instances")
    learned = {str(r["id"]): r for r in rows if r["dataset"] == "chu" and r["arm"] == "L"}
    for row in rows:
        prepass = learned[str(row["id"])]["time_total_s"] if row["dataset"] == "chu" and row["arm"] == "I" else 0.0
        row["required_L_prepass_s"] = prepass
        row["time_with_required_prepass_s"] = row["time_total_s"] + prepass
    summary, pairs, paired_summary = outcome_tables(rows)
    variance = variance_table(traces)
    write_csv(dest / "per_instance.csv", rows)
    write_csv(dest / "summary.csv", summary)
    write_csv(dest / "paired_per_instance.csv", pairs)
    write_csv(dest / "paired_summary.csv", paired_summary)
    write_csv(dest / "theta_variance.csv", variance)
    reproduction = json.loads((dest / "reproduction.json").read_text(encoding="utf-8"))
    def table(values, columns):
        header = "| " + " | ".join(columns) + " |\n| " + " | ".join("---" for _ in columns) + " |\n"
        def cell(v):
            return f"{v:.6g}" if isinstance(v, float) else str(v)
        return header + "\n".join("| " + " | ".join(cell(r.get(c, "")) for c in columns) + " |" for r in values)
    text = ["# T5 v6 attribution", "",
            f"smoke_only: {meta['smoke_only']}. Only synthetic_N100 selects ablations/G-min; Chu confirms.",
            ("SMOKE ONLY: engineering checks, not evidence for the research gate." if meta["smoke_only"] else
             "Single frozen checkpoint pair; no training-seed generalization is established."),
            "Chu gaps use best-known values, not certified optima. Two-sided Wilcoxon p values are unadjusted.",
            "G-F0 follows the requested neutralization; it does not reproduce the new project's root-LP price formula.",
            "", "## L reproduction", "", "```json", json.dumps(reproduction, ensure_ascii=False, indent=2), "```",
            "", "## Aggregate outcomes", "",
            table([r for r in summary if r['M'] == 'all'],
                  ["dataset", "arm", "count", "mean_gap_pct", "p90_gap_pct", "worst_gap_pct", "mean_time_lp_s", "mean_time_total_s", "mean_time_with_required_prepass_s"]),
            "", "## Paired differences (arm minus control, percentage points)", "",
            table([r for r in paired_summary if r['M'] == 'all'],
                  ["dataset", "arm", "control", "mean_gap_delta_pp", "wins", "ties", "losses", "wilcoxon_two_sided_p"]),
            "", "## Chu L variance decomposition", "",
            "Equal weight per actual decision. Constant fields have blank variance shares. Matrix trace means diagonal sum.", "",
            table(variance, ["field", "total_population_variance", "between_category_fraction", "between_instance_fraction", "within_trajectory_fraction"]),
            "", "M/N strata and per-instance LP/total timings are in summary.csv, paired_summary.csv and per_instance.csv.",
            "I is a post-hoc same-instance L-trajectory average, a diagnostic rather than a deployable static policy. "
            "time_total_s measures its injected rollout; time_with_required_prepass_s also charges its required L rollout.",
            "No training is performed; the research gate is decided by Claude/user.", ""]
    (dest / "report.md").write_text("\n".join(text), encoding="utf-8")
    def digest(path):
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    # Reporting runs later in a separate process: fingerprint its actual source,
    # rather than assuming it still matches the inference-start source snapshot.
    report_manifest = {"schema": "t5_report_v1", "status": "complete", "smoke_only": meta["smoke_only"],
        "report_source_sha256": digest(Path(__file__).resolve()),
        "input_sha256": {name: digest(dest / name) for name in ("rows.jsonl", "traces.jsonl", "reproduction.json")},
        "output_sha256": {name: digest(dest / name) for name in ("per_instance.csv", "summary.csv", "paired_per_instance.csv",
                             "paired_summary.csv", "theta_variance.csv", "report.md")}}
    (dest / "report_manifest.json").write_text(json.dumps(report_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"T5 reports: {len(rows)} outcomes, {len(pairs)} pairs, {len(variance)} theta statistics")


if __name__ == "__main__":
    main()
