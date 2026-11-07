"""Standalone T6 paired reports. No inference/model modules are imported."""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args(argv)
    dest = args.input.resolve()
    spec = importlib.util.spec_from_file_location("_t6_report_utils", Path(__file__).with_name("t5_report.py"))
    tools = importlib.util.module_from_spec(spec); spec.loader.exec_module(tools)
    meta = json.loads((dest / "meta.json").read_text(encoding="utf-8"))
    rows = tools.read_jsonl(dest / "rows.jsonl")
    crossfit = json.loads((dest / "crossfit.json").read_text(encoding="utf-8"))
    index, groups = {}, defaultdict(list)
    for r in rows:
        k = (r["dataset"], r["arm"], str(r["id"]))
        if k in index or not r["feasible"] or not r["maximal"]:
            raise ValueError(f"Duplicated or invalid result: {k}")
        for field in ("objective", "reference", "gap_pct", "time_total_s", "time_lp_s"):
            if not np.isfinite(r[field]):
                raise ValueError(f"Non-finite {field}: {k}")
        if r["reference"] <= 0 or min(r["time_total_s"],r["time_lp_s"]) < 0 or not np.isclose(
                r["gap_pct"],100*(1-r["objective"]/r["reference"]),rtol=0,atol=1e-9):
            raise ValueError(f"Invalid reference/gap/time: {k}")
        index[k] = r
        for m,n in (("all","all"),(r["M"],r["N"])):
            groups[(r["dataset"],r["arm"],m,n)].append(r)
    summary = []
    for (dataset,arm,m,n),values in groups.items():
        gaps = np.asarray([r["gap_pct"] for r in values])
        summary.append({"dataset":dataset,"arm":arm,"M":m,"N":n,"count":len(values),
            "mean_gap_pct":float(gaps.mean()),"p90_gap_pct":float(np.quantile(gaps,.9)),"worst_gap_pct":float(gaps.max()),
            "mean_time_lp_s":float(np.mean([r["time_lp_s"] for r in values])),
            "mean_time_total_s":float(np.mean([r["time_total_s"] for r in values])),"reused_T5":values[0]["reused_T5"]})
    pairs, pair_groups, recoveries = [], defaultdict(list), []
    for dataset in ("synthetic_N100","chu"):
        arms = {k[1] for k in index if k[0] == dataset}
        expected_arms = {"L","G","D","S1","S2","S3","S4","S5"} | ({"C"} if dataset == "chu" else set())
        ids = {k[2] for k in index if k[:2] == (dataset,"L")}
        if arms != expected_arms or not ids or (not meta["smoke_only"] and len(ids) != (180 if dataset == "chu" else 128)):
            raise ValueError("T6 requires all registered arms and instance counts")
        for arm in arms:
            if {k[2] for k in index if k[:2] == (dataset,arm)} != ids:
                raise ValueError(f"Incomplete paired coverage: {dataset}/{arm}")
            for iid in ids:
                lhs, reference = index[(dataset,arm,iid)], index[(dataset,"L",iid)]
                for field in ("M","N","tightness","reference","reference_kind","reference_proven","data_sha256"):
                    if lhs[field] != reference[field]:
                        raise ValueError(f"Unmatched paired {field}")
                for control in ("L","G"):
                    if arm == control:
                        continue
                    rhs = index[(dataset,control,iid)]
                    pair = {"dataset":dataset,"arm":arm,"control":control,"id":iid,"M":lhs["M"],"N":lhs["N"],
                            "gap_delta_pp":lhs["gap_pct"]-rhs["gap_pct"],"objective_delta":lhs["objective"]-rhs["objective"]}
                    pairs.append(pair)
                    for m,n in (("all","all"),(lhs["M"],lhs["N"])):
                        pair_groups[(dataset,arm,control,m,n)].append(pair["gap_delta_pp"])
        for arm in ("S1","S2","S3","S4","S5"):
            strata = {("all","all")} | {(index[(dataset,"L",iid)]["M"],index[(dataset,"L",iid)]["N"]) for iid in ids}
            for m,n in sorted(strata,key=str):
                selected = [iid for iid in ids if m == "all" or (index[(dataset,"L",iid)]["M"],index[(dataset,"L",iid)]["N"]) == (m,n)]
                g,l,s = [float(np.mean([index[(dataset,a,iid)]["gap_pct"] for iid in selected])) for a in ("G","L",arm)]
                denominator = g-l
                recoveries.append({"dataset":dataset,"arm":arm,"M":m,"N":n,"count":len(selected),
                    "G_gap_pct":g,"L_gap_pct":l,"S_gap_pct":s,"G_minus_L_pp":denominator,
                    "G_minus_S_pp":g-s,"recovery":None if abs(denominator) < 1e-12 else (g-s)/denominator,
                    "denominator_positive":denominator>0,"definition":"ratio of mean gap differences; not mean of per-instance ratios"})
    paired = [{"dataset":d,"arm":a,"control":c,"M":m,"N":n,**tools.paired_statistics(v)}
              for (d,a,c,m,n),v in pair_groups.items()]
    # These contrasts directly answer the clock/grouping/closed-loop questions.
    contrasts = []
    for dataset in ("synthetic_N100","chu"):
        ids = sorted(k[2] for k in index if k[:2] == (dataset,"L"))
        for arm,control in (("S2","S1"),("S3","S2"),("S4","S2"),("S5","S2")):
            delta = [index[(dataset,arm,i)]["gap_pct"]-index[(dataset,control,i)]["gap_pct"] for i in ids]
            contrasts.append({"dataset":dataset,"arm":arm,"control":control,**tools.paired_statistics(delta)})
    for name,data in (("per_instance",rows),("summary",summary),("paired_per_instance",pairs),("paired_summary",paired),
                      ("recovery",recoveries),("contrasts",contrasts),("crossfit_R2",crossfit)):
        tools.write_csv(dest / f"{name}.csv", data)
    def table(values, columns):
        def cell(v):
            return "" if v is None else (f"{v:.6g}" if isinstance(v,float) else str(v))
        return "| "+" | ".join(columns)+" |\n| "+" | ".join("---" for _ in columns)+" |\n"+"\n".join(
            "| "+" | ".join(cell(r.get(c)) for c in columns)+" |" for r in values)
    text = ["# T6 single-trajectory phase schedules","",f"smoke_only: {meta['smoke_only']}",
        "SMOKE: engineering checks only." if meta["smoke_only"] else "One frozen checkpoint pair. Claude/user decides the research gate.",
        "Every S1-S5 result is one greedy episode. L/G/C/D references are reused from T5. No training or multi-start/search.",
        "Synthetic C was not evaluated in T5 and is not invented or rerun. Chu uses best-known references, not certified optima.",
        "", "## Gap and timing", "",table([r for r in summary if r["M"] == "all"],
            ["dataset","arm","count","mean_gap_pct","p90_gap_pct","worst_gap_pct","mean_time_lp_s","mean_time_total_s"]),
        "", "## Paired differences (percentage points, arm minus control)", "",
        table([r for r in paired if r["M"] == "all" and r["arm"].startswith("S")],
            ["dataset","arm","control","mean_gap_delta_pp","wins","ties","losses","wilcoxon_two_sided_p"]),
        "Two-sided Wilcoxon p values are unadjusted. Reused timings keep the T5 environment; new timings include interpolation and assertions.",
        "", "## Recovery", "", table([r for r in recoveries if r["M"] == "all"],
            ["dataset","arm","G_minus_L_pp","G_minus_S_pp","recovery","denominator_positive"]),
        "Recovery is not clipped to [0,1]; near-zero denominators are blank, and nonpositive denominators are flagged. M/N strata: recovery.csv.",
        "", "## Clock, grouping, reduced dictionary and live price contrasts", "",
        table(contrasts,["dataset","arm","control","mean_gap_delta_pp","wilcoxon_two_sided_p"]),
        "If the formal S2-S1 absolute mean difference is <.02 pp, S2 can be the main presentation arm; no automatic research gate is applied.",
        "", "## Synthetic-only two-fold diagnostics", "",
        table([r for r in crossfit if r["grouping"] == "category" and (r["field"] in ("profit_exponent","lp_primal_weight","companion_sum","scarcity_exponent"))],
            ["grouping","clock","field","instances","raw_R2","within_trajectory_R2"]),
        "Full fields and M-only grouping: crossfit_R2.csv. Complete instances alternate within each category; quantiles/knots fit only the opposite fold.",
        "Within-trajectory R2 centers both observed and predicted series by their own held-out trajectory means. This is scoring only, not a deployed clock.",
        "Ten quantile bins may yield fewer distinct knots after empty/tied bins. Interactions are constant; schedule prices are uniform except live-L S5.", ""]
    (dest/"report.md").write_text("\n".join(text),encoding="utf-8")
    def digest(path):
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b""):
                h.update(chunk)
        return h.hexdigest()
    manifest = {"schema":"t6_report_v1","smoke_only":meta["smoke_only"],"source_sha256":digest(Path(__file__)),
        "utility_source_sha256":digest(Path(__file__).with_name("t5_report.py")),
        "input_sha256":{n:digest(dest/n) for n in ("rows.jsonl","crossfit.json")},
        "fit_artifact_sha256":{n:digest(dest/n) for n in ("traces.jsonl","schedules.json","fitting_progress.jsonl")},
        "output_sha256":{n:digest(dest/n) for n in ("per_instance.csv","summary.csv","paired_per_instance.csv","paired_summary.csv",
                                                "recovery.csv","contrasts.csv","crossfit_R2.csv","report.md")}}
    (dest/"report_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(f"T6 reports: {len(rows)} rows, {len(pairs)} pairs, {len(crossfit)} cross-fit diagnostics")


if __name__ == "__main__":
    main()
