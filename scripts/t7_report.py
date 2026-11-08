"""T7 policy and sampled-interface reports, in a separate statistics process."""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--input",required=True,type=Path)
    dest=parser.parse_args(argv).input.resolve()
    spec=importlib.util.spec_from_file_location("_t7_stats",Path(__file__).with_name("t5_report.py"))
    tools=importlib.util.module_from_spec(spec);spec.loader.exec_module(tools)
    meta=json.loads((dest/"meta.json").read_text(encoding="utf-8"))
    rows=tools.read_jsonl(dest/"rows.jsonl");index={}
    for r in rows:
        if r.get("reused_reference") and r["arm"] in ("L","G","S2"):
            r.setdefault("controller_calls",r["steps"] if r["arm"]=="L" else 0)
            r["controller_count_source"]="inferred from original per-step L / fixed G / uniform-price S2 contract"
        k=(r["dataset"],r["arm"],str(r["id"]))
        if k in index or not r["feasible"] or not r["maximal"]:raise ValueError("Invalid or duplicate policy row")
        if not np.isfinite([r["gap_pct"],r["objective"],r["time_total_s"],r["time_lp_s"]]).all():raise ValueError("Nonfinite row")
        if not np.isclose(r["gap_pct"],100*(1-r["objective"]/r["reference"]),rtol=0,atol=1e-9):raise ValueError("Invalid gap")
        index[k]=r
    summary,pairs,paired,recovery,additivity=[],[],[],[],[]
    if "policy" in meta["stages"]:
        arms=("L","G","S2","SL05","SL08","SL09","LS08","K2","K8")
        for dataset in ("synthetic_N100","chu"):
            ids=sorted(k[2] for k in index if k[:2]==(dataset,"L"))
            if not meta["smoke_only"] and len(ids)!=(128 if dataset=="synthetic_N100" else 180):raise ValueError("Incomplete policy instances")
            for arm in arms:
                if {k[2] for k in index if k[:2]==(dataset,arm)}!=set(ids):raise ValueError("Incomplete policy arm")
                values=[index[(dataset,arm,i)] for i in ids]
                for r in values:
                    control=index[(dataset,"L",str(r["id"]))]
                    for f in ("M","N","reference","reference_kind","data_sha256"):
                        if r[f]!=control[f]:raise ValueError("Unmatched policy reference")
                for m,n in [("all","all"),*sorted({(r["M"],r["N"]) for r in values})]:
                    selected=[r for r in values if m=="all" or (r["M"],r["N"])==(m,n)]
                    gap=np.asarray([r["gap_pct"] for r in selected]);switched=[r for r in selected if r.get("switch_step") is not None]
                    item={"dataset":dataset,"arm":arm,"M":m,"N":n,"count":len(selected),"mean_gap_pct":float(gap.mean()),
                        "p90_gap_pct":float(np.quantile(gap,.9)),"worst_gap_pct":float(gap.max()),
                        "mean_time_total_s":float(np.mean([r["time_total_s"] for r in selected])),
                        "mean_time_lp_s":float(np.mean([r["time_lp_s"] for r in selected])),"switch_count":len(switched),
                        "switch_fraction":len(switched)/len(selected)}
                    for field in ("controller_calls","controller_inclusive_lp_s","controller_excluding_lp_s"):
                        item["mean_"+field]=float(np.mean([r[field] for r in selected])) if all(field in r for r in selected) else None
                    for field in ("switch_step","switch_feasible","switch_remaining_steps","switch_remaining_after_action"):
                        item["mean_"+field+"_when_switched"]=float(np.mean([r[field] for r in switched])) if switched else None
                    summary.append(item)
                    g,l=[float(np.mean([index[(dataset,a,str(r["id"]))]["gap_pct"] for r in selected])) for a in ("G","L")]
                    if arm not in ("L","G"):
                        recovery.append({"dataset":dataset,"arm":arm,"M":m,"N":n,"count":len(selected),"G_minus_L_pp":g-l,
                            "G_minus_arm_pp":g-float(gap.mean()),"recovery":None if abs(g-l)<1e-12 else (g-float(gap.mean()))/(g-l),
                            "denominator_positive":g>l})
                    for control in ("L","G"):
                        if arm==control:continue
                        delta=[r["gap_pct"]-index[(dataset,control,str(r["id"]))]["gap_pct"] for r in selected]
                        paired.append({"dataset":dataset,"arm":arm,"control":control,"M":m,"N":n,**tools.paired_statistics(delta)})
                for control in ("L","G"):
                    if arm==control:continue
                    for r in values:
                        pairs.append({"dataset":dataset,"arm":arm,"control":control,"id":r["id"],"M":r["M"],"N":r["N"],
                            "gap_delta_pp":r["gap_pct"]-index[(dataset,control,str(r["id"]))]["gap_pct"]})
            rates={r["arm"]:r["recovery"] for r in recovery if r["dataset"]==dataset and r["M"]=="all"}
            total=None if rates["SL08"] is None or rates["LS08"] is None else rates["SL08"]+rates["LS08"]
            additivity.append({"dataset":dataset,"SL08_recovery":rates["SL08"],"LS08_recovery":rates["LS08"],
                               "sum":total,"sum_minus_one":None if total is None else total-1})
    for name,data in (("per_instance",rows),("summary",summary),("paired_per_instance",pairs),("paired_summary",paired),
                      ("recovery",recovery),("additivity",additivity)):
        tools.write_csv(dest/f"{name}.csv",data)
    states=tools.read_jsonl(dest/"states.jsonl") if "capacity" in meta["stages"] else []
    if "capacity" in meta["stages"]:
        replays=tools.read_jsonl(dest/"replay.jsonl")
        if len({(r["dataset"],r["id"]) for r in replays})!=len(replays):raise ValueError("Duplicate L replay")
        state_keys={(r["dataset"],r["id"],r["step"],r["executor"]) for r in states}
        if len(state_keys)!=len(states):raise ValueError("Duplicate state/executor row")
        expected_keys={(r["dataset"],r["id"],step,e) for r in replays for step in range(r["steps"]) for e in ("E_full","E_min")}
        if state_keys!=expected_keys or any(not r["sequence_matches_L"] for r in replays):raise ValueError("Incomplete L replay states")
        if not meta["smoke_only"] and (sum(r["dataset"]=="synthetic_N100" for r in replays)!=128 or sum(r["dataset"]=="chu" for r in replays)!=36):
            raise ValueError("Incomplete formal capacity source coverage")
    groups=defaultdict(list);regions=[];target_groups=defaultdict(list)
    for dataset in ("synthetic_N100","chu"):
        data=[r for r in states if r["dataset"]==dataset]
        if not data:continue
        unique={(r["id"],r["step"]):r["tau_cap"] for r in data}
        edges=np.quantile(list(unique.values()),np.linspace(0,1,11))
        for r in data:
            if (not 0<=r["rho_local"]<=1 or not 0<=r["rho_full"]<=1 or not 0<=r["stability"]<=1 or
                not 1<=r["reachable_full"]<=r["feasible_count"] or not 1<=r["reachable_local"]<=r["feasible_count"] or
                r["global_draws"]!=meta["global_draws"] or r["local_draws"]!=meta["local_draws"]):raise ValueError("Invalid capacity row/budget")
            decile=int(np.clip(np.searchsorted(edges,r["tau_cap"],side="right")-1,0,9))
            for subset in ("all","L_S2_different"):
                if subset!="all" and not r["L_S2_different"]:continue
                for axis,value in (("all","all"),("tau_cap_decile",decile),("N",r["N"])):
                    groups[(dataset,r["executor"],subset,axis,value)].append(r)
            if "L_in_target" in r:target_groups[(r["executor"],r["reference_proven"])].append(r)
        for executor in ("E_full","E_min"):
            members=[r for r in data if r["executor"]==executor];different=sum(r["L_S2_different"] for r in members)
            for label,predicate in (("rho<0.2",lambda v:v<.2),("0.2<=rho<=0.5",lambda v:.2<=v<=.5),("rho>0.5",lambda v:v>.5)):
                selected=[r for r in members if predicate(r["rho_local"])];d=sum(r["L_S2_different"] for r in selected)
                regions.append({"dataset":dataset,"executor":executor,"region":label,"states":len(selected),"different_steps":d,
                    "state_share":len(selected)/len(members),"different_fraction_in_region":None if not selected else d/len(selected),
                    "share_of_all_differences":None if not different else d/different})
    capacity=[]
    for (dataset,executor,subset,axis,value),members in groups.items():
        row={"dataset":dataset,"executor":executor,"subset":subset,"axis":axis,"value":value,"states":len(members)}
        for field in ("feasible_count","reachable_full","reachable_local","rho_full","rho_local","stability","stability_projected_L"):
            values=np.asarray([r[field] for r in members],dtype=float)
            row.update({field+"_mean":float(values.mean()),field+"_p10":float(np.quantile(values,.1)),
                        field+"_median":float(np.median(values)),field+"_p90":float(np.quantile(values,.9))})
        capacity.append(row)
    targets=[{"executor":e,"reference_proven":p,"states":len(v),"L_in_target_fraction":float(np.mean([r["L_in_target"] for r in v])),
              "sampled_empty_fraction":float(np.mean([r["sampled_target_intersection_empty"] for r in v]))} for (e,p),v in target_groups.items()]
    oracle=tools.read_jsonl(dest/"oracle.jsonl") if "oracle" in meta["stages"] else []
    oracle_groups=defaultdict(list)
    if "oracle" in meta["stages"]:
        count=meta["target_counts"]["proven_optimum"]+(meta["target_counts"]["MILP_incumbent"] if meta["oracle_targets"]=="all" else 0)
        keys={(r["id"],r["executor"]) for r in oracle}
        if len(keys)!=len(oracle) or len(oracle)!=2*count:raise ValueError("Incomplete oracle episode coverage")
    for r in oracle:
        if not r["feasible"] or not r["maximal"]:raise ValueError("Invalid oracle")
        if r["reference_proven"] and r["gap_pct"] < -1e-7:raise ValueError("Oracle exceeds a certified optimal reference")
        oracle_groups[(r["executor"],r["target_kind"])].append(r)
    oracle_summary=[]
    for (executor,kind),members in oracle_groups.items():
        empty=[r for r in members if r["first_empty"] is not None]
        oracle_summary.append({"executor":executor,"target_kind":kind,"count":len(members),
            "mean_gap_pct":float(np.mean([r["gap_pct"] for r in members])),"p90_gap_pct":float(np.quantile([r["gap_pct"] for r in members],.9)),
            "worst_gap_pct":float(max(r["gap_pct"] for r in members)),"first_empty_count":len(empty),
            "mean_first_empty_step":None if not empty else float(np.mean([r["first_empty"]["step"] for r in empty])),
            "mean_first_empty_tau_cap":None if not empty or any("tau_cap" not in r["first_empty"] for r in empty) else float(np.mean([r["first_empty"]["tau_cap"] for r in empty])),
            "mean_first_empty_feasible_count":None if not empty else float(np.mean([r["first_empty"]["feasible_count"] for r in empty])),
            "mean_time_total_s":float(np.mean([r["time_total_s"] for r in members])),
            "mean_time_lp_s":float(np.mean([r["time_lp_s"] for r in members]))})
    for name,data in (("capacity_summary",capacity),("capacity_regions",regions),("target_summary",targets),
                      ("oracle_per_instance",oracle),("oracle_summary",oracle_summary)):
        tools.write_csv(dest/f"{name}.csv",data)
    def table(values,columns):
        def cell(v):return "" if v is None else f"{v:.6g}" if isinstance(v,float) else str(v)
        return "| "+" | ".join(columns)+" |\n| "+" | ".join("---" for _ in columns)+" |\n"+"\n".join(
            "| "+" | ".join(cell(r.get(c)) for c in columns)+" |" for r in values)
    text=["# T7 value location and sampled interface capacity","",f"smoke_only: {meta['smoke_only']}; stages: {meta['stages']}",
        "SMOKE: no research-gate conclusions." if meta["smoke_only"] else "Single frozen checkpoint pair; research gate belongs to Claude/user.",
        "Policy arms use one greedy episode. Theta probes only rescore a fixed physical state; they are not sampled policy rollouts.",
        "", "## Policy", "",table([r for r in summary if r["M"]=="all"],
            ["dataset","arm","mean_gap_pct","p90_gap_pct","worst_gap_pct","mean_controller_calls","mean_time_total_s","mean_time_lp_s"]),
        "", "## Recovery and additivity", "",table([r for r in recovery if r["M"]=="all"],
            ["dataset","arm","recovery","denominator_positive"]),"",table(additivity,["dataset","SL08_recovery","LS08_recovery","sum","sum_minus_one"]),
        "Switches not reached are censored (null), not assigned to the final step. Remaining steps include the first switched decision; after-action counts are separate.",
        "Paired differences/p values and M/N strata: paired_summary.csv. Controller inclusive-LP time overlaps LP time; do not add these columns.",
        "", "## Capacity: L and S2 differ on the same original L state", "",
        table([r for r in capacity if r["axis"]=="all" and r["subset"]=="L_S2_different"],
            ["dataset","executor","states","rho_local_mean","rho_local_median","rho_local_p90","stability_mean","stability_median"]),
        "Full clock-decile/N distributions and rho regions: capacity_summary.csv, capacity_regions.csv. State summaries are descriptive, not independent training samples.",
        "Primary stability/difference use actual E_full L and S2 choices for both dictionaries; projected E_min choices and projected-L stability are separate fields.",
        "", "## Target-aware diagnostics", "",table(oracle_summary,["executor","target_kind","count","mean_gap_pct","first_empty_count","mean_first_empty_step"]),
        "Targets are split into proven optima and unproven MILP incumbents. Only the proven group supports an x* interpretation; the rest is incumbent-guided diagnostics.",
        "Finite sampled reachable sets underestimate R(s). Empty intersections mean not found within the sampled budget. Oracle objectives are attained lower bounds, not certified ceilings; alternate optima also matter.",
        "Even exact per-state reachability would not certify the best possible trajectory: the target intersection uses a greedy S2 ordering without backtracking.",
        "The target-only prefix ends when the target is exhausted; S2 then completes to maximality. Empty target-phase intersections trigger S2 fallback and are recorded separately from completion.",
        "No training, input ablation, multi-trajectory policy inference or ILS is performed.",""]
    (dest/"report.md").write_text("\n".join(text),encoding="utf-8")
    def digest(path):
        h=hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b""):h.update(chunk)
        return h.hexdigest()
    manifest={"schema":"t7_report_v1","smoke_only":meta["smoke_only"],"source_sha256":digest(Path(__file__)),
        "input_sha256":{p.name:digest(p) for p in dest.glob("*.jsonl")},
        "output_sha256":{p.name:digest(p) for p in [*dest.glob("*.csv"),dest/"report.md"]}}
    (dest/"report_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(f"T7 report: {len(rows)} policy/reference rows, {len(states)} state/executor rows, {len(oracle)} oracle episodes")


if __name__=="__main__":main()
