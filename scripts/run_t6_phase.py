"""T6 single-trajectory schedule experiment; legacy source is read-only. No training.

python scripts/run_t6_phase.py --device cuda:0
python scripts/run_t6_phase.py --limit 2 --output results/t6_code_checks/smoke --device cuda:0
"""
import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy_root", default="D:/PhD_Doctor/Study/Codes/RL_ModelInhance")
    parser.add_argument("--checkpoint_dir", default="checkpoints/lp_dual_v6")
    parser.add_argument("--instances_dir", default="results/t5_v6")
    parser.add_argument("--t5_results", default="results/t5_v6/attribution")
    parser.add_argument("--output", default="results/t6_phase")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, choices=(0, 2), default=0,
                        help="smoke: two evaluation cases per dataset, two N100 fitting cases per category")
    args = parser.parse_args(argv)
    t5 = load_source("_t6_t5_runtime", PROJECT / "scripts/t5_v6_attribution.py")
    root, checkpoints = map(t5.path_from_project, (args.legacy_root, args.checkpoint_dir))
    source, previous, dest = map(t5.path_from_project, (args.instances_dir, args.t5_results, args.output))
    if dest.is_relative_to(root):
        raise ValueError("Output must be outside the read-only legacy project")
    if dest.exists():
        raise FileExistsError(f"Preserve existing T6 results: {dest}")
    export = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    prior = json.loads((previous / "meta.json").read_text(encoding="utf-8"))
    reproduction = json.loads((previous / "reproduction.json").read_text(encoding="utf-8"))
    if (export.get("schema") != "t5_export_v1" or export.get("status") != "complete" or export.get("smoke_only") is not False or
            prior.get("schema") != "t5_v6_attribution_v1" or prior.get("status") != "complete" or prior.get("smoke_only") is not False or
            prior.get("legacy_unchanged") is not True or reproduction.get("instance_count") != 180 or
            reproduction.get("mismatch_count") != 0 or reproduction.get("gate_passed") is not True):
        raise ValueError("T6 requires the complete formal T5 data and reproduced baseline, even for limited smoke")
    if prior["export_manifest_sha256"] != t5.sha256(source / "manifest.json"):
        raise ValueError("T5 results and exchanged instance manifests do not match")
    prior_report = json.loads((previous / "report_manifest.json").read_text(encoding="utf-8"))
    if prior_report.get("smoke_only") is not False or prior_report["input_sha256"]["rows.jsonl"] != t5.sha256(previous / "rows.jsonl"):
        raise ValueError("Reused T5 results differ from their frozen report input SHA")
    before = t5.legacy_snapshot(root)
    if before["source_sha256"] != prior["legacy_before"]["source_sha256"]:
        raise ValueError("Legacy source differs from the reused T5 executor")
    dest.mkdir(parents=True)
    meta = {"schema": "t6_phase_v1", "status": "running", "smoke_only": bool(args.limit), "arguments": vars(args),
            "legacy_before": before, "t5_meta_sha256": t5.sha256(previous / "meta.json"),
            "t5_rows_sha256": t5.sha256(previous / "rows.jsonl"), "export_manifest_sha256": t5.sha256(source / "manifest.json"),
            "source": {"commit": t5.git(PROJECT, "rev-parse", "HEAD"), "status": t5.git(PROJECT, "status", "--short"),
                       "sha256": {p.name: t5.sha256(p) for p in (PROJECT / "scripts").glob("*.py")
                                  if p.name.startswith("t5") or "t6" in p.name}},
            "protocol": "Single greedy trajectory; T5 references reused, never rerun. No training, sampling or ILS.",
            "fit": "Only fresh L decisions from the original synthetic_trace bundle; all decisions equally weighted. "
                   "Ten clock-quantile bins, bin-median knots, empty bins skipped, tied knots merged, linear interpolation and endpoint clamping.",
            "r2": "Two folds alternate complete sorted instance IDs within category; fits never see held-out decisions. "
                  "Raw R2 and trajectory-centered R2 are both reported; centering is diagnostic only, never used by deployment.",
            "timing": "Instrumented synchronized construction including clock/interpolation/range checks and actual LP solves; "
                      "fit-trace copying and serialization excluded. Reused T5 timings retain their original environment.",
            "reference_scope": "Chu: L/G/C/D; synthetic: L/G/D. T5 did not evaluate synthetic C; no extra reference is run.",
            "smoke_design": "--limit2 fits two N100 trajectories per M/ratio category (12 total), and evaluates two instances per dataset."}
    t5.write_json(dest / "meta.json", meta)
    streams = []
    try:
        (np, torch, lp, theta_class, env_class, instance_class, instance_batch, load_benchmark,
         construct, load_checkpoint, resolve_device, seed_everything, verify, helper) = t5.bootstrap(root)
        phase = load_source("_t6_schedule", PROJECT / "scripts/t6_schedule.py")
        seed_everything(20260910, deterministic=True, cpu_threads=4)
        device = resolve_device(args.device)
        evaluation, _ = t5.read_bundle(np, instance_class, source, export, "eval", "synthetic_N100", False)
        fitting, _ = t5.read_bundle(np, instance_class, source, export, "trace", "synthetic_trace", False)
        dataset = root / "MKP_dataset/Benchmarks_Chu.mat"
        if t5.sha256(dataset) != prior["dataset_sha256"]:
            raise ValueError("Chu dataset differs from the reused T5 baseline")
        instances = load_benchmark(dataset, resources=(5, 10), items=(100, 250, 500))
        if Counter((i.m, i.n) for i in instances) != {(m,n):30 for m in (5,10) for n in (100,250,500)}:
            raise ValueError("Formal T6 Chu requires six groups of thirty")
        chu = []
        for inst in instances:
            ratio = float(inst.capacities.sum()/inst.weights.sum())
            chu.append({"id": inst.instance_id, "dataset": "chu", "M": inst.m, "N": inst.n,
                "tightness": min((.25,.5,.75), key=lambda a:abs(a-ratio)), "inst": inst,
                "reference": float(inst.reference), "reference_kind": "Chu best known; not proven optimum",
                "reference_proven": False, "data_sha256": t5.data_hash(np, inst.profits, inst.weights, inst.capacities)})
        if len({r["id"] for r in chu}) != 180:
            raise ValueError("Duplicate Chu instance IDs")
        if args.limit:
            evaluation = [next(r for r in evaluation if r["M"] == m) for m in (5,10)]
            chu = [next(r for r in chu if r["M"] == m and r["N"] == 100) for m in (5,10)]
            counts, limited = Counter(), []
            for r in fitting:
                group = phase.key(r, "category")
                if r["N"] == 100 and counts[group] < 2:
                    limited.append(r); counts[group] += 1
            fitting = limited
        records = {"synthetic_N100": evaluation, "chu": chu}
        expected = {(dataset_name, r["id"]): r for dataset_name, members in records.items() for r in members}
        old_rows = read_rows(previous / "rows.jsonl")
        index = {}
        for row in old_rows:
            pair = (row["dataset"], row["id"])
            if pair not in expected or row["arm"] not in ("L","G","C","D"):
                continue
            instance = expected[pair]
            for field in ("M", "N", "tightness", "reference", "reference_kind", "reference_proven", "data_sha256"):
                if row[field] != instance[field]:
                    raise ValueError(f"T5/T6 instance contract mismatch: {pair}, {field}")
            k = (*pair, row["arm"])
            if k in index:
                raise ValueError("Duplicate reused baseline row")
            index[k] = {**row, "reused_T5": True}
        for pair in expected:
            for arm in (("L","G","C","D") if pair[0] == "chu" else ("L","G","D")):
                if (*pair, arm) not in index:
                    raise ValueError(f"Missing T5 baseline: {pair}, {arm}")
        models, actors, configs = {}, {}, {}
        for m in (5,10):
            path = checkpoints / f"mkp_M{m}_best.pt"
            if t5.sha256(path) != prior["checkpoints"][str(m)]["sha256"]:
                raise ValueError(f"M{m} checkpoint differs from T5")
            model, payload = load_checkpoint(path, device, expected_m=m)
            if json.loads(json.dumps(payload["model_config"])) != prior["checkpoints"][str(m)]["config"]:
                raise ValueError("T5 model configuration differs")
            models[m] = model.eval().requires_grad_(False)
            actors[m], configs[m] = model.actor, model.config
        means = json.loads((previous / "theta_means.json").read_text(encoding="utf-8"))
        meta.update(device=str(device), torch=str(torch.__version__), numpy=np.__version__, checkpoints=prior["checkpoints"],
                    fitting_instances=len(fitting), evaluation_instances=len(evaluation), chu_instances=len(chu),
                    t5_means_sha256=t5.sha256(previous / "theta_means.json"))
        t5.write_json(dest / "meta.json", meta)
        def open_stream(name):
            stream = (dest / name).open("x", encoding="utf-8"); streams.append(stream); return stream
        traces_file, fitting_file, rows_file = map(open_stream, ("traces.jsonl", "fitting_progress.jsonl", "rows.jsonl"))
        def emit(stream, value):
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n"); stream.flush()
        def sync():
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        @torch.no_grad()
        def run(record, actor=None, capture=False):
            model = models[record["M"]]
            model.actor = actors[record["M"]] if actor is None else actor.to(device).eval()
            captured, overhead, lp_time, lp_calls = [], [0.], [0.], [0]
            original_solver = lp.solve_lp_relaxation
            def timed_solver(*a, **kw):
                sync(); start = time.perf_counter()
                try:
                    return original_solver(*a, **kw)
                finally:
                    sync(); lp_time[0] += time.perf_counter()-start; lp_calls[0] += 1
            def hook(module, inputs, output):
                sync(); start = time.perf_counter()
                state = inputs[0]
                clocks = {n:float(v[0]) for n,v in phase.clocks(state).items()}
                captured.append({"dataset":"synthetic_trace", "id":record["id"], "M":record["M"], "N":record["N"],
                    "tightness":record["tightness"], "step":len(captured), **clocks,
                    "theta":helper.theta_to_dict(output[0])})
                sync(); overhead[0] += time.perf_counter()-start
            handle = model.actor.register_forward_hook(hook) if capture else None
            lp.solve_lp_relaxation = timed_solver
            try:
                sync(); start = time.perf_counter()
                result = construct(instance_batch(record["inst"],1,device), model, stochastic=False)
                sync(); elapsed = time.perf_counter()-start-overhead[0]
            finally:
                lp.solve_lp_relaxation = original_solver
                if handle is not None:
                    handle.remove()
                model.actor = actors[record["M"]]
            objective, _, _, sequence = verify(record["inst"], result, 1e-7)
            if capture and len(sequence) != len(captured):
                raise ValueError("Missing fitting decision records")
            previous_clocks = {name:-1. for name in phase.CLOCKS}
            for value in captured:
                helper.validate_theta(value["theta"], configs[record["M"]])
                for name in phase.CLOCKS:
                    if value[name] < previous_clocks[name]-1e-12:
                        raise ValueError("Nonmonotone fitting clock")
                    previous_clocks[name] = value[name]
                emit(traces_file, {**value, "theta":{n:v.tolist() for n,v in value["theta"].items()}})
            row = {k:v for k,v in record.items() if k != "inst"}
            row.update(objective=objective, feasible=True, maximal=True, steps=len(sequence),
                selected_item_indices=sequence, time_total_s=elapsed, time_lp_s=lp_time[0], lp_solve_calls=lp_calls[0],
                trace_copy_excluded_s=overhead[0], reused_T5=False)
            return row, captured
        fitting_records = []
        for record in fitting:
            row, traces = run(record, capture=True)
            emit(fitting_file, {**row,"arm":"L"}); fitting_records.extend(traces)
            print(f"[fit] {record['id']}: {len(traces)} decisions, {row['time_total_s']:.2f}s", flush=True)
        tables = {}
        for grouping in ("category","M"):
            for clock in phase.CLOCKS:
                tables[f"{grouping}/{clock}"] = phase.fit_tables(fitting_records, grouping, clock, configs, helper)
        t5.write_json(dest / "schedules.json", {"schema":"t6_schedule_v1", "smoke_only":bool(args.limit), "tables":tables,
            "prices":"uniform; original vectors are never averaged by resource position",
            "clock_bounds":[0,1], "fit_dataset":"synthetic_trace"})
        r2 = phase.crossfit(fitting_records, configs, helper)
        t5.write_json(dest / "crossfit.json", r2)
        for row in index.values():
            emit(rows_file, row)
        for dataset_name, members in records.items():
            for record in members:
                for arm, (grouping, clock) in phase.ARMS.items():
                    table = tables[f"{grouping}/{clock}"][phase.key(record, grouping)]
                    base = None
                    if arm == "S4":
                        base = helper.theta_to_dict(means["C"][phase.key(record,"category")])
                        base["resource_prices"] = torch.ones_like(base["resource_prices"])
                    actor = phase.ScheduleActor(table, theta_class, configs[record["M"]], helper, base=base,
                        dynamic=phase.MINIMAL if arm == "S4" else phase.DYNAMIC,
                        price_actor=actors[record["M"]] if arm == "S5" else None)
                    row, _ = run(record, actor=actor)
                    row.update(arm=arm, grouping=grouping, clock=clock, clock_bounds=actor.clock_bounds,
                               price_source="live_L" if arm == "S5" else "uniform",
                               gap_pct=100*(1-row["objective"]/record["reference"]))
                    if actor.decisions != row["steps"]:
                        raise ValueError("Schedule evaluation must make one theta decision per selected item")
                    emit(rows_file,row)
                    print(f"[{dataset_name}] {arm} {record['id']}: {row['gap_pct']:.4f}%, {row['time_total_s']:.2f}s",flush=True)
        for stream in streams:
            stream.close()
        subprocess.run([sys.executable,"-B",str(PROJECT/"scripts/t6_report.py"),"--input",str(dest)],check=True)
        meta["report_manifest"] = json.loads((dest/"report_manifest.json").read_text(encoding="utf-8"))
        meta.update(status="complete", new_evaluation_episodes=5*(len(evaluation)+len(chu)), reused_reference_rows=len(index))
    except Exception as exc:
        meta.update(status="failed",error=repr(exc)); raise
    finally:
        for stream in streams:
            stream.close()
        meta["legacy_after"] = t5.legacy_snapshot(root)
        meta["legacy_unchanged"] = meta["legacy_after"] == before
        if not meta["legacy_unchanged"]:
            meta["status"] = "failed"
        t5.write_json(dest/"meta.json",meta)
        if not meta["legacy_unchanged"]:
            raise RuntimeError("Legacy project changed during T6")
    print(f"T6 artifacts: {dest}",flush=True)
    return dest


if __name__ == "__main__":
    main()
