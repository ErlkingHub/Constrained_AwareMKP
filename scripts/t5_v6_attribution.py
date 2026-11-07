"""Read-only, legacy-only T5 inference. Never trains or imports new mkp modules.

First export data with t5_export_instances.py in a separate process. This entry
point places legacy_root first on sys.path, and runs reporting in another process.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]


def path_from_project(value):
    path = Path(value)
    return (path if path.is_absolute() else PROJECT / path).resolve()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def git(root, *args):
    result = subprocess.run(["git", "--no-optional-locks", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), *args],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(f"Cannot record read-only git provenance: {result.stderr.strip()}")
    return result.stdout.strip()


def legacy_snapshot(root):
    names = ("mkp_Model.py", "mkp_env.py", "mkp_config.py", "mkp_lp.py", "mkp_basis.py",
             "mkp_utils.py", "mkp_data.py", "mkp_performance.py", "read_mat.py")
    return {"commit": git(root, "rev-parse", "HEAD"), "branch": git(root, "branch", "--show-current"),
            "status": git(root, "status", "--porcelain=v1"),
            "source_sha256": {name: sha256(root / name) for name in names}}


def bootstrap(root):
    # Remove the new project and empty cwd entries, rather than just prepending.
    sys.path[:] = [str(root)] + [p for p in sys.path if p and Path(p).resolve() not in (PROJECT, PROJECT / "scripts", root)]
    import numpy as np
    import torch
    import mkp_lp
    from mkp_Model import StrategyTheta
    from mkp_env import MKPEnv
    from mkp_data import MKPInstance, instance_batch, load_benchmark
    from mkp_utils import construct_solution, load_checkpoint, resolve_device, seed_everything
    from Benchmark_zeroShot import verify_solution
    spec = importlib.util.spec_from_file_location("_t5_theta", PROJECT / "scripts/t5_theta.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    for name, module in list(sys.modules.items()):
        if (name.startswith("mkp_") or name in ("Benchmark_zeroShot", "read_mat")) and getattr(module, "__file__", None):
            if not Path(module.__file__).resolve().is_relative_to(root):
                raise RuntimeError(f"Module isolation violated: {name}: {module.__file__}")
    return np, torch, mkp_lp, StrategyTheta, MKPEnv, MKPInstance, instance_batch, load_benchmark, construct_solution, load_checkpoint, resolve_device, seed_everything, verify_solution, helper


def data_hash(np, p, w, c):
    digest = hashlib.sha256()
    for name, value in (("p", p), ("w", w), ("c", c)):
        value = np.ascontiguousarray(value, dtype=np.float64)
        digest.update(name.encode()); digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode()); digest.update(value.tobytes())
    return digest.hexdigest()


def read_bundle(np, cls, directory, manifest, key, expected_dataset, smoke):
    entry = manifest["bundles"][key]
    path = directory / ("instances_eval.npz" if key == "eval" else "instances_trace.npz")
    if sha256(path) != entry["sha256"]:
        raise ValueError(f"Bundle SHA mismatch: {path}")
    records = []
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata["schema"] != "t5_instance_bundle_v1" or metadata["dataset"] != expected_dataset:
            raise ValueError("Wrong T5 exchange format or dataset")
        if metadata["smoke_only"] and not smoke:
            raise ValueError("Formal inference cannot use smoke exports")
        if metadata["count"] != len(metadata["records"]) or entry["count"] != metadata["count"]:
            raise ValueError("Bundle count mismatch")
        for i, record in enumerate(metadata["records"]):
            arrays = [archive[f"{name}_{i}"].copy() for name in ("p", "w", "c")]
            if any(a.dtype != np.float64 for a in arrays) or data_hash(np, *arrays) != record["data_sha256"]:
                raise ValueError(f"Invalid exchanged arrays: {record['id']}")
            inst = cls(str(record["id"]), *arrays, reference=record["reference"])
            if (inst.m, inst.n) != (record["M"], record["N"]):
                raise ValueError("Declared instance dimensions differ")
            records.append({**record, "inst": inst, "dataset": expected_dataset})
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate exchanged instance IDs")
    categories = Counter((r["M"], r["N"], r["tightness"]) for r in records)
    if key == "trace":
        expected = {(m, n, a) for m in (5, 10) for n in (100, 250, 500) for a in (.25, .5, .75)}
        if set(categories) != expected or set(categories.values()) != ({1} if metadata["smoke_only"] else {8}):
            raise ValueError("Trace bundle lacks the eighteen registered categories/budgets")
    elif Counter(r["M"] for r in records) != {5: 1 if metadata["smoke_only"] else 64,
                                               10: 1 if metadata["smoke_only"] else 64}:
        raise ValueError("Evaluation bundle lacks the registered M5/M10 counts")
    if smoke:
        seen = set()
        kept = []
        for record in records:
            category = (record["M"], record["N"], record["tightness"]) if key == "trace" else record["M"]
            if category not in seen:
                kept.append(record); seen.add(category)
        records = kept
    return records, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy_root", default="D:/PhD_Doctor/Study/Codes/RL_ModelInhance")
    parser.add_argument("--checkpoint_dir", default="checkpoints/lp_dual_v6")
    parser.add_argument("--instances_dir", default="results/t5_v6")
    parser.add_argument("--output", default="results/t5_v6/attribution")
    parser.add_argument("--historical", default=None, help="Historical v6 run directory")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, choices=(0, 2), default=0, help="2 means smoke-only, two cases per evaluated dataset")
    args = parser.parse_args(argv)
    root, checkpoint_dir = path_from_project(args.legacy_root), path_from_project(args.checkpoint_dir)
    source, dest = path_from_project(args.instances_dir), path_from_project(args.output)
    historical = path_from_project(args.historical) if args.historical else root / "results/20261022_195937_430788"
    if dest.is_relative_to(root) or source.is_relative_to(root):
        raise ValueError("T5 outputs and exchanged data must be outside the read-only legacy project")
    if dest.exists():
        raise FileExistsError(f"Preserve existing T5 output: {dest}")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "t5_export_v1" or manifest.get("status") != "complete":
        raise ValueError("A complete T5 export is required")
    if manifest["smoke_only"] and not args.limit:
        raise ValueError("Use --limit 2 for smoke exports; they cannot be formal experiments")
    before = legacy_snapshot(root)
    history = json.loads((historical / "manifest.json").read_text(encoding="utf-8"))
    history_path = historical / "solutions.jsonl"
    saved = {}
    for line in history_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["method"] == "learned_dynamic":
            iid = str(row["instance_id"])
            if iid in saved:
                raise ValueError("Duplicate historical L result")
            saved[iid] = row
    if len(saved) != 180:
        raise ValueError("Require all 180 historical L objectives")
    dest.mkdir(parents=True)
    meta = {"schema": "t5_v6_attribution_v1", "status": "running", "smoke_only": bool(args.limit),
            "arguments": vars(args), "legacy_before": before, "export_manifest_sha256": sha256(source / "manifest.json"),
            "historical_manifest_sha256": sha256(historical / "manifest.json"),
            "historical_solutions_sha256": sha256(history_path), "checkpoints": {},
            "new_source": {"commit": git(PROJECT, "rev-parse", "HEAD"), "status": git(PROJECT, "status", "--short"),
                "source_sha256": {p.name: sha256(p) for p in (PROJECT / "scripts").glob("t5*.py")}},
            "protocol": "Only legacy decoder/LP/feasibility and greedy construction; actor-only theta injection. No training.",
            "timing": "Synchronized instrumented construction including transfer and actual LP solves. Trace-copy time is "
                      "subtracted, serialization/audits excluded. Recording barriers may affect execution latency; no warmup.",
            "statistics": "G/C fit every actual trace decision with equal weight, in activated space; Chu confirms only."}
    write_json(dest / "meta.json", meta)
    row_file = trace_file = None
    try:
        (np, torch, lp_module, theta_class, env_class, instance_class, instance_batch, load_benchmark,
         construct_solution, load_checkpoint, resolve_device, seed_everything, verify_solution, helper) = bootstrap(root)
        seed_everything(int(history["config"]["seed"]), deterministic=True, cpu_threads=int(history["config"]["cpu_threads"]))
        device = resolve_device(args.device)
        evaluation, eval_meta = read_bundle(np, instance_class, source, manifest, "eval", "synthetic_N100", bool(args.limit))
        fitting, trace_meta = read_bundle(np, instance_class, source, manifest, "trace", "synthetic_trace", bool(args.limit))
        if eval_meta["smoke_only"] != trace_meta["smoke_only"] or eval_meta["smoke_only"] != manifest["smoke_only"]:
            raise ValueError("Export smoke flags differ")
        if eval_meta["seed"] != 20240607 or trace_meta["seed"] != 20261107:
            raise ValueError("T5 generation seeds differ from the registered protocol")
        dataset_path = root / "MKP_dataset/Benchmarks_Chu.mat"
        if sha256(dataset_path) != history["dataset_sha256"]:
            raise ValueError("Chu dataset differs from historical reproduction data")
        instances = load_benchmark(dataset_path, resources=(5, 10), items=(100, 250, 500))
        if (Counter((i.m, i.n) for i in instances) != {(m, n): 30 for m in (5, 10) for n in (100, 250, 500)}
                or {i.instance_id for i in instances} != set(saved)):
            raise ValueError("Require six Chu groups of 30, identical to historical IDs")
        if args.limit:
            instances = [next(i for i in instances if i.m == m and i.n == 100) for m in (5, 10)]
        chu = []
        for inst in instances:
            actual_ratio = float(inst.capacities.sum() / inst.weights.sum())
            chu.append({"id": inst.instance_id, "M": inst.m, "N": inst.n, "inst": inst, "dataset": "chu",
                        "tightness": min((.25, .5, .75), key=lambda a: abs(a-actual_ratio)),
                        "reference": float(inst.reference), "reference_kind": "Chu best known; not proven optimum",
                        "reference_proven": False, "actual_capacity_ratio": actual_ratio,
                        "data_sha256": data_hash(np, inst.profits, inst.weights, inst.capacities)})
        models, actors = {}, {}
        for m in (5, 10):
            path = checkpoint_dir / f"mkp_M{m}_best.pt"
            actual_sha = sha256(path)
            if actual_sha != history["checkpoints"][str(m)]["sha256"]:
                raise ValueError(f"M{m} checkpoint does not match historical best SHA")
            model, payload = load_checkpoint(path, device, expected_m=m)
            models[m] = model.eval().requires_grad_(False)
            actors[m] = model.actor
            meta["checkpoints"][str(m)] = {"path": str(path), "sha256": actual_sha,
                "config": payload["model_config"], "checkpoint_version": payload["checkpoint_version"],
                "controller_version": payload["controller_version"], "decoder_version": payload["decoder_version"]}
        meta.update(device=str(device), torch=str(torch.__version__), evaluation_instances=len(evaluation),
                    fitting_instances=len(fitting), chu_instances=len(chu), dataset_sha256=sha256(dataset_path))
        write_json(dest / "meta.json", meta)
        row_file = (dest / "rows.jsonl").open("x", encoding="utf-8")
        trace_file = (dest / "traces.jsonl").open("x", encoding="utf-8")
        outcomes, learned_traces = [], {}

        def sync():
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        def emit(stream, row):
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"); stream.flush()

        @torch.no_grad()
        def solve(record, arm, theta=None, capture=False, evaluated=True):
            inst, m = record["inst"], record["M"]
            model = models[m]
            model.actor = actors[m] if theta is None else helper.injected_actor(theta, theta_class).to(device)
            captured, overhead = [], [0.0]
            lp_elapsed, lp_calls = [0.0], [0]
            original_solver = lp_module.solve_lp_relaxation
            def timed_solver(*a, **kw):
                sync(); start = time.perf_counter()
                try:
                    return original_solver(*a, **kw)
                finally:
                    sync(); lp_elapsed[0] += time.perf_counter() - start; lp_calls[0] += 1
            def hook(module, inputs, output):
                sync(); start = time.perf_counter()
                captured.append(helper.theta_to_dict(output[0]))
                sync(); overhead[0] += time.perf_counter() - start
            handle = model.actor.register_forward_hook(hook) if capture else None
            lp_module.solve_lp_relaxation = timed_solver
            try:
                sync(); start = time.perf_counter()
                result = construct_solution(instance_batch(inst, 1, device), model, stochastic=False)
                sync(); elapsed = max(0.0, time.perf_counter() - start - overhead[0])
            finally:
                lp_module.solve_lp_relaxation = original_solver
                if handle is not None:
                    handle.remove()
                model.actor = actors[m]
            objective, _, _, sequence = verify_solution(inst, result, float(history["config"]["feasibility_tolerance"]))
            for step, value in enumerate(captured):
                helper.validate_theta(value, model.config)
                emit(trace_file, {"dataset": record["dataset"], "arm": arm, "id": record["id"],
                    "M": m, "N": inst.n, "tightness": record["tightness"], "step": step,
                    "theta": {k: v.tolist() for k, v in value.items()}})
            if capture and len(captured) != len(sequence):
                raise RuntimeError("Theta records do not match actual construction decisions")
            row = {key: value for key, value in record.items() if key != "inst"}
            row.update(arm=arm, objective=objective, feasible=True, maximal=True,
                       time_lp_s=lp_elapsed[0], lp_solve_calls=lp_calls[0], time_total_s=elapsed,
                       trace_copy_excluded_s=overhead[0], steps=len(sequence), selected_item_indices=sequence)
            if evaluated:
                row["gap_pct"] = 100 * (1 - objective / record["reference"])
                outcomes.append(row); emit(row_file, row)
            else:
                emit(fitting_progress, row)
            if capture:
                learned_traces[(record["dataset"], record["id"])] = captured
            print(f"[{record['dataset']}] {arm} {record['id']}: objective={objective:g}, {elapsed:.2f}s", flush=True)
            return row, captured

        # Complete preflight before fitting constants or evaluating any other arm.
        deltas, mismatch = [], []
        for record in chu:
            row, _ = solve(record, "L", capture=True)
            old_objective = float(saved[record["id"]]["objective"])
            delta = row["gap_pct"] - 100 * (1 - old_objective / record["reference"])
            deltas.append(delta)
            if row["objective"] != old_objective:
                mismatch.append({"id": record["id"], "historical_objective": old_objective,
                                 "actual_objective": row["objective"], "gap_delta_pp": delta})
        reproduction = {"status": "partial_smoke" if args.limit else "complete", "smoke_only": bool(args.limit),
                        "instance_count": len(chu), "historical_instance_count": 180,
                        "mismatch_count": len(mismatch), "mismatches": mismatch,
                        "mean_gap_delta_pp": float(np.mean(deltas)), "threshold_pp": .02,
                        "gate_assessed": not bool(args.limit)}
        reproduction["gate_passed"] = None if args.limit else abs(reproduction["mean_gap_delta_pp"]) <= .02
        if reproduction["gate_passed"] is False:
            reproduction["status"] = "review_required"
        write_json(dest / "reproduction.json", reproduction)
        if not args.limit and abs(reproduction["mean_gap_delta_pp"]) > .02:
            raise RuntimeError("L reproduction differs by >0.02 pp: stop before any other arm")
        global_records, category_records = defaultdict(list), defaultdict(list)
        with (dest / "fitting_progress.jsonl").open("x", encoding="utf-8") as fitting_progress:
            for record in fitting:
                _, values = solve(record, "L", capture=True, evaluated=False)
                global_records[record["M"]].extend(values)
                category_records[(record["M"], record["N"], record["tightness"])].extend(values)
        means = {m: helper.mean_theta(values, models[m].config) for m, values in global_records.items()}
        category_means = {key: helper.mean_theta(values, models[key[0]].config) for key, values in category_records.items()}
        defaults = {}
        for m in (5, 10):
            record = next(r for r in fitting if r["M"] == m)
            state = env_class(*instance_batch(record["inst"], 1, device)).state
            defaults[m] = helper.get_default(models[m].config, state)
        serial = lambda theta: {k: v.tolist() for k, v in theta.items()}
        write_json(dest / "theta_means.json", {"G": {str(k): serial(v) for k, v in means.items()},
            "C": {"|".join(map(str,k)): serial(v) for k,v in category_means.items()},
            "D": {str(k): serial(v) for k,v in defaults.items()},
            "weighting": "equal actual decisions, activated theta; G per M, C per M/N/nominal ratio"})
        for record in evaluation:
            m = record["M"]
            solve(record, "L", capture=True)
            solve(record, "D", defaults[m])
            solve(record, "G", means[m])
            for term in helper.ABLATION_FIELDS:
                solve(record, f"G-no-{term}", helper.ablated(means[m], term, models[m].config))
        g_gap = float(np.mean([r["gap_pct"] for r in outcomes if r["dataset"] == "synthetic_N100" and r["arm"] == "G"]))
        ablation_deltas = {term: float(np.mean([r["gap_pct"] for r in outcomes if r["dataset"] == "synthetic_N100"
                                              and r["arm"] == f"G-no-{term}"])) - g_gap for term in helper.ABLATION_FIELDS}
        retained = [term for term, delta in ablation_deltas.items() if delta >= .05]
        selection = {"schema": "t5_dictionary_selection_v1", "smoke_only": bool(args.limit),
                     "dataset": "synthetic_N100", "count": len(evaluation), "removed_minus_G_gap_pp": ablation_deltas,
                     "G_min_retained": retained, "threshold_pp": .05,
                     "rule": "retain only terms whose individual removal worsens synthetic mean gap by >=.05 pp"}
        write_json(dest / "selection.json", selection)
        for record in evaluation:
            m = record["M"]
            for arm, keep in (("G-min", retained), ("G-F0", ("prices", "profit_exponent"))):
                theta = helper.theta_to_dict(means[m])
                for term in helper.ABLATION_FIELDS:
                    if term not in keep:
                        theta = helper.ablated(theta, term, models[m].config)
                solve(record, arm, theta)
        for record in chu:
            m = record["M"]
            solve(record, "D", defaults[m]); solve(record, "G", means[m])
            solve(record, "C", category_means[(m, record["N"], record["tightness"])])
            own_mean = helper.mean_theta(learned_traces[("chu", record["id"])], models[m].config)
            solve(record, "I", own_mean)
        row_file.close(); row_file = None
        trace_file.close(); trace_file = None
        subprocess.run([sys.executable, "-B", str(PROJECT / "scripts/t5_report.py"), "--input", str(dest)], check=True)
        meta["report_manifest"] = json.loads((dest / "report_manifest.json").read_text(encoding="utf-8"))
        meta.update(status="complete", evaluated_episodes=len(outcomes), fitting_episodes=len(fitting),
                    artifact_files=[p.name for p in dest.iterdir() if p.is_file()])
    except Exception as exc:
        meta.update(status="failed", error=repr(exc))
        raise
    finally:
        if row_file is not None:
            row_file.close()
        if trace_file is not None:
            trace_file.close()
        meta["legacy_after"] = legacy_snapshot(root)
        meta["legacy_unchanged"] = meta["legacy_after"] == before
        if not meta["legacy_unchanged"]:
            meta.update(status="failed", legacy_error="Legacy status/source snapshot changed")
        write_json(dest / "meta.json", meta)
        if not meta["legacy_unchanged"]:
            raise RuntimeError("Legacy project changed during T5")
    print(f"T5 artifacts: {dest}", flush=True)
    return dest


if __name__ == "__main__":
    main()
