"""T0 diagnostic: load immutable main_v0 source and its original delta checkpoint.

This is a separately named historical diagnostic, outside the weak-decoder method.
Arm c implements exactly delta=0,u=0,w=initial and RETAINS learned kappa.
The optional c_fullfixed also resets kappa and is the fully fixed heuristic control.

    python legacy_t0.py --checkpoint <main_v0/best.pt> --max_instances 1 --stages greedy,sample2
    python legacy_t0.py --checkpoint <main_v0/best.pt> --arms a,b,c,c_fullfixed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import types
from datetime import datetime
from pathlib import Path

import torch

from Benchmark_zeroShot import _sync, evaluate_stage
from mkp_artifacts import sha256_file, source_provenance, write_json
from mkp_config import PROJECT_ROOT, resolve_path
from mkp_evaluation import jsonable, paired_report, parse_stages, summarize, write_csv
from mkp_Trainer import set_seed
from read_mat import read_mat


def legacy_modules(revision, output_dir=None):
    """Load source into a separate package without replacing current mkp_* imports."""
    result = subprocess.run(["git", "rev-parse", "--verify", revision + "^{commit}"],
                            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    commit = result.stdout.strip()
    package_name = "_mkp_legacy_t0_" + commit[:12]
    package = types.ModuleType(package_name)
    package.__path__ = []
    sys.modules[package_name] = package
    sources, modules = {}, {}
    for filename in ("mkp_config.py", "mkp_env.py", "mkp_Model.py"):
        raw = subprocess.run(["git", "show", f"{commit}:{filename}"], cwd=PROJECT_ROOT,
                             capture_output=True, check=True).stdout
        original = raw.decode("utf-8")
        sources[filename] = {"sha256": hashlib.sha256(raw).hexdigest()}
        if output_dir is not None:
            dest = Path(output_dir) / "legacy_source"
            dest.mkdir(exist_ok=True)
            (dest / filename).write_bytes(raw)
        source = original.replace("from mkp_env import", f"from {package_name}.mkp_env import")
        name = filename[:-3]
        module = types.ModuleType(package_name + "." + name)
        module.__file__ = str(Path(PROJECT_ROOT) / filename)
        module.__package__ = package_name
        sys.modules[module.__name__] = module
        exec(compile(source, f"git:{commit}:{filename}", "exec"), module.__dict__)
        modules[name] = module
    return modules, commit, sources


def intervention(arm, cfg):
    """Forward hook changes only the named legacy controller outputs."""
    def hook(module, inputs, output):
        out = dict(output)
        if arm != "a":
            out["residual"] = torch.zeros_like(out["residual"])
        if arm in ("c", "c_fullfixed"):
            out["log_price"] = torch.zeros_like(out["log_price"])
            logits = torch.zeros_like(out["prim_w"])
            logits[:, cfg.model.primitives.index(cfg.model.init_primitive)] = cfg.model.init_primitive_logit
            out["prim_w"] = logits.softmax(-1)
        if arm == "c_fullfixed":
            out["kappa"] = torch.full_like(out["kappa"], cfg.model.kappa_init)
        return out
    return hook


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--revision", default="6a4cf82", help="immutable legacy source revision")
    parser.add_argument("--arms", default="a,b,c")
    parser.add_argument("--stages", default="greedy,sample256,greedy+ILS,sample256+ILS")
    parser.add_argument("--max_instances", type=int, default=0)
    parser.add_argument("--sizes", default="100,250,500")
    parser.add_argument("--dims", default="5,10")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--ils_rounds", type=int, default=100)
    parser.add_argument("--ls_max_iters", type=int, default=200)
    parser.add_argument("--ls_pool", type=int, default=16)
    parser.add_argument("--ls_drop_k", type=int, default=32)
    parser.add_argument("--max_rollouts_per_chunk", type=int, default=256)
    parser.add_argument("--results_dir", default="results/legacy_t0")
    args = parser.parse_args(argv)
    arms = args.arms.split(",")
    if len(set(arms)) != len(arms) or any(a not in ("a", "b", "c", "c_fullfixed") for a in arms):
        raise ValueError("--arms must be a unique subset of a,b,c,c_fullfixed")
    if args.threads < 1 or min(args.max_rollouts_per_chunk, args.ls_pool) < 1:
        raise ValueError("threads, rollout chunk and search pool must be positive")
    if min(args.ils_rounds, args.ls_max_iters, args.ls_drop_k, args.max_instances) < 0:
        raise ValueError("budgets and max_instances must be nonnegative")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    checkpoint = resolve_path(args.checkpoint)
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)
    out_dir = Path(resolve_path(args.results_dir)) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir.mkdir(parents=True)
    modules, commit, sources = legacy_modules(args.revision, out_dir)
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = modules["mkp_config"].config_from_dict(ck["config"])
    model = modules["mkp_Model"].build_model(cfg).to(device).eval()
    model.load_state_dict(ck["model"], strict=True)
    env = modules["mkp_env"].MKPEnv(cfg.env)
    ec = cfg.eval
    for key in ("num_samples", "ils_rounds", "ls_max_iters", "ls_pool", "ls_drop_k", "max_rollouts_per_chunk"):
        setattr(ec, key, args.samples if key == "num_samples" else getattr(args, key))
    stages = [s["name"] for s in parse_stages(args.stages.split(","), args.samples)]
    dataset_path = resolve_path("MKP_dataset/Benchmarks_Chu.mat")
    data = [d for d in read_mat(dataset_path) if d["M"] in [int(v) for v in args.dims.split(",")]
            and d["N"] in [int(v) for v in args.sizes.split(",")]]
    data = data[:args.max_instances] if args.max_instances else data
    if not data:
        raise ValueError("No selected Chu instances")
    meta = {"schema": "legacy_t0_v1", "status": "running", "legacy_commit": commit,
            "legacy_sources": sources, "source": source_provenance(), "checkpoint": checkpoint,
            "checkpoint_sha256": sha256_file(checkpoint), "config": cfg.to_dict(), "arguments": vars(args),
            "data_sha256": sha256_file(dataset_path), "instance_ids": [d["id"] for d in data],
            "interventions": {"a": "original legacy checkpoint", "b": "delta=0 only",
                              "c": "delta=0,u=0,w=initial; LEARNED KAPPA RETAINED: not fully fixed for sampling",
                              "c_fullfixed": "delta=0,u=0,w=initial,kappa=initial: fully fixed heuristic"},
            "boundary": "Legacy historical diagnostic contains learned item residuals in arm a. "
                        "None of these results validate the new weak-decoder method.",
            "timing": "Shared root LP charged once to each arm/stage. Trace capture included; JSON output "
                      "and feasibility/maximality audits excluded. Sampling excludes separate greedy rollout.",
            "paired_delta_sign": "a-minus-b and b-minus-c gap differences are percentage points; negative is better"}
    protocol_sha = hashlib.sha256(json.dumps({"legacy_commit": commit, "env": cfg.to_dict()["env"],
                                             "model": cfg.to_dict()["model"]}, sort_keys=True).encode()).hexdigest()
    write_json(out_dir / "meta.json", meta)
    rows = []
    try:
        with (out_dir / "trajectories.jsonl").open("w", encoding="utf-8") as stream:
            for index, d in enumerate(data):
                inst = modules["mkp_env"].instances_from_dicts([d], device=device)
                _sync(device)
                start = time.perf_counter()
                pre = env.precompute(inst)
                _sync(device)
                lp_seconds = time.perf_counter() - start
                for arm in arms:
                    handle = model.controller.register_forward_hook(intervention(arm, cfg))
                    try:
                        for stage in stages:
                            def sink(value):
                                stream.write(json.dumps(jsonable({"id": d["id"], "seed": args.seed, **value})) + "\n")
                            rec = evaluate_stage(model, env, inst, ec, device, False, pre, stage,
                                                 "legacy_" + arm, args.seed, d["id"], lp_seconds, sink)
                            sol = rec.pop("solution")
                            row = {"id": d["id"], "M": d["M"], "N": d["N"],
                                   "alpha": round(float(inst.c.sum() / inst.r.sum()), 2),
                                   "seed": args.seed, "evaluation_seed": args.seed, "arm": arm,
                                   "data_sha256": meta["data_sha256"], "protocol_sha256": protocol_sha,
                                   "best_known": float(d["best"]), **rec}
                            row["gap_pct"] = (row["best_known"] - row["objective"]) / row["best_known"] * 100
                            rows.append(row)
                            sink({"arm": arm, "stage": stage, "best_solution": sol, "objective": row["objective"]})
                            print(f"[{index + 1}/{len(data)}] id={d['id']} arm={arm} {stage} "
                                  f"gap={row['gap_pct']:.4f}% time={row['time_total_s']:.2f}s", flush=True)
                    finally:
                        handle.remove()
                write_csv(out_dir / "per_instance.csv", rows)
                stream.flush()
        write_csv(out_dir / "summary.csv", summarize(rows))
        pairs, stats = paired_report(rows)
        write_csv(out_dir / "paired_per_instance.csv", pairs)
        write_csv(out_dir / "paired_summary.csv", stats)
        meta["status"], meta["row_count"] = "complete", len(rows)
    except Exception as exc:
        meta["status"], meta["error"] = "failed", repr(exc)
        raise
    finally:
        write_json(out_dir / "meta.json", meta)
    print(f"Results: {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
