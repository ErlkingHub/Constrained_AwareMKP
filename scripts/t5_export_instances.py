"""Export new-project synthetic data for a separate, read-only v6 process.

    python scripts/t5_export_instances.py
    python scripts/t5_export_instances.py --limit 2 --output results/t5_code_checks/export_smoke

The default exports the unchanged 128-instance N100 evaluation set plus a fresh
144-instance trace-fitting set. --limit is either 0 or 2: the latter exports one
evaluation example per M and one trace example per each of the 18 categories.
No model, legacy module, inference or training is imported or run here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from mkp_artifacts import instance_sha, sha256_file, source_provenance, write_json
from mkp_config import config_from_dict, resolve_path
from mkp_env import MKPInstances, generate_instances
from mkp_reference import milp_reference

TRACE_SEED = 20261107
BUNDLE_SCHEMA = "t5_instance_bundle_v1"


def export_data_sha256(p, w, c):
    """Hash the exact exported float64 values/layout; copied by the legacy reader.

    Each array contributes its name, tuple-shaped string, dtype string and C-order
    bytes. w has shape (N,M), unlike the new environment's r with shape (B,M,N).
    The separate source_data_sha256 preserves the original float32 project hash.
    """
    digest = hashlib.sha256()
    for name, value in (("p", p), ("w", w), ("c", c)):
        array = np.ascontiguousarray(value, dtype=np.float64)
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _one(batch, index):
    return MKPInstances(batch.p[index:index+1], batch.r[index:index+1], batch.c[index:index+1])


def _append(records, arrays, inst, key, tightness, reference=None,
            reference_kind="trace fitting only; not evaluated", reference_proven=False, **extra):
    i = len(records)
    p = np.ascontiguousarray(inst.p[0].cpu().numpy(), dtype=np.float64)
    w = np.ascontiguousarray(inst.r[0].transpose(0, 1).cpu().numpy(), dtype=np.float64)
    c = np.ascontiguousarray(inst.c[0].cpu().numpy(), dtype=np.float64)
    if p.shape != (inst.N,) or w.shape != (inst.N, inst.M) or c.shape != (inst.M,):
        raise ValueError("Invalid exported array shape")
    if not all(np.isfinite(a).all() for a in (p, w, c)) or np.any(p <= 0) or np.any(w < 0) or np.any(c <= 0):
        raise ValueError("MKP export contains non-finite or invalid values")
    if reference is not None and (not math.isfinite(reference) or reference <= 0):
        raise ValueError("Invalid integer evaluation reference")
    records.append({"id": key, "M": inst.M, "N": inst.N, "tightness": float(tightness),
                    "actual_capacity_ratio": float(c.sum() / w.sum()),
                    "reference": reference, "reference_kind": reference_kind,
                    "reference_proven": bool(reference_proven),
                    "data_sha256": export_data_sha256(p, w, c),
                    "source_data_sha256": instance_sha(inst), **extra})
    arrays.update({f"p_{i}": p, f"w_{i}": w, f"c_{i}": c})


def evaluation_data(cfg, smoke_only):
    """Generate the same complete batches before selecting smoke examples.

    Full runs use the identical two 64-instance reference-cache keys as T3/T4.
    Smoke uses the previously established one-instance cache for each M and
    cannot substitute for the formal 128-instance evaluation set.
    """
    records, arrays, batches = [], {}, []
    rng = torch.Generator().manual_seed(cfg.train.val_seed)
    for m in (5, 10):
        batch = generate_instances(64, 100, m, cfg.env.train_tightness, cfg.env.weight_max,
                                   cfg.env.profit_noise, generator=rng)
        take = 1 if smoke_only else 64
        selected = MKPInstances(batch.p[:take], batch.r[:take], batch.c[:take])
        refs = milp_reference(selected, cfg.train.val_milp_time_limit, cfg.train.val_mip_rel_gap,
                              cfg.train.reference_cache)
        batches.append({"M": m, "N": 100, "generated_count": 64, "exported_count": take,
                        "source_batch_sha256": instance_sha(batch),
                        "exported_source_batch_sha256": instance_sha(selected),
                        "reference_path": refs["path"], "reference_sha256": refs["sha256"],
                        "reference_settings": refs["settings"],
                        "reference_proven_count": int(refs["optimal"].sum())})
        for b in range(take):
            inst = _one(batch, b)
            ratio = float(inst.c.sum() / inst.r.sum())
            alpha = min((.25, .5, .75), key=lambda a: abs(a-ratio))
            bound = float(refs["bound"][b])
            _append(records, arrays, inst, f"syn100_M{m}_{b:03d}", alpha,
                    reference=float(refs["obj"][b]),
                    reference_kind="MILP optimum" if refs["optimal"][b] else "MILP incumbent",
                    reference_proven=refs["optimal"][b], reference_sha256=refs["sha256"],
                    reference_bound=bound if math.isfinite(bound) else None,
                    reference_status=int(refs["status"][b]))
    return records, arrays, batches


def trace_data(cfg, smoke_only):
    records, arrays, categories = [], {}, []
    rng = torch.Generator().manual_seed(TRACE_SEED)
    for m in (5, 10):
        for n in (100, 250, 500):
            for alpha in (.25, .5, .75):
                # Always generate eight: smoke and full exports share their first
                # example within each category and the same RNG consumption order.
                batch = generate_instances(8, n, m, (alpha,), cfg.env.weight_max,
                                           cfg.env.profit_noise, generator=rng)
                take = 1 if smoke_only else 8
                categories.append({"M": m, "N": n, "tightness": alpha,
                                   "generated_count": 8, "exported_count": take,
                                   "source_batch_sha256": instance_sha(batch)})
                for b in range(take):
                    _append(records, arrays, _one(batch, b), f"trace_M{m}_N{n}_a{alpha:g}_{b:03d}", alpha)
    return records, arrays, categories


def previous_stage_hashes(source_batches):
    """Read existing T3/T4 manifests only; never invoke either experiment."""
    expected = {b["M"]: b["source_batch_sha256"] for b in source_batches}
    output = []
    for name, relative in (("T3-a", "results/t3a_scan/meta.json"), ("T4-a", "results/t4a_sms/meta.json")):
        path = Path(resolve_path(relative))
        record = {"stage": name, "file": str(path), "available": path.is_file(), "comparisons": []}
        if path.is_file():
            previous = json.loads(path.read_text(encoding="utf-8"))
            record.update(sha256=sha256_file(path), status=previous.get("status"),
                          smoke_only=bool(previous.get("smoke_only")))
            for item in previous.get("data", []):
                if item.get("dataset") != "synthetic_N100" or item.get("N") != 100 or item.get("M") not in expected:
                    continue
                # Smoke manifests describe a subset hash and are not formal
                # identity witnesses for an entire 64-instance source batch.
                if previous.get("smoke_only"):
                    continue
                record["comparisons"].append({"M": item["M"],
                    "previous_data_sha256": item["data_sha256"],
                    "source_batch_sha256": expected[item["M"]],
                    "matches": item["data_sha256"] == expected[item["M"]]})
        output.append(record)
    return output


def write_bundle(path, dataset, seed, smoke_only, records, arrays, **extra):
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate instance IDs in export")
    metadata = {"schema": BUNDLE_SCHEMA, "dataset": dataset, "seed": int(seed),
                "smoke_only": bool(smoke_only), "count": len(records), "records": records,
                "layout": {"p": "p_{i}: (N,)", "w": "w_{i}: (N,M)", "c": "c_{i}: (M,)",
                           "dtype": "float64", "allow_pickle_required": False}, **extra}
    payload = {"metadata": np.asarray(json.dumps(metadata, ensure_ascii=False, allow_nan=False)), **arrays}
    with path.open("xb") as stream:
        np.savez_compressed(stream, **payload)
    # Verify the exchange format immediately, using the same pickle-free mode
    # required of the independent legacy process. This performs no inference.
    with np.load(path, allow_pickle=False) as bundle:
        recovered = json.loads(str(bundle["metadata"].item()))
        if recovered != metadata:
            raise RuntimeError("NPZ metadata did not round-trip")
        for i, record in enumerate(records):
            values = [bundle[f"{name}_{i}"] for name in ("p", "w", "c")]
            if any(a.dtype != np.float64 for a in values) or export_data_sha256(*values) != record["data_sha256"]:
                raise RuntimeError("NPZ values did not round-trip with matching content hashes")
    return {"file": str(path), "sha256": sha256_file(path), "count": len(records),
            "dataset": dataset, "smoke_only": bool(smoke_only)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="runs/t1p_static_s1234/config.json")
    parser.add_argument("--output", default="results/t5_v6")
    parser.add_argument("--limit", type=int, choices=(0, 2), default=0,
                        help="2 exports only two evaluation and eighteen trace smoke examples")
    args = parser.parse_args(argv)
    source = Path(resolve_path(args.config))
    cfg = config_from_dict(json.loads(source.read_text(encoding="utf-8")))
    if (cfg.device.seed != 1234 or cfg.train.val_sizes != [100] or cfg.train.val_dims != [5, 10]
            or cfg.train.val_instances != 64):
        raise ValueError("T5 requires the inherited seed1234 N100/M5,10 x64 validation configuration")
    if TRACE_SEED == cfg.train.val_seed:
        raise ValueError("Trace-fitting seed must be independent of the evaluation generation seed")
    dest = Path(resolve_path(args.output))
    if dest.exists():
        raise FileExistsError(f"Preserve existing T5 exports: {dest}")
    dest.mkdir(parents=True)
    smoke_only = bool(args.limit)
    manifest = {"schema": "t5_export_v1", "status": "running", "smoke_only": smoke_only,
                "arguments": vars(args), "source_config": str(source),
                "source_config_sha256": sha256_file(source), "source": source_provenance(),
                "config": cfg.to_dict(), "training_seed": cfg.device.seed,
                "evaluation_generation_seed": cfg.train.val_seed, "trace_generation_seed": TRACE_SEED,
                "bundles": {},
                "isolation": "Only new-project generator/reference modules run here. The v6 reader must run "
                             "in a separate process. No legacy path is imported or written.",
                "layout_conversion": "Raw integer-valued generator data are exactly preserved; float32 is "
                                     "upcast to float64, and r(M,N) is transposed to w(N,M). No normalization.",
                "hashes": "data_sha256 hashes exported float64 p/w/c name+shape+dtype+bytes. "
                          "source_data_sha256 and batch hashes use the original new-project float32 layout.",
                "scope": "Evaluation references are MILP incumbents with proven-optimum flags. "
                         "Trace records have null reference and are fitting data only. No inference/training."}
    write_json(dest / "manifest.json", manifest)
    try:
        eval_records, eval_arrays, batches = evaluation_data(cfg, smoke_only)
        prior = previous_stage_hashes(batches)
        manifest.update(evaluation_source_batches=batches, previous_stage_hashes=prior)
        if any(not c["matches"] for p in prior for c in p["comparisons"]):
            raise ValueError("Generated N100 source hashes differ from an existing formal T3/T4 manifest")
        trace_records, trace_arrays, categories = trace_data(cfg, smoke_only)
        if len(eval_records) != (2 if smoke_only else 128) or len(trace_records) != (18 if smoke_only else 144):
            raise RuntimeError("Export counts differ from the registered T5 design")
        manifest["trace_categories"] = categories
        manifest["bundles"]["eval"] = write_bundle(dest / "instances_eval.npz", "synthetic_N100",
            cfg.train.val_seed, smoke_only, eval_records, eval_arrays, source_batches=batches)
        manifest["bundles"]["trace"] = write_bundle(dest / "instances_trace.npz", "synthetic_trace",
            TRACE_SEED, smoke_only, trace_records, trace_arrays, categories=categories,
            selection_scope="Fit activated-space G/C theta constants only; not evaluation")
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        write_json(dest / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "smoke_only": smoke_only,
                      "bundles": manifest["bundles"]}, indent=2))
    return dest


if __name__ == "__main__":
    main()
