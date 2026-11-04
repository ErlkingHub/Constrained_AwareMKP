"""Cached HiGHS integer references for generated validation instances only.

An incumbent from a time-limited solve is a reference, not a proven optimum.
Every cache stores that distinction, its dual bound and its feasible binary vector.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp

from mkp_artifacts import instance_sha, sha256_file
from mkp_config import resolve_path


def milp_reference(inst, time_limit=10.0, mip_rel_gap=1e-9, cache_dir="cache", log=print):
    """Return obj/optimal/bound/solution arrays and cache provenance for one batch."""
    if time_limit <= 0 or mip_rel_gap < 0:
        raise ValueError("MILP reference needs a positive time limit and nonnegative gap")
    settings = {"schema": "synthetic_milp_reference_v1", "data_sha256": instance_sha(inst),
                "time_limit": float(time_limit), "mip_rel_gap": float(mip_rel_gap),
                "scipy_version": scipy.__version__}
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    directory = resolve_path(cache_dir)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"milp_{key}.npz")
    p, r, c = (getattr(inst, k).detach().cpu().double().numpy() for k in ("p", "r", "c"))
    if os.path.isfile(path):
        try:
            with np.load(path, allow_pickle=False) as cache:
                if json.loads(str(cache["settings"].item())) != settings:
                    raise ValueError("solver/data settings mismatch")
                result = {k: cache[k].copy() for k in ("obj", "optimal", "bound", "solution", "status")}
        except (KeyError, ValueError, OSError) as exc:
            raise RuntimeError(f"Invalid MILP reference cache {path}: {exc}") from exc
        log(f"[ref] loaded {inst.B} MILP references from {path}")
    else:
        objectives, optimal, bounds, solutions, statuses = [], [], [], [], []
        log(f"[ref] solving {inst.B} synthetic MILPs, time limit {time_limit}s per instance")
        for b in range(inst.B):
            solved = milp(-p[b], integrality=np.ones(inst.N), bounds=Bounds(0, 1),
                          constraints=LinearConstraint(r[b], -np.inf, c[b]),
                          options={"time_limit": float(time_limit), "mip_rel_gap": float(mip_rel_gap)})
            if solved.x is None:
                raise RuntimeError(f"No MILP validation incumbent for instance {b}: {solved.message}")
            x = np.rint(solved.x).astype(np.int8)
            if np.any(x < 0) or np.any(x > 1) or np.any(r[b] @ x > c[b] + 1e-5):
                raise RuntimeError(f"Infeasible MILP validation reference for instance {b}")
            objectives.append(float(p[b] @ x))
            optimal.append(solved.status == 0)
            bound = getattr(solved, "mip_dual_bound", None)
            bounds.append(float(-bound) if bound is not None else float("nan"))
            solutions.append(x)
            statuses.append(int(solved.status))
            if (b + 1) % 8 == 0 or b == inst.B - 1:
                log(f"[ref] solved {b + 1}/{inst.B}")
        result = {"obj": np.array(objectives), "optimal": np.array(optimal, dtype=bool),
                  "bound": np.array(bounds), "solution": np.stack(solutions),
                  "status": np.array(statuses)}
        np.savez_compressed(path, **result, settings=json.dumps(settings, sort_keys=True))
    if result["obj"].shape != (inst.B,) or not np.isfinite(result["obj"]).all() or np.any(result["obj"] <= 0):
        raise RuntimeError(f"Invalid MILP reference cache: {path}")
    if result["solution"].shape != (inst.B, inst.N):
        raise RuntimeError(f"Invalid MILP solution shape in {path}")
    if any(result[name].shape != (inst.B,) for name in ("optimal", "bound", "status")):
        raise RuntimeError(f"Invalid MILP metadata shape in {path}")
    x = result["solution"]
    if not np.all((x == 0) | (x == 1)) or np.any(np.einsum("bmn,bn->bm", r, x) > c + 1e-5):
        raise RuntimeError(f"Nonbinary or infeasible MILP reference cache: {path}")
    actual = np.einsum("bn,bn->b", p, x)
    if not np.allclose(actual, result["obj"], rtol=1e-9, atol=1e-5):
        raise RuntimeError(f"MILP reference objective disagrees with solution in {path}")
    finite_bounds = np.isfinite(result["bound"])
    if np.any(result["bound"][finite_bounds] + 1e-5 < result["obj"][finite_bounds]):
        raise RuntimeError(f"MILP reference bound is below incumbent in {path}")
    if not np.all((result["optimal"] == 0) | (result["optimal"] == 1)) or not np.array_equal(
            result["optimal"].astype(bool), result["status"] == 0):
        raise RuntimeError(f"MILP optimal/status metadata disagree in {path}")
    result.update({"path": path, "sha256": sha256_file(path), "settings": settings})
    log(f"[ref] proven optimal: {int(result['optimal'].sum())}/{inst.B}")
    return result
