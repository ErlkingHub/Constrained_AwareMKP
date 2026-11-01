"""Validated MKP data and root LP bounds; training generation never reads MAT files.

All arrays use the canonical orientation ``p[N], w[N,M], c[M]`` in float64.
The benchmark's ``best_known`` is metadata, never an input to a policy or LP.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linprog


DISTRIBUTIONS = ("uncorrelated", "weakly_correlated", "strongly_correlated", "subset_sum")


@dataclass
class MKPInstance:
    id: str
    p: np.ndarray
    w: np.ndarray
    c: np.ndarray
    distribution: str = "unknown"
    capacity_ratio: float = float("nan")
    best_known: float | None = None

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.p = np.array(self.p, dtype=np.float64, order="C", copy=True)
        self.w = np.array(self.w, dtype=np.float64, order="C", copy=True)
        self.c = np.array(self.c, dtype=np.float64, order="C", copy=True)
        if self.p.ndim != 1 or self.c.ndim != 1 or self.w.ndim != 2:
            raise ValueError("Expected p[N], w[N,M], c[M]; flatten/transpose at the data boundary")
        if not len(self.p) or not len(self.c) or self.w.shape != (len(self.p), len(self.c)):
            raise ValueError(f"Invalid MKP shapes: {self.p.shape}, {self.w.shape}, {self.c.shape}")
        for name in ("p", "w", "c"):
            value = getattr(self, name)
            if not np.all(np.isfinite(value)) or np.any(value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.best_known is not None:
            self.best_known = float(self.best_known)
            if not np.isfinite(self.best_known) or self.best_known < 0:
                raise ValueError("best_known must be finite and nonnegative when provided")
        if not np.isnan(self.capacity_ratio) and (
            not np.isfinite(self.capacity_ratio) or self.capacity_ratio < 0
        ):
            raise ValueError("capacity_ratio must be nonnegative or NaN when unknown")

    @property
    def N(self) -> int:
        return self.p.size

    @property
    def M(self) -> int:
        return self.c.size

    @property
    def n(self) -> int:
        return self.N

    @property
    def m(self) -> int:
        return self.M

    def fingerprint(self) -> str:
        """Stable digest of ID, shape and numeric problem data (not reference labels)."""
        digest = hashlib.sha256()
        digest.update(json.dumps([self.id, self.N, self.M], ensure_ascii=False).encode("utf-8"))
        for value in (self.p, self.w, self.c):
            digest.update(np.asarray(value, dtype="<f8", order="C").tobytes(order="C"))
        return digest.hexdigest()


@dataclass(frozen=True)
class LPInfo:
    prices: np.ndarray
    upper_bound: float
    solve_seconds: float


def solve_root_lp(instance: MKPInstance) -> LPInfo:
    """Solve max p*x with W^T*x <= c and 0 <= x <= 1, once at the root.

    For any y >= 0, ``y*c + sum(max(p-W*y, 0))`` is a valid box-LP
    dual upper bound. We reconstruct this bound from projected solver prices,
    rather than treating a possibly optimistic floating-point primal value as
    an upper bound. A small outward rounding allowance protects comparisons.
    """
    started = time.perf_counter()
    result = linprog(-instance.p, A_ub=instance.w.T, b_ub=instance.c,
                     bounds=(0.0, 1.0), method="highs")
    if not result.success:
        raise RuntimeError(f"Root LP failed for {instance.id}: {result.message}")
    prices = np.maximum(-np.asarray(result.ineqlin.marginals, dtype=np.float64), 0.0)
    if prices.shape != (instance.M,) or not np.all(np.isfinite(prices)):
        raise RuntimeError(f"Root LP returned invalid resource prices for {instance.id}")
    dual = float(prices @ instance.c + np.maximum(instance.p - instance.w @ prices, 0.0).sum())
    bound = min(float(instance.p.sum()), dual)
    if bound > 0:
        bound = float(np.nextafter(bound + 1e-10 * max(1.0, bound), np.inf))
    return LPInfo(prices=prices, upper_bound=bound, solve_seconds=time.perf_counter() - started)


def generate_instance(
    rng: np.random.Generator,
    n: int,
    m: int,
    distribution: str,
    capacity_ratio: float,
    instance_id: str,
    *,
    coefficient_max: int = 1000,
    correlation_noise: float = 100.0,
    correlation_offset: float = 100.0,
) -> MKPInstance:
    """Generate an independent synthetic instance; no Chu entries are used.

    w_ij ~ U_integer[1,coefficient_max]; c_j=floor(a*sum_i w_ij).
    Profit families: U_integer[1,coefficient_max], mean_j(w)+uniform noise,
    mean_j(w)+offset, or sum_j(w), respectively. Profits are rounded and >= 1.
    These definitions are controlled test families, not exact reproductions of
    the Chu benchmark's original generator or claims of distribution identity.
    """
    if isinstance(n, bool) or isinstance(m, bool) or int(n) != n or int(m) != m or n < 1 or m < 1:
        raise ValueError("n and m must be positive integers")
    if distribution not in DISTRIBUTIONS:
        raise ValueError(f"Unknown distribution {distribution!r}; expected one of {DISTRIBUTIONS}")
    if not np.isfinite(capacity_ratio) or not 0 < capacity_ratio <= 1:
        raise ValueError("Synthetic capacity_ratio must lie in (0, 1]")
    if isinstance(coefficient_max, bool) or int(coefficient_max) != coefficient_max or coefficient_max < 1:
        raise ValueError("coefficient_max must be a positive integer")
    if not np.isfinite(correlation_noise) or correlation_noise < 0:
        raise ValueError("correlation_noise must be finite and nonnegative")
    if not np.isfinite(correlation_offset) or correlation_offset < 0:
        raise ValueError("correlation_offset must be finite and nonnegative")
    w = rng.integers(1, int(coefficient_max) + 1, size=(int(n), int(m))).astype(np.float64)
    if distribution == "uncorrelated":
        p = rng.integers(1, int(coefficient_max) + 1, size=int(n)).astype(np.float64)
    elif distribution == "weakly_correlated":
        p = np.maximum(1.0, np.rint(w.mean(axis=1) + rng.uniform(-correlation_noise, correlation_noise, size=int(n))))
    elif distribution == "strongly_correlated":
        p = np.maximum(1.0, np.rint(w.mean(axis=1) + correlation_offset))
    else:
        p = w.sum(axis=1)
    return MKPInstance(instance_id, p, w, np.floor(capacity_ratio * w.sum(axis=0)),
                       distribution=distribution, capacity_ratio=float(capacity_ratio))


def load_chu(path: str | Path) -> list[MKPInstance]:
    """Adapt the project's existing read_mat.py reader without filtering entries."""
    from read_mat import read_mat
    from mkp_utils import resolve_path

    filename = resolve_path(path)
    if not filename.is_file():
        raise FileNotFoundError(f"Chu benchmark file does not exist: {filename}")
    records = read_mat(str(filename))
    if not records:
        raise ValueError(f"read_mat returned no instances for {filename}; check the MAT schema")
    instances = []
    seen: set[str] = set()
    for record in records:
        n, m = int(record["N"]), int(record["M"])
        p = np.asarray(record["P"], dtype=np.float64).reshape(-1)
        c = np.asarray(record["B"], dtype=np.float64).reshape(-1)
        raw = np.asarray(record["R"], dtype=np.float64)
        # The supplied Chu reader exposes resource-by-item R. Prefer that
        # orientation in the square case, whose dimensions are ambiguous.
        if raw.shape == (m, n):
            w = raw.T
        elif raw.shape == (n, m):
            w = raw
        else:
            raise ValueError(f"Chu record {record['id']}: unexpected R shape {raw.shape}, M={m}, N={n}")
        if p.shape != (n,) or c.shape != (m,):
            raise ValueError(f"Chu record {record['id']}: P/B shapes disagree with declared N/M")
        name = f"Chu_{int(record['id']):03d}"
        if name in seen:
            raise ValueError(f"Duplicate benchmark instance ID: {name}")
        seen.add(name)
        sums = w.sum(axis=0)
        active = sums > 0
        ratio = float(np.mean(c[active] / sums[active])) if active.any() else float("nan")
        instance = MKPInstance(name, p, w, c, distribution="chu", capacity_ratio=ratio,
                               best_known=float(record["best"]))
        if instance.best_known <= 0:
            raise ValueError(f"Chu record {name}: positive best-known value required for percentage gaps")
        instances.append(instance)
    return instances
