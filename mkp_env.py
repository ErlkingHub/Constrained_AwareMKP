"""Exact feasible MKP construction states; all physical accounting is float64.

The executor can only add a currently feasible item. No repair, removal, learned
feasibility predictor, or benchmark reference value participates in transitions.
"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mkp_data import LPInfo, MKPInstance


class MKPState:
    def __init__(self, instance: "MKPInstance", lp: "LPInfo", device="cpu"):
        self.instance = instance
        self.lp = lp
        self.device = torch.device(device)
        self.p = torch.as_tensor(instance.p, dtype=torch.float64, device=self.device)
        self.w = torch.as_tensor(instance.w, dtype=torch.float64, device=self.device)
        self.c = torch.as_tensor(instance.c, dtype=torch.float64, device=self.device)
        self.dual = torch.as_tensor(lp.prices, dtype=torch.float64, device=self.device)
        self.n, self.m = self.w.shape
        if self.p.shape != (self.n,) or self.c.shape != (self.m,):
            raise ValueError("Expected p[N], w[N,M], c[M]")
        if not all(torch.isfinite(x).all() for x in (self.p, self.w, self.c, self.dual)):
            raise ValueError("Non-finite MKP coefficients or LP prices")
        if any((x < 0).any() for x in (self.p, self.w, self.c, self.dual)):
            raise ValueError("MKP coefficients and LP prices must be nonnegative")
        self.chosen = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.used = torch.zeros_like(self.c)
        self.remaining = self.c.clone()
        self.selected: list[int] = []
        self.objective = 0.0
        self._cache: dict = {}
        self._refresh()

    def _refresh(self):
        # Compare against original capacities, without an acceptance tolerance.
        self.feasible = (~self.chosen) & ((self.w + self.used) <= self.c).all(dim=1)
        self.done = not bool(self.feasible.any())

    def clone(self) -> "MKPState":
        """Snapshot for PPO: static tensors and immutable descriptor cache shared."""
        state = copy.copy(self)
        state.chosen = self.chosen.clone()
        state.used = self.used.clone()
        state.remaining = self.remaining.clone()
        state.feasible = self.feasible.clone()
        state.selected = list(self.selected)
        # Step replaces the dictionary; cached tensors are never modified.
        state._cache = dict(self._cache)
        return state

    def step(self, action: int) -> float:
        action = int(action)
        if action < 0 or action >= self.n or not bool(self.feasible[action]):
            raise ValueError(f"Item {action} is not a feasible unselected action")
        self.chosen[action] = True
        self.selected.append(action)
        self.used = self.used + self.w[action]
        self.remaining = self.c - self.used
        reward = float(self.p[action])
        self.objective += reward
        self._cache = {}
        self._refresh()
        return reward

    def potential(self) -> float:
        """Residual dual upper bound / root bound, with terminal potential zero.

        Any y >= 0 yields U(s)=y*c_rem+sum_F max(0,p_i-w_i*y), including
        the fixed root LP dual. Taking min(U, sum_F p) remains an upper bound.
        This computes no new LP and is NOT the exact residual LP optimum.
        """
        if self.done:
            return 0.0
        fp, fw = self.p[self.feasible], self.w[self.feasible]
        upper = self.dual @ self.remaining + (fp - fw @ self.dual).clamp_min(0).sum()
        upper = torch.minimum(upper, fp.sum()).clamp_min(0)
        return float(upper) / max(float(self.lp.upper_bound), 1e-12)

    def pair_descriptors(self, chunk_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact blocked-profit fraction and best compatible partner profit.

        Only distinct currently feasible items are compared. Complexity is
        O(F^2 M), peak workspace O(chunk_size F M); no pair tensor is stored in
        rollout snapshots. The compatibility descriptor is an observable pair
        feature, not a search or a change to the construction action.
        """
        if "pair" in self._cache:
            return self._cache["pair"]
        indices = self.feasible.nonzero(as_tuple=True)[0]
        fw, fp = self.w[indices], self.p[indices]
        count = len(indices)
        blocked = torch.zeros(self.n, dtype=torch.float64, device=self.device)
        partner = torch.zeros_like(blocked)
        total_profit = fp.sum().clamp_min(1e-12)
        profit_scale = self.p.max().clamp_min(1e-12)
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            # Use the same accumulated-capacity predicate as feasibility.
            fits = ((fw[start:stop, None] + fw[None] + self.used) <= self.c).all(-1)
            rows = torch.arange(stop - start, device=self.device)
            cols = torch.arange(start, stop, device=self.device)
            distinct = torch.ones_like(fits)
            distinct[rows, cols] = False
            blocked[indices[start:stop]] = ((~fits & distinct) * fp).sum(-1) / total_profit
            candidates = torch.where(fits & distinct, fp[None], torch.zeros_like(fp)[None])
            partner[indices[start:stop]] = candidates.max(-1).values / profit_scale
        self._cache["pair"] = (blocked, partner)
        return blocked, partner

    def solution_metrics(self) -> dict:
        """Metrics computed in raw units; zero-capacity utilization is zero."""
        feasible = bool((self.used <= self.c).all()) and len(set(self.selected)) == len(self.selected)
        utilization = torch.where(self.c > 0, self.used / self.c.clamp_min(1e-12), 0)
        return {
            "objective": self.objective,
            "selected": list(self.selected),
            "steps": len(self.selected),
            "feasible": feasible,
            "maximal": self.done,
            "utilization": utilization.detach().cpu().tolist(),
            "utilization_mean": float(utilization.mean()),
            "utilization_min": float(utilization.min()),
            "utilization_max": float(utilization.max()),
        }
