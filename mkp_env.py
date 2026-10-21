"""
mkp_env.py
==========
Dynamic, batched Multidimensional Knapsack (MKP) environment.

    max  sum_j p_j x_j   s.t.  sum_j r_ij x_j <= c_i  (i = 1..M),  x_j in {0,1}

The API mirrors RL4CO's `RL4COEnvBase` (generator -> reset -> step -> get_reward) so it can be
ported into an RL4CO `MKPEnv` 1:1, but it is implemented in plain PyTorch to keep the
project dependency-light and fast (no TensorDict / Lightning needed).

Episode
-------
At every step the agent picks ONE not-yet-processed item.
  * fits    -> item is packed, remaining capacities shrink, profit is collected
  * no fit  -> item is *rejected* (discarded, capacities untouched) and counted as a
               constraint violation attempt (penalised in the reward).
The episode ends when no unprocessed item fits any more (or every item is processed).

Why no feasibility mask is required (see README):
capacities are monotonically non-increasing, so an item that does not fit now can never fit
later. Discarding it is therefore loss-free for the final objective, and the solution is
feasible by construction. The agent, however, must *perceive* feasibility from the state
(it is given continuous slack / load features, never a hard filter) to avoid wasted steps
and penalties. Only already-processed items are excluded from the choice set, which is a
structural "visited" mask, not a constraint mask.

Observation = (static item/resource/edge features, dynamic item/resource/edge features,
global features); all features are scale-free so the policy transfers across N, M and the
tightness ratio alpha.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, List

import torch

AVAILABLE, SELECTED, REJECTED = 0, 1, 2
_EPS = 1e-6


# ======================================================================================
# Instances
# ======================================================================================
@dataclass
class MKPInstances:
    """A batch of MKP instances with identical (N, M). Raw (unnormalised) values."""
    p: torch.Tensor        # (B, N) profits
    r: torch.Tensor        # (B, M, N) resource consumption
    c: torch.Tensor        # (B, M) capacities
    best: Optional[torch.Tensor] = None   # (B,) best-known objective (benchmarks only)

    @property
    def B(self) -> int:
        return self.p.shape[0]

    @property
    def N(self) -> int:
        return self.p.shape[1]

    @property
    def M(self) -> int:
        return self.r.shape[1]

    def to(self, device) -> "MKPInstances":
        return MKPInstances(self.p.to(device), self.r.to(device), self.c.to(device),
                            None if self.best is None else self.best.to(device))

    def repeat(self, k: int) -> "MKPInstances":
        """Repeat every instance k times (interleaved: b0,b0,..,b1,b1,..)."""
        rep = lambda t: None if t is None else t.repeat_interleave(k, dim=0)
        return MKPInstances(rep(self.p), rep(self.r), rep(self.c), rep(self.best))


def generate_instances(batch_size: int, n_items: int, n_dims: int,
                       tightness: Sequence[float] = (0.25, 0.5, 0.75),
                       weight_max: int = 1000, profit_noise: float = 500.0,
                       device="cpu", generator: Optional[torch.Generator] = None) -> MKPInstances:
    """Chu & Beasley (1998) generator (the distribution of Benchmarks_Chu.mat):
        r_ij ~ U{0..1000},  c_i = ceil(alpha * sum_j r_ij),  p_j = floor(sum_i r_ij / M + 500 q_j)
    with q_j ~ U(0,1) -> profits positively correlated with weights (hard instances).
    alpha is drawn per instance from `tightness`.
    """
    g = generator
    r = torch.randint(0, weight_max + 1, (batch_size, n_dims, n_items), generator=g).float()
    q = torch.rand(batch_size, n_items, generator=g)
    p = torch.floor(r.sum(1) / n_dims + profit_noise * q).clamp_min(1.0)
    t = torch.tensor(list(tightness), dtype=torch.float)
    alpha = t[torch.randint(0, len(t), (batch_size,), generator=g)]
    c = torch.ceil(alpha[:, None] * r.sum(2))
    return MKPInstances(p.to(device), r.to(device), c.to(device))


def instances_from_dicts(insts: List[dict], device="cpu") -> MKPInstances:
    """Stack benchmark dicts produced by `read_mat.read_mat` (same N, M required)."""
    p = torch.stack([torch.as_tensor(d["P"], dtype=torch.float).reshape(-1) for d in insts])
    r = torch.stack([torch.as_tensor(d["R"], dtype=torch.float).reshape(d["M"], d["N"]) for d in insts])
    c = torch.stack([torch.as_tensor(d["B"], dtype=torch.float).reshape(-1) for d in insts])
    best = torch.tensor([float(d["best"]) for d in insts], dtype=torch.float)
    return MKPInstances(p, r, c, best).to(device)


# ======================================================================================
# Lagrangian dual of the LP relaxation (batched, GPU friendly)
# ======================================================================================
@torch.no_grad()
def lp_dual(p: torch.Tensor, r: torch.Tensor, c: torch.Tensor,
            iters: int = 300, lr: float = 0.05):
    """Approximately solve the LP-relaxation dual of the MKP

        UB = min_{mu >= 0}  sum_i mu_i + sum_j max(0, v_j - a_j^T mu)

    in the normalised space a_ij = r_ij / c_i, v_j = p_j / mean(p), using smoothed
    projected Adam on all instances in parallel. Any mu >= 0 gives a valid upper bound,
    so the returned bound is always valid (only possibly slightly loose).

    Returns
        mu   (B, M) dual prices per unit of *capacity fraction* (normalised space)
        ub   (B,)   valid LP upper bound in raw profit units
    The dual prices are the classic surrogate multipliers used by Pirkul (1987) and
    Chu & Beasley (1998) for pseudo-utility ratios u_j = p_j / sum_i mu_i r_ij / c_i.
    """
    p = p.float(); r = r.float(); c = c.float()
    scale = p.mean(1, keepdim=True)                       # (B,1)
    v = p / scale                                         # (B,N)
    a = r / c.clamp_min(_EPS)[:, :, None]                 # (B,M,N)
    B, M, N = a.shape
    # init: price every resource so that an average item costs about its value
    mu = (v.mean(1, keepdim=True) / (M * a.mean(2).clamp_min(_EPS))).clamp_min(_EPS)
    m1 = torch.zeros_like(mu); m2 = torch.zeros_like(mu)
    b1, b2 = 0.9, 0.999
    best_mu = mu.clone()
    best_val = torch.full((B,), float("inf"), device=p.device)

    def dual_value(m):
        red = v - torch.einsum("bm,bmn->bn", m, a)
        return m.sum(1) + red.clamp_min(0).sum(1)

    for t in range(1, iters + 1):
        frac = t / iters
        tau = 0.05 * (1 - frac) + 1e-3 * frac            # smoothing temperature
        red = v - torch.einsum("bm,bmn->bn", mu, a)       # reduced costs (B,N)
        sig = torch.sigmoid(red / tau)
        grad = 1.0 - torch.einsum("bn,bmn->bm", sig, a)
        m1 = b1 * m1 + (1 - b1) * grad
        m2 = b2 * m2 + (1 - b2) * grad * grad
        step = (m1 / (1 - b1 ** t)) / ((m2 / (1 - b2 ** t)).sqrt() + 1e-8)
        cur_lr = lr * (0.5 * (1 + math.cos(math.pi * frac)) + 0.01)
        mu = (mu - cur_lr * step * mu.mean(1, keepdim=True).clamp_min(1e-3)).clamp_min(0.0)
        val = dual_value(mu)
        better = val < best_val
        best_val = torch.where(better, val, best_val)
        best_mu = torch.where(better[:, None], mu, best_mu)
    return best_mu, best_val * scale.squeeze(1)


# ======================================================================================
# State
# ======================================================================================
@dataclass
class MKPState:
    inst: MKPInstances         # raw instance data (repeated per rollout)
    mu: torch.Tensor           # (B, M) static LP dual prices (normalised space)
    ub: torch.Tensor           # (B,) LP upper bound
    c_rem: torch.Tensor        # (B, M) remaining capacity
    status: torch.Tensor       # (B, N) int8: 0 available, 1 selected, 2 rejected
    profit: torch.Tensor       # (B,)
    n_reject: torch.Tensor     # (B,)
    done: torch.Tensor         # (B,) bool
    fits: torch.Tensor         # (B, N) bool: available & fits remaining capacity
    t: int = 0

    @property
    def available(self) -> torch.Tensor:
        return self.status == AVAILABLE

    @property
    def solution(self) -> torch.Tensor:
        return (self.status == SELECTED).float()


class MKPEnv:
    """Batched dynamic MKP environment (RL4CO-like interface)."""

    name = "mkp"

    def __init__(self, env_cfg=None):
        self.cfg = env_cfg
        self.dual_iters = getattr(env_cfg, "dual_iters", 300)
        self.dual_lr = getattr(env_cfg, "dual_lr", 0.05)
        self.reject_penalty = getattr(env_cfg, "reject_penalty", 0.05)

    # ------------------------------------------------------------------ generator
    def generate(self, batch_size, n_items, n_dims, device="cpu", generator=None) -> MKPInstances:
        c = self.cfg
        return generate_instances(batch_size, n_items, n_dims,
                                  tightness=getattr(c, "train_tightness", (0.25, 0.5, 0.75)),
                                  weight_max=getattr(c, "weight_max", 1000),
                                  profit_noise=getattr(c, "profit_noise", 500.0),
                                  device=device, generator=generator)

    # ------------------------------------------------------------------ reset / step
    @torch.no_grad()
    def precompute(self, inst: MKPInstances):
        """Instance-level (static) quantities, computed once before repeating rollouts."""
        mu, ub = lp_dual(inst.p, inst.r, inst.c, self.dual_iters, self.dual_lr)
        return mu, ub

    @torch.no_grad()
    def reset(self, inst: MKPInstances, mu: torch.Tensor, ub: torch.Tensor) -> MKPState:
        B, N = inst.p.shape
        dev = inst.p.device
        status = torch.zeros(B, N, dtype=torch.int8, device=dev)
        st = MKPState(inst=inst, mu=mu, ub=ub, c_rem=inst.c.clone(), status=status,
                      profit=torch.zeros(B, device=dev), n_reject=torch.zeros(B, device=dev),
                      done=torch.zeros(B, dtype=torch.bool, device=dev),
                      fits=torch.zeros(B, N, dtype=torch.bool, device=dev))
        self._refresh(st)
        return st

    @staticmethod
    def _refresh(st: MKPState) -> None:
        fit_all = (st.inst.r <= st.c_rem[:, :, None] + _EPS).all(1)       # (B,N)
        st.fits = fit_all & st.available
        st.done = ~st.fits.any(1)

    @torch.no_grad()
    def step(self, st: MKPState, action: torch.Tensor) -> MKPState:
        """action: (B,) item indices. Finished rows are ignored (no-op)."""
        B = action.shape[0]
        ar = torch.arange(B, device=action.device)
        active = ~st.done
        r_a = st.inst.r[ar, :, action]                                   # (B,M)
        was_avail = st.status[ar, action] == AVAILABLE
        ok = (r_a <= st.c_rem + _EPS).all(1) & was_avail & active
        bad = (~ok) & active
        st.c_rem = st.c_rem - r_a * ok[:, None].float()
        st.profit = st.profit + st.inst.p[ar, action] * ok.float()
        st.n_reject = st.n_reject + bad.float()
        new_status = torch.where(ok, torch.full_like(action, SELECTED),
                                 torch.full_like(action, REJECTED)).to(torch.int8)
        cur = st.status[ar, action]
        st.status[ar, action] = torch.where(active & was_avail, new_status, cur)
        st.t += 1
        self._refresh(st)
        return st

    def get_reward(self, st: MKPState) -> torch.Tensor:
        """Normalised reward: profit / LP-bound  -  penalty * (#rejected) / N."""
        N = st.inst.N
        return st.profit / st.ub.clamp_min(_EPS) - self.reject_penalty * st.n_reject / N

    # ------------------------------------------------------------------ observations
    @staticmethod
    def _zscore(x: torch.Tensor, valid: Optional[torch.Tensor] = None, dim: int = -1):
        if valid is None:
            mean = x.mean(dim, keepdim=True)
            std = x.std(dim, keepdim=True, unbiased=False)
        else:
            w = valid.float()
            cnt = w.sum(dim, keepdim=True).clamp_min(1.0)
            mean = (x * w).sum(dim, keepdim=True) / cnt
            std = (((x - mean) ** 2) * w).sum(dim, keepdim=True).div(cnt).sqrt()
        return ((x - mean) / (std + 1e-6)).clamp(-5, 5)

    @torch.no_grad()
    def static_features(self, inst: MKPInstances, mu: torch.Tensor, ub: torch.Tensor):
        """Instance-level features (computed once).

        Returns item (B,N,F_is), resource (B,M,F_rs), edge (B,M,N,F_es) features.
        """
        p, r, c = inst.p, inst.r, inst.c
        B, M, N = r.shape
        a = r / c.clamp_min(_EPS)[:, :, None]                 # fraction of capacity (B,M,N)
        rel = r / r.mean(2, keepdim=True).clamp_min(_EPS)     # weight relative to dim mean
        v = p / p.mean(1, keepdim=True)
        price = torch.einsum("bm,bmn->bn", mu, a)              # dual cost of each item (norm. space)
        ratio = torch.log(v.clamp_min(_EPS)) - torch.log(price.clamp_min(_EPS))
        eff = torch.log(v.clamp_min(_EPS)) - torch.log(a.sum(1).clamp_min(_EPS))
        red = (v - price) / (v.abs() + _EPS)                   # relative reduced cost
        rank = ratio.argsort(1).argsort(1).float() / max(N - 1, 1)
        item = torch.stack([
            p / p.max(1, keepdim=True).values,
            self._zscore(v),
            rel.mean(1), rel.max(1).values, rel.min(1).values, rel.std(1, unbiased=False),
            self._zscore(ratio), self._zscore(eff), red.clamp(-3, 1),
            torch.sigmoid(10.0 * red), rank,
        ], dim=-1)

        alpha = c / r.sum(2).clamp_min(_EPS)                   # tightness per dim
        mu_share = mu / (mu.sum(1, keepdim=True) + _EPS)
        pc = p - p.mean(1, keepdim=True)
        rc = r - r.mean(2, keepdim=True)
        corr = (rc * pc[:, None, :]).sum(2) / (rc.norm(dim=2) * pc.norm(dim=1, keepdim=True) + _EPS)
        res = torch.stack([
            alpha, mu_share, mu_share * M, rel.std(2, unbiased=False), corr,
            torch.full_like(alpha, math.log(M) / math.log(30.0)),
        ], dim=-1)

        price_share = mu[:, :, None] * a / (v[:, None, :] + _EPS)   # how much of item value goes to dim i
        edge = torch.stack([rel.clamp(0, 5) / 2.0, price_share.clamp(0, 5)], dim=-1)
        return item, res, edge

    @torch.no_grad()
    def dynamic_features(self, st: MKPState):
        """Step-dependent features describing the *current* residual problem.

        Returns item (B,N,F_id), resource (B,M,F_rd), edge (B,M,N,F_ed), global (B,F_g).
        """
        inst = st.inst
        p, r, c = inst.p, inst.r, inst.c
        B, M, N = r.shape
        avail = st.available
        availf = avail.float()
        crem = st.c_rem.clamp_min(0.0)
        load = r / (crem[:, :, None] + 1.0)                    # (B,M,N) consumption of remaining cap
        over = (r > crem[:, :, None] + _EPS).float()           # per-dim overflow indicator
        slack_after = ((crem[:, :, None] - r) / c[:, :, None].clamp_min(_EPS))  # (B,M,N)
        min_slack = slack_after.min(1).values.clamp(-1, 1)
        max_load = load.max(1).values.clamp(0, 3)
        fits = (over.sum(1) == 0).float()
        dyn_ratio = torch.log(p.clamp_min(_EPS)) - torch.log(load.sum(1).clamp_min(_EPS))
        item = torch.stack([
            availf, (st.status == SELECTED).float(), (st.status == REJECTED).float(),
            fits * availf, max_load, min_slack, over.mean(1),
            self._zscore(dyn_ratio, avail) * availf,
        ], dim=-1)

        n_av = availf.sum(1, keepdim=True).clamp_min(1.0)       # (B,1)
        mean_r_av = (r * availf[:, None, :]).sum(2) / n_av       # (B,M)
        demand = (r * availf[:, None, :]).sum(2) / (crem + 1.0)  # (B,M)
        fit_frac = ((1 - over) * availf[:, None, :]).sum(2) / n_av
        rem_frac = crem / c.clamp_min(_EPS)
        res = torch.stack([
            rem_frac,
            torch.log1p(crem / (mean_r_av + 1.0)) / math.log(N + 1.0),
            torch.log1p(demand) / math.log(N + 1.0),
            fit_frac,
            (rem_frac - rem_frac.mean(1, keepdim=True)),
        ], dim=-1)

        edge = torch.stack([load.clamp(0, 3) / 3.0, over], dim=-1)

        sel = (st.status == SELECTED).float().sum(1)
        glob = torch.stack([
            1.0 - n_av.squeeze(1) / N,
            sel / N,
            st.profit / st.ub.clamp_min(_EPS),
            rem_frac.mean(1), rem_frac.min(1).values,
            st.fits.float().sum(1) / n_av.squeeze(1),
        ], dim=-1)
        return item, res, edge, glob


# feature sizes (kept in one place so the model can be built without data)
F_ITEM_STATIC, F_RES_STATIC, F_EDGE_STATIC = 11, 6, 2
F_ITEM_DYN, F_RES_DYN, F_EDGE_DYN, F_GLOBAL = 8, 5, 2, 6


@torch.no_grad()
def greedy_dual_ratio(inst: MKPInstances, mu: torch.Tensor) -> torch.Tensor:
    """Reference heuristic: add items in decreasing static pseudo-utility ratio
    p_j / sum_i mu_i r_ij / c_i while they fit (Pirkul 1987 / Chu & Beasley 1998).
    Returns objective values (B,)."""
    p, r, c = inst.p, inst.r, inst.c
    a = r / c[:, :, None]
    price = torch.einsum("bm,bmn->bn", mu, a).clamp_min(_EPS)
    order = (p / price).argsort(1, descending=True)
    crem = c.clone()
    obj = torch.zeros_like(p[:, 0])
    ar = torch.arange(p.shape[0], device=p.device)
    for k in range(p.shape[1]):
        j = order[:, k]
        rj = r[ar, :, j]
        ok = (rj <= crem + _EPS).all(1)
        crem = crem - rj * ok[:, None].float()
        obj = obj + p[ar, j] * ok.float()
    return obj
