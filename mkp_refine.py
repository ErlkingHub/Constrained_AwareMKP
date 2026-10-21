"""
mkp_refine.py
=============
Optional post-hoc refinement of policy solutions with classic, interpretable MKP moves,
fully batched on the GPU. It is *not* used during training; the benchmark reports gaps both
with and without it.

Moves (all keep feasibility):
  * ADD / fill   : add fitting items in decreasing pseudo-utility ratio
                   (the ADD phase of Chu & Beasley's (1998) repair operator).
  * SWAP 1-1     : exhaustive best-improvement exchange of one packed item for one unpacked
                   item (O(N^2 M) per iteration, evaluated in parallel).
  * DROP-REFILL  : drop one packed item and greedily refill (covers 1-k exchanges).
  * ILS          : iterated local search; randomly drop a few low-ratio items, refill with a
                   noisy ratio order, re-optimise, keep if not worse (Lourenco et al., 2003).

The learned policy supplies diverse, high-quality starting points (multi-start), the
moves only polish them.
"""
from __future__ import annotations

import torch

_EPS = 1e-6


def _ratio(p, r, c, mu):
    a = r / c[:, :, None]
    price = torch.einsum("bm,bmn->bn", mu, a)
    return p / price.clamp_min(_EPS)


@torch.no_grad()
def greedy_fill(x, p, r, c, score):
    """Add fitting unpacked items in decreasing `score` until none fits. x: (B,N) float."""
    x = x.clone()
    B = x.shape[0]
    ar = torch.arange(B, device=x.device)
    for _ in range(x.shape[1]):
        slack = c - torch.einsum("bmn,bn->bm", r, x)
        cand = (x < 0.5) & (r <= slack[:, :, None] + _EPS).all(1)
        if not cand.any():
            break
        j = torch.where(cand, score, torch.full_like(score, -float("inf"))).argmax(1)
        has = cand.any(1)
        x[ar[has], j[has]] = 1.0
    return x


@torch.no_grad()
def swap_11(x, p, r, c):
    """One best-improvement 1-1 exchange per row. Returns (x_new, improved mask)."""
    B, M, N = r.shape
    slack = c - torch.einsum("bmn,bn->bm", r, x)             # (B,M)
    inn = x > 0.5
    feas = torch.ones(B, N, N, dtype=torch.bool, device=x.device)   # [b, a(in), o(out)]
    for i in range(M):
        ri = r[:, i, :]
        feas &= (ri[:, None, :] - ri[:, :, None]) <= slack[:, i, None, None] + _EPS
    delta = p[:, None, :] - p[:, :, None]
    valid = feas & inn[:, :, None] & (~inn)[:, None, :] & (delta > 0.5)
    delta = torch.where(valid, delta, torch.full_like(delta, -1.0))
    best = delta.view(B, -1).max(1)
    improved = best.values > 0
    a = best.indices // N
    o = best.indices % N
    x = x.clone()
    rows = torch.nonzero(improved).squeeze(1)
    x[rows, a[rows]] = 0.0
    x[rows, o[rows]] = 1.0
    return x, improved


@torch.no_grad()
def drop_refill(x, p, r, c, score, drop_k=32):
    """For the `drop_k` packed items with the lowest score, try: drop it, refill greedily.
    Apply the best strictly improving result per row."""
    B, M, N = r.shape
    inn = x > 0.5
    k = int(min(drop_k, N))
    s = torch.where(inn, score, torch.full_like(score, float("inf")))
    cand = s.argsort(1)[:, :k]                                # (B,k) lowest-ratio packed items
    cand_valid = torch.gather(inn, 1, cand)
    xx = x[:, None, :].expand(B, k, N).clone()
    xx.scatter_(2, cand[:, :, None], 0.0)
    flat = xx.reshape(B * k, N)
    rep = lambda t: t.repeat_interleave(k, 0)
    # the dropped item may not be re-added (otherwise refill would just undo the drop)
    sc = rep(score).clone()
    sc.scatter_(1, cand.reshape(-1, 1), -float("inf"))
    filled = _fill_excluding(flat, rep(p), rep(r), rep(c), sc)
    obj_new = (filled * rep(p)).sum(1).view(B, k)
    obj_new = torch.where(cand_valid, obj_new, torch.full_like(obj_new, -1.0))
    obj_old = (x * p).sum(1)
    best = obj_new.max(1)
    improved = best.values > obj_old + 0.5
    out = x.clone()
    rows = torch.nonzero(improved).squeeze(1)
    if rows.numel():
        out[rows] = filled.view(B, k, N)[rows, best.indices[rows]]
    return out, improved


@torch.no_grad()
def _fill_excluding(x, p, r, c, score):
    """greedy_fill where score == -inf marks forbidden items."""
    x = x.clone()
    B = x.shape[0]
    ar = torch.arange(B, device=x.device)
    allowed = torch.isfinite(score)
    for _ in range(x.shape[1]):
        slack = c - torch.einsum("bmn,bn->bm", r, x)
        cand = (x < 0.5) & allowed & (r <= slack[:, :, None] + _EPS).all(1)
        if not cand.any():
            break
        j = torch.where(cand, score, torch.full_like(score, -float("inf"))).argmax(1)
        has = cand.any(1)
        x[ar[has], j[has]] = 1.0
    return x


@torch.no_grad()
def local_search(x, p, r, c, mu, max_iters=200, drop_k=32):
    """Fill -> repeat {swap 1-1 ; drop-refill} until no row improves."""
    score = _ratio(p, r, c, mu)
    x = greedy_fill(x, p, r, c, score)
    active = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    for _ in range(max_iters):
        idx = torch.nonzero(active).squeeze(1)
        if idx.numel() == 0:
            break
        xs, imp1 = swap_11(x[idx], p[idx], r[idx], c[idx])
        xs = greedy_fill(xs, p[idx], r[idx], c[idx], score[idx])
        xs2, imp2 = drop_refill(xs, p[idx], r[idx], c[idx], score[idx], drop_k)
        x[idx] = xs2
        active[idx] = imp1 | imp2
    return x


@torch.no_grad()
def iterated_local_search(x, p, r, c, mu, rounds=100, perturb_max=4, max_iters=200,
                          drop_k=32, generator=None):
    """ILS over a batch of solutions (each row independent). Returns best x per row."""
    B, M, N = r.shape
    score = _ratio(p, r, c, mu)
    x = local_search(x, p, r, c, mu, max_iters, drop_k)
    best_x = x.clone()
    best_obj = (x * p).sum(1)
    cur = x.clone()
    cur_obj = best_obj.clone()
    logscore = torch.log(score.clamp_min(_EPS))
    for _ in range(rounds):
        # perturbation: drop k packed items, biased to low pseudo-utility ratio
        k = int(torch.randint(2, max(perturb_max, 2) + 1, (1,), generator=generator).item())
        inn = cur > 0.5
        logits = torch.where(inn, -2.0 * (logscore - logscore.mean(1, keepdim=True)),
                             torch.full_like(logscore, -float("inf")))
        n_in = inn.sum(1)
        kk = int(min(k, max(int(n_in.min().item()), 0)))
        y = cur.clone()
        if kk > 0:
            g = torch.rand(logits.shape, generator=generator, device="cpu").to(logits.device)
            gumbel = -torch.log(-torch.log(g.clamp(1e-9, 1 - 1e-9)))
            drop = (logits + gumbel).topk(kk, dim=1).indices
            y.scatter_(1, drop, 0.0)
        noise = torch.rand(score.shape, generator=generator, device="cpu").to(score.device)
        y = greedy_fill(y, p, r, c, logscore + 0.3 * noise)
        y = local_search(y, p, r, c, mu, max_iters, drop_k)
        obj = (y * p).sum(1)
        accept = obj >= cur_obj
        cur = torch.where(accept[:, None], y, cur)
        cur_obj = torch.where(accept, obj, cur_obj)
        better = obj > best_obj
        best_x = torch.where(better[:, None], y, best_x)
        best_obj = torch.where(better, obj, best_obj)
    return best_x, best_obj


def is_feasible(x, r, c):
    return (torch.einsum("bmn,bn->bm", r, x) <= c + _EPS).all(1)
