"""Fixed, transparent item-level bases for the parameter-free MKP executor."""
from __future__ import annotations

import torch

from mkp_env import MKPState


@torch.no_grad()
def blocked_mass(st: MKPState, chunk_size: int = 64) -> torch.Tensor:
    """Profit share of other currently fitting items blocked by selecting each item.

    Only ``st.fits`` items contribute as candidates or blocked alternatives; unavailable
    and infeasible candidates receive zero. The comparison is strict, as in the method
    definition: item k is blocked by i if some r_jk > c_rem_j - r_ji.

    Work is O(B*N²*M). Candidate chunks and resource-wise accumulation keep the largest
    pairwise temporary at (B, chunk_size, N), rather than (B, N, N, M).
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    profits = st.inst.p.float()
    fits = st.fits
    weighted_profits = profits * fits
    total = weighted_profits.sum(1, keepdim=True)
    denominator = torch.where(total > 0, total, torch.ones_like(total))
    batch, n_items = profits.shape
    out = torch.zeros_like(profits)
    for start in range(0, n_items, chunk_size):
        end = min(start + chunk_size, n_items)
        blocked = torch.zeros(batch, end - start, n_items, dtype=torch.bool, device=profits.device)
        for dim in range(st.inst.M):
            remaining = st.c_rem[:, dim, None] - st.inst.r[:, dim, start:end]
            blocked |= st.inst.r[:, dim, None, :] > remaining[:, :, None]
        blocked &= fits[:, None, :]
        blocked &= fits[:, start:end, None]
        blocked[:, torch.arange(end - start, device=profits.device),
                torch.arange(start, end, device=profits.device)] = False
        blocked_profit = torch.bmm(blocked.float(), weighted_profits[:, :, None]).squeeze(-1)
        out[:, start:end] = blocked_profit / denominator
    return out
