"""Exact float64 LP-duality identities, used as fixed state-level criteria."""
from dataclasses import dataclass
import torch
from mkp_v6_cache import state_stamp
from mkp_v6_lp import lp_relaxation

NAMES = ('sur','reg','blk','fill','pair')
SIGNS = (1.,-1.,-1.,1.,1.)


@dataclass
class CertBasis:
    values: torch.Tensor       # B,N,5, float64, unscaled mean-profit units
    rank: torch.Tensor
    core: torch.Tensor
    scale: torch.Tensor
    reduced: torch.Tensor
    dual_value: torch.Tensor   # raw units, evaluated at exactly raw_prices
    raw_prices: torch.Tensor


@torch.no_grad()
def dual_identity(state, selected, prices):
    """(I1) for any raw-unit nonnegative dual, not necessarily LP optimal."""
    feasible=state.feasible
    reduced=state.profits-torch.einsum('bnm,bm->bn',state.weights,prices)
    dual=(prices*state.remaining).sum(-1)+(reduced.clamp_min(0)*feasible).sum(-1)
    slack=state.remaining-torch.einsum('bnm,bn->bm',state.weights,selected.double())
    parts=torch.stack(((prices*slack).sum(-1),(reduced.neg().clamp_min(0)*selected).sum(-1),
                       (reduced.clamp_min(0)*feasible*~selected).sum(-1)),-1)
    return dual,parts


@torch.no_grad()
def cert_basis(state, config, *, prices=None):
    """One chunked pair pass; optional raw prices support identity diagnostics."""
    lp=lp_relaxation(state,config)
    key=(state_stamp(state),id(lp),config.opportunity_chunk_size)
    if prices is None and getattr(state,'_cert_basis',None) is not None:
        if state._cert_basis[0]==key:return state._cert_basis[1]
    cache=prices is None
    prices=lp.raw_prices if prices is None else prices.double()
    feasible=state.feasible;p=state.profits;w=state.weights
    scale=lp.scale
    fill=torch.einsum('bnm,bm->bn',w,prices)/scale[:,None]
    reduced=p/scale[:,None]-fill
    sur=reduced.clamp_min(0)*feasible;reg=reduced.neg().clamp_min(0)*feasible
    blocked=torch.zeros_like(p);pair=torch.zeros_like(p)
    owners,items=feasible.nonzero(as_tuple=True)
    indices=torch.arange(p.shape[1],device=p.device)
    for rows,cols in zip(owners.split(config.opportunity_chunk_size),items.split(config.opportunity_chunk_size)):
        after=state.remaining[rows]-w[rows,cols]
        fits=(w[rows]<=after[:,None]).all(-1)
        other=feasible[rows] & (indices[None]!=cols[:,None])
        blocked[rows,cols]=(sur[rows]*other*~fits).sum(-1)
        pair[rows,cols]=(p[rows]/scale[rows,None]*other*fits).amax(-1)
    values=torch.stack((sur,reg,blocked,fill*feasible,pair),-1)
    ratio=(reduced.abs()/(p/scale[:,None]).clamp_min(config.eps)).masked_fill(~feasible,float('inf'))
    # Stable item-index tie breaking also makes the measured core reproducible.
    order=ratio.argsort(dim=-1,stable=True);positions=order.argsort(dim=-1,stable=True)
    count=feasible.sum(-1);rank=positions.double()/(count-1).clamp_min(1)[:,None]
    core=(positions<torch.minimum(count,torch.full_like(count,2*w.shape[-1]+10))[:,None]) & feasible
    dual=(prices*state.remaining).sum(-1)+(reduced.clamp_min(0)*feasible).sum(-1)*scale
    result=CertBasis(values,rank*feasible,core,scale,reduced,dual,prices)
    if cache:state._cert_basis=(key,result)
    return result
