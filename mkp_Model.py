"""
mkp_Model.py
============
Hierarchical, constraint-aware policy for the Multidimensional Knapsack Problem.

    MKP instance
      -> Dynamic MKP environment (mkp_env.MKPEnv)
      -> State Encoder        : item <-> resource bipartite graph Transformer (static, once)
                                + dynamic state injection at every step
      -> High-Level Strategy Controller : dual-axis Transformer over the *current* state that
                                outputs an interpretable strategy  (lambda_t, w_t, kappa_t, delta)
      -> Low-Level Weak Decoder : parameter-free scorer combining a handful of classic MKP
                                primitives with the strategy, softmax / argmax
      -> select item, env updates remaining capacities, repeat until nothing fits.

Multi-head attention captures two kinds of competition (the "dual axis"):
  * item  <-> item      self-attention   : profit competition among the remaining items
  * item  <-> resource  cross-attention  : resource competition; attention scores are biased by
                                           the (static + per-step) consumption matrix, so every
                                           head sees how much of each *remaining* capacity an
                                           item would consume (MatNet-style mixed scores)
  * resource <-> resource self-attention : competition / coupling between resource dimensions

Strategy (high level, per step t):
  lambda_t in R^M_+  learned resource prices = LP-dual prices * exp(+-price_log_range)
  w_t in simplex^K   mixture over K interpretable primitives
  kappa_t > 0        decisiveness (inverse temperature)
  delta_j            bounded per-item residual correction (|delta| <= residual_max)

Weak decoder (low level, no parameters):
  score_j = kappa_t * sum_k w_tk * z(phi_k(j)) + delta_j,     z = z-score over remaining items
  phi_profit      = log p_j
  phi_efficiency  = log p_j - log sum_i r_ij / c_i                      (static efficiency)
  phi_price_ratio = log p_j - log sum_i lambda_ti r_ij / c_rem_i        (dynamic surrogate ratio)
  phi_slack       = min_i (c_rem_i - r_ij) / c_i                        (bottleneck slack, <0 if infeasible)

References (ideas adopted, all re-implemented here):
  [1] Chu & Beasley (1998) J. Heuristics 4:63-86 - instance generator and LP-dual pseudo-utility
      ratio -> training distribution, `price_ratio` primitive and LP-dual features.
  [2] Pirkul (1987) Naval Res. Logist. 34:161-172 - surrogate-dual multipliers for MKP greedy
      -> resource prices as the controller's main "strategy" variable.
  [3] Senju & Toyoda (1968) Manage. Sci. 14:B53-B68 - effective gradient: weight resources by
      their *remaining* capacity -> c_rem in the price-ratio primitive.
  [4] Gasse et al. (2019) NeurIPS - variable/constraint bipartite graph of a MILP -> item/resource
      tokens of the state encoder.
  [5] Kwon et al. (2021) MatNet, NeurIPS - matrix (edge) values injected into cross-attention
      scores between row/column nodes -> edge-biased item<->resource attention.
  [6] Kool et al. (2019) ICLR - Transformer encoder + REINFORCE for CO.
  [7] Kwon et al. (2020) POMO, NeurIPS - shared multi-sample baseline (used in mkp_Trainer).
"""
from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from mkp_env import (MKPEnv, MKPInstances, MKPState,
                     F_ITEM_STATIC, F_RES_STATIC, F_EDGE_STATIC,
                     F_ITEM_DYN, F_RES_DYN, F_EDGE_DYN, F_GLOBAL)

NEG = -1e4          # finite "minus infinity" for logits (keeps entropy / grads NaN-free)
_EPS = 1e-6
PRIMITIVES = ("profit", "efficiency", "price_ratio", "slack")


# ======================================================================================
# Attention building blocks
# ======================================================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d: int, h: int, dropout: float = 0.0):
        super().__init__()
        assert d % h == 0, "embed_dim must be divisible by num_heads"
        self.h, self.dk, self.dropout = h, d // h, dropout
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)

    def forward(self, q_in, kv_in, attn_mask=None):
        B, Lq, d = q_in.shape
        Lk = kv_in.shape[1]
        q = self.q(q_in).view(B, Lq, self.h, self.dk).transpose(1, 2)
        k, v = self.kv(kv_in).view(B, Lk, 2, self.h, self.dk).permute(2, 0, 3, 1, 4)
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                             dropout_p=self.dropout if self.training else 0.0)
        return self.o(out.transpose(1, 2).reshape(B, Lq, d))


class FeedForward(nn.Module):
    def __init__(self, d, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d * mult), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d * mult, d))

    def forward(self, x):
        return self.net(x)


class EdgeBias(nn.Module):
    """Maps item-resource edge features (B,M,N,F) to per-head attention biases for both
    directions: resource<-item (B,H,M,N) and item<-resource (B,H,N,M)."""

    def __init__(self, f_edge, hidden, h):
        super().__init__()
        self.h = h
        self.mlp = nn.Sequential(nn.Linear(f_edge, hidden), nn.ReLU(), nn.Linear(hidden, 2 * h))

    def forward(self, edge):
        e = self.mlp(edge)                                   # (B,M,N,2H)
        b_res = e[..., :self.h].permute(0, 3, 1, 2)          # (B,H,M,N)
        b_item = e[..., self.h:].permute(0, 3, 2, 1)         # (B,H,N,M)
        return b_res.contiguous(), b_item.contiguous()


class DualAxisBlock(nn.Module):
    """One layer of the bipartite item/resource Transformer (pre-LN).

    1. item self-attention          -> profit competition among (remaining) items
    2. item <- resource cross-attn  -> each item reads the resource dimensions it stresses
    3. resource <- item cross-attn  -> each resource aggregates the demand competing for it
    4. resource self-attention      -> coupling / competition between resource dimensions
    5. position-wise FFNs
    Cross-attention scores are biased by edge features (consumption matrix).
    """

    def __init__(self, d, h, ff_mult, dropout, f_edge, edge_hidden):
        super().__init__()
        self.edge = EdgeBias(f_edge, edge_hidden, h)
        self.ln_x1, self.item_self = nn.LayerNorm(d), MultiHeadAttention(d, h, dropout)
        self.ln_x2, self.ln_y2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.item_from_res = MultiHeadAttention(d, h, dropout)
        self.ln_y3, self.ln_x3 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.res_from_item = MultiHeadAttention(d, h, dropout)
        self.ln_y4, self.res_self = nn.LayerNorm(d), MultiHeadAttention(d, h, dropout)
        self.ln_x5, self.ff_x = nn.LayerNorm(d), FeedForward(d, ff_mult, dropout)
        self.ln_y5, self.ff_y = nn.LayerNorm(d), FeedForward(d, ff_mult, dropout)

    def forward(self, x, y, edge, keep: Optional[torch.Tensor] = None):
        b_res, b_item = self.edge(edge)
        self_mask = None
        if keep is not None:                                  # hide processed items as keys
            km = keep[:, None, None, :]
            self_mask = km
            b_res = b_res.masked_fill(~km, float("-inf"))
        hx = self.ln_x1(x)
        x = x + self.item_self(hx, hx, self_mask)
        x = x + self.item_from_res(self.ln_x2(x), self.ln_y2(y), b_item)
        y = y + self.res_from_item(self.ln_y3(y), self.ln_x3(x), b_res)
        hy = self.ln_y4(y)
        y = y + self.res_self(hy, hy)
        x = x + self.ff_x(self.ln_x5(x))
        y = y + self.ff_y(self.ln_y5(y))
        return x, y


# ======================================================================================
# State encoder
# ======================================================================================
class StateEncoder(nn.Module):
    """Bipartite item/resource graph encoder.

    * encode_static : run once per instance on scale-free static features + LP-dual features.
    * fuse_dynamic  : at each step, inject the dynamic residual-problem features
                      (remaining capacities, per-item load/slack/overflow, status, progress).
    """

    def __init__(self, cfg):
        super().__init__()
        d = cfg.embed_dim
        self.item_in = nn.Sequential(nn.Linear(F_ITEM_STATIC, d), nn.GELU(), nn.Linear(d, d))
        self.res_in = nn.Sequential(nn.Linear(F_RES_STATIC, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([
            DualAxisBlock(d, cfg.num_heads, cfg.ff_mult, cfg.dropout, F_EDGE_STATIC, cfg.edge_hidden)
            for _ in range(cfg.encoder_layers)])
        self.ln_x, self.ln_y = nn.LayerNorm(d), nn.LayerNorm(d)
        self.dyn_item = nn.Sequential(nn.Linear(F_ITEM_DYN, d), nn.GELU(), nn.Linear(d, d))
        self.dyn_res = nn.Sequential(nn.Linear(F_RES_DYN, d), nn.GELU(), nn.Linear(d, d))
        self.dyn_glob = nn.Sequential(nn.Linear(F_GLOBAL, d), nn.GELU(), nn.Linear(d, d))

    def encode_static(self, item_s, res_s, edge_s):
        x, y = self.item_in(item_s), self.res_in(res_s)
        for blk in self.blocks:
            x, y = blk(x, y, edge_s)
        return self.ln_x(x), self.ln_y(y)

    def fuse_dynamic(self, x_s, y_s, item_d, res_d, glob):
        g = self.dyn_glob(glob)
        x = x_s + self.dyn_item(item_d) + g[:, None, :]
        y = y_s + self.dyn_res(res_d) + g[:, None, :]
        return x, y, g


# ======================================================================================
# High-level strategy controller
# ======================================================================================
class StrategyController(nn.Module):
    """Dual-axis Transformer over the current state + interpretable strategy heads."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.embed_dim
        self.cfg = cfg
        self.prims = list(cfg.primitives)
        for p in self.prims:
            assert p in PRIMITIVES, f"unknown primitive '{p}'"
        K = len(self.prims)
        self.blocks = nn.ModuleList([
            DualAxisBlock(d, cfg.num_heads, cfg.ff_mult, cfg.dropout,
                          F_EDGE_STATIC + F_EDGE_DYN, cfg.edge_hidden)
            for _ in range(cfg.controller_layers)])
        self.ln_x, self.ln_y = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ctx = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, d))
        self.price_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.prim_head = nn.Linear(d, K)
        self.kappa_head = nn.Linear(d, 1)
        self.res_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self._init_heads()

    def _init_heads(self):
        cfg = self.cfg
        # zero-init -> the untrained policy is exactly the (stochastic) classic heuristic
        # defined by the initial primitive mixture: a strong, interpretable warm start.
        for head in (self.price_head[-1], self.res_head[-1]):
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
        nn.init.zeros_(self.prim_head.weight)
        with torch.no_grad():
            self.prim_head.bias.zero_()
            if cfg.init_primitive in self.prims:
                self.prim_head.bias[self.prims.index(cfg.init_primitive)] = cfg.init_primitive_logit
        nn.init.zeros_(self.kappa_head.weight)
        target = max(cfg.kappa_init - cfg.kappa_min, 1e-3)
        with torch.no_grad():
            self.kappa_head.bias.fill_(math.log(math.expm1(target)))   # softplus^-1

    def forward(self, x, y, g, edge, avail):
        done = ~avail.any(1)
        keep = avail | done[:, None]                          # avoid fully-masked rows
        for blk in self.blocks:
            x, y = blk(x, y, edge, keep)
        x, y = self.ln_x(x), self.ln_y(y)
        w = avail.to(x.dtype)[:, :, None]
        cnt = w.sum(1).clamp_min(1.0)
        x_mean = (x * w).sum(1) / cnt
        x_max = x.masked_fill(~keep[:, :, None], -1e4).max(1).values
        ctx = self.ctx(torch.cat([x_mean, x_max, y.mean(1), g], dim=-1))       # (B,d)
        cfg = self.cfg
        M, N = y.shape[1], x.shape[1]
        price = cfg.price_log_range * torch.tanh(
            self.price_head(torch.cat([y, ctx[:, None, :].expand(-1, M, -1)], -1)).squeeze(-1))
        prim_w = torch.softmax(self.prim_head(ctx).float(), dim=-1)
        kappa = cfg.kappa_min + F.softplus(self.kappa_head(ctx).float()).squeeze(-1)
        resid = cfg.residual_max * torch.tanh(
            self.res_head(torch.cat([x, ctx[:, None, :].expand(-1, N, -1)], -1)).squeeze(-1).float())
        return {"log_price": price.float(), "prim_w": prim_w, "kappa": kappa, "residual": resid}


# ======================================================================================
# Low-level weak decoder (parameter-free, interpretable)
# ======================================================================================
class WeakDecoder:
    def __init__(self, model_cfg, env_cfg):
        self.prims = list(model_cfg.primitives)
        self.price_floor = model_cfg.price_floor
        self.mode = getattr(env_cfg, "feasibility_mode", "perceive")
        self.beta = getattr(env_cfg, "soft_mask_beta", 5.0)

    @staticmethod
    def _z(x, valid):
        w = valid.float()
        cnt = w.sum(1, keepdim=True).clamp_min(1.0)
        mean = (x * w).sum(1, keepdim=True) / cnt
        var = (((x - mean) ** 2) * w).sum(1, keepdim=True).div(cnt)
        std = (var + 1e-8).sqrt()                          # +eps: finite grad when var == 0
        return ((x - mean) / (std + 1e-6)).clamp(-5, 5)

    def primitives(self, st: MKPState, lam: torch.Tensor) -> Dict[str, torch.Tensor]:
        p, r, c = st.inst.p, st.inst.r, st.inst.c
        crem = st.c_rem.clamp_min(0.0)
        logp = torch.log(p.clamp_min(_EPS))
        out = {}
        if "profit" in self.prims:
            out["profit"] = logp
        if "efficiency" in self.prims:
            out["efficiency"] = logp - torch.log((r / c[:, :, None]).sum(1).clamp_min(_EPS))
        if "price_ratio" in self.prims:
            cost = torch.einsum("bm,bmn->bn", lam, r / (crem[:, :, None] + 1.0))
            out["price_ratio"] = logp - torch.log(cost.clamp_min(1e-9))
        if "slack" in self.prims:
            out["slack"] = ((crem[:, :, None] - r) / c[:, :, None]).min(1).values.clamp(-1, 1)
        return out

    def __call__(self, st: MKPState, strat: Dict[str, torch.Tensor]):
        avail = st.available
        mu = st.mu
        base = mu + self.price_floor * mu.mean(1, keepdim=True) + 1e-6
        lam = base * torch.exp(strat["log_price"])
        prim = self.primitives(st, lam)
        z = torch.stack([self._z(prim[k], avail) for k in self.prims], dim=-1)   # (B,N,K)
        score = strat["kappa"][:, None] * (z * strat["prim_w"][:, None, :]).sum(-1) + strat["residual"]
        if self.mode == "soft_mask":                           # ablation: soft penalty on overflow
            over = (st.inst.r - st.c_rem[:, :, None]).clamp_min(0) / (st.inst.r + 1.0)
            viol = over.max(1).values                          # in [0,1), >0 iff infeasible
            score = score - self.beta * ((viol > 0).float() + viol)
        elif self.mode == "hard_mask":
            score = score.masked_fill(~st.fits, NEG)
        score = score.masked_fill(~avail, NEG)                # structural: processed items
        score = torch.where(st.done[:, None], torch.zeros_like(score), score)
        return score, lam


# ======================================================================================
# Full policy
# ======================================================================================
class ConstrainedMKPPolicy(nn.Module):
    def __init__(self, cfg):
        """cfg: full Config (uses cfg.model and cfg.env)."""
        super().__init__()
        self.mcfg, self.ecfg = cfg.model, cfg.env
        self.encoder = StateEncoder(cfg.model)
        self.controller = StrategyController(cfg.model)
        self.decoder = WeakDecoder(cfg.model, cfg.env)

    # one decision step: dynamic fusion + controller (checkpoint-able, no randomness)
    def _step_core(self, x_s, y_s, edge_s, item_d, res_d, edge_d, glob, avail):
        x, y, g = self.encoder.fuse_dynamic(x_s, y_s, item_d, res_d, glob)
        edge = torch.cat([edge_s, edge_d], dim=-1)     # static + per-step consumption features
        strat = self.controller(x, y, g, edge, avail)
        return strat["log_price"], strat["prim_w"], strat["kappa"], strat["residual"]

    def forward(self, env: MKPEnv, inst: MKPInstances, num_samples: int = 1,
                decode: str = "sample", temperature: float = 1.0, amp: bool = False,
                pre: Optional[tuple] = None, return_trace: bool = False):
        """Roll out `num_samples` trajectories per instance.

        Returns dict with reward, profit, n_reject, log_likelihood, entropy (all (B*K,)),
        solution (B*K, N), ub, steps, and mean strategy statistics.
        """
        dev = inst.p.device
        amp_ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
                   if (amp and dev.type == "cuda") else nullcontext())
        mu, ub = pre if pre is not None else env.precompute(inst)
        item_s, res_s, edge_s = env.static_features(inst, mu, ub)
        with amp_ctx:
            x_s, y_s = self.encoder.encode_static(item_s, res_s, edge_s)
        K = num_samples
        if K > 1:
            rep = lambda t: t.repeat_interleave(K, 0)
            x_s, y_s, edge_s = rep(x_s), rep(y_s), rep(edge_s)
            inst, mu, ub = inst.repeat(K), rep(mu), rep(ub)
        st = env.reset(inst, mu, ub)
        BK = inst.B
        logp_sum = torch.zeros(BK, device=dev)
        ent_sum = torch.zeros(BK, device=dev)
        n_dec = torch.zeros(BK, device=dev)
        stats = {"kappa": 0.0, "prim_w": 0.0, "abs_residual": 0.0, "log_price_abs": 0.0}
        n_stat = 0
        trace = [] if return_trace else None
        use_ckpt = self.training and self.mcfg.grad_checkpoint and torch.is_grad_enabled()

        while not bool(st.done.all()):
            item_d, res_d, edge_d, glob = env.dynamic_features(st)
            avail = st.available
            with amp_ctx:
                if use_ckpt:
                    out = checkpoint(self._step_core, x_s, y_s, edge_s, item_d, res_d, edge_d, glob,
                                     avail, use_reentrant=False)
                else:
                    out = self._step_core(x_s, y_s, edge_s, item_d, res_d, edge_d, glob, avail)
            strat = dict(zip(("log_price", "prim_w", "kappa", "residual"), out))
            logits, lam = self.decoder(st, strat)
            logits = logits.float()
            if decode == "greedy":
                logp_all = torch.log_softmax(logits, -1)
                action = logits.argmax(-1)
            else:
                logp_all = torch.log_softmax(logits / temperature, -1)
                action = torch.multinomial(logp_all.exp(), 1).squeeze(-1)
            active = (~st.done).float()
            logp_sum = logp_sum + logp_all.gather(1, action[:, None]).squeeze(1) * active
            ent = -(logp_all.exp() * logp_all).sum(-1)
            ent_sum = ent_sum + ent * active
            n_dec = n_dec + active
            with torch.no_grad():                               # strategy statistics (no sync)
                n_act = active.sum().clamp_min(1.0)
                stats["kappa"] = stats["kappa"] + (strat["kappa"] * active).sum() / n_act
                stats["prim_w"] = stats["prim_w"] + (strat["prim_w"] * active[:, None]).sum(0) / n_act
                af = avail.float()
                stats["abs_residual"] = stats["abs_residual"] + (strat["residual"].abs() * af).sum() / af.sum().clamp_min(1.0)
                stats["log_price_abs"] = stats["log_price_abs"] + strat["log_price"].abs().mean()
                n_stat += 1
                if return_trace:
                    trace.append({"action": action.detach().cpu(), "kappa": strat["kappa"].detach().cpu(),
                                  "prim_w": strat["prim_w"].detach().cpu(), "lambda": lam.detach().cpu(),
                                  "fits_before": st.fits.gather(1, action[:, None]).squeeze(1).cpu()})
            st = env.step(st, action)

        n_stat = max(n_stat, 1)
        out = {
            "reward": env.get_reward(st),
            "profit": st.profit,
            "n_reject": st.n_reject,
            "log_likelihood": logp_sum,
            "entropy": ent_sum / n_dec.clamp_min(1.0),
            "solution": st.solution,
            "ub": st.ub,
            "steps": st.t,
            "kappa": float(stats["kappa"] / n_stat),
            "prim_w": (stats["prim_w"] / n_stat).tolist() if torch.is_tensor(stats["prim_w"]) else None,
            "abs_residual": float(stats["abs_residual"] / n_stat),
            "log_price_abs": float(stats["log_price_abs"] / n_stat),
            "primitives": list(self.mcfg.primitives),
        }
        if return_trace:
            out["trace"] = trace
        return out


def build_model(cfg) -> ConstrainedMKPPolicy:
    return ConstrainedMKPPolicy(cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
