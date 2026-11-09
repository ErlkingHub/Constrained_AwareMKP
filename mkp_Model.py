"""Item-resource relation encoder, resource-level theta controller and fixed executor.

T1 interface: M centered log-price corrections u and scalar alpha; optional T2 gamma.
The encoder has item tokens, but no learned item-level output enters the score.
Static encoding is cached once; the original per-step representation is preserved.
Theta-PG samples latent controls at declared decision points and executes greedy.
"""
from __future__ import annotations

import math
import time
from contextlib import nullcontext
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from mkp_env import (MKPEnv, MKPInstances, MKPState,
                     F_ITEM_STATIC, F_RES_STATIC, F_EDGE_STATIC,
                     F_ITEM_DYN, F_RES_DYN, F_EDGE_DYN, F_GLOBAL)

NEG = float("-inf")
_EPS = 1e-6


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
        lp = cfg.executor == "v6" and cfg.ctrl_residual_lp_features
        self.dyn_item = nn.Sequential(nn.Linear(F_ITEM_DYN + 4*lp, d), nn.GELU(), nn.Linear(d, d))
        self.dyn_res = nn.Sequential(nn.Linear(F_RES_DYN + 3*lp, d), nn.GELU(), nn.Linear(d, d))
        self.dyn_glob = nn.Sequential(nn.Linear(F_GLOBAL + 2*lp, d), nn.GELU(), nn.Linear(d, d))

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
def theta_from_pre(v, z, u_max, g=None, h=None):
    """Transparent activation of resource and scalar latent actions."""
    u = u_max * torch.tanh(v.float())
    theta = {"u": u - u.mean(1, keepdim=True),
             "alpha": 0.5 + 2.0 * torch.sigmoid(z.float())}
    if g is not None:
        theta["gamma"] = F.softplus(g.float())
    if h is not None:
        theta["eta"] = F.softplus(h.float())
    return theta


def theta_pre_log_prob(means, draws, sigmas):
    """Detached-action score function; tensor scales include their -log(sigma).

    The fixed Python-float path retains T1-P's constant-free density verbatim.
    Tensor scales omit only Normal's constant -log(2*pi)/2 per active dimension.
    Zero scales describe deterministic dimensions and contribute no score term.
    """
    logp = torch.zeros_like(means["alpha_pre"])
    for name, mean in means.items():
        sigma = sigmas[name]
        if torch.is_tensor(sigma):
            sigma = sigma.to(device=mean.device, dtype=mean.dtype)
            active = sigma > 0
            safe = torch.where(active, sigma, torch.ones_like(sigma))
            term = (-0.5 * ((draws[name].detach() - mean) / safe).square()
                    - safe.log()).masked_fill(~active, 0.0)
        else:
            if float(sigma) == 0:
                continue
            term = -0.5 * ((draws[name].detach() - mean) / float(sigma)).square()
        logp = logp + (term.sum(1) if term.ndim == 2 else term)
    return logp


def sample_theta_pre(means, sigmas, batch_size, num_samples, antithetic=False,
                     generator=None, include_center=False):
    """Detached Gaussian draws; zero scales are deterministic and are not sampled."""
    if antithetic and num_samples % 2:
        raise ValueError("Antithetic theta sampling requires even num_samples")
    if antithetic and include_center:
        raise ValueError("Center insertion is incompatible with antithetic pairs")
    draws = {}
    for name, mean in means.items():
        sigma = sigmas[name]
        if torch.is_tensor(sigma):
            sigma = sigma.to(device=mean.device, dtype=mean.dtype)
            if not bool(torch.isfinite(sigma).all()) or bool((sigma < 0).any()):
                raise ValueError("Theta noise scales must be finite and nonnegative")
            enabled = bool((sigma > 0).any())
        else:
            sigma = float(sigma)
            if not math.isfinite(sigma) or sigma < 0:
                raise ValueError("Theta noise scales must be finite and nonnegative")
            enabled = sigma > 0
        if enabled:
            shape = (batch_size, num_samples // 2 if antithetic else num_samples, *mean.shape[1:])
            noise_device = mean.device if generator is None else generator.device
            eps = torch.randn(shape, generator=generator, device=noise_device).to(mean.device)
            if antithetic:
                eps = torch.stack((eps, -eps), dim=2)
            draw = (mean + sigma * eps.reshape_as(mean)).detach()
        else:
            draw = mean.detach().clone()
        if include_center:
            draw = draw.clone()
            draw[::num_samples] = mean.detach()[::num_samples]
        draws[name] = draw
    logp = theta_pre_log_prob(means, draws, sigmas)
    if include_center:
        mask = torch.ones_like(logp)
        mask[::num_samples] = 0
        logp = logp * mask
    return draws, logp


class StrategyController(nn.Module):
    """Dual-axis Transformer over the current state + interpretable strategy heads."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.embed_dim
        self.cfg = cfg
        groups = list(cfg.theta_groups)
        if len(groups) != len(set(groups)) or "price" not in groups:
            raise ValueError("theta_groups must contain price exactly once")
        if set(groups) - {"price", "alpha", "block", "lp"}:
            raise ValueError("Supported theta groups are price, alpha, block, lp")
        self.scalar_names = [name for name in ("alpha", "block", "lp") if name in groups]
        self.blocks = nn.ModuleList([
            DualAxisBlock(d, cfg.num_heads, cfg.ff_mult, cfg.dropout,
                          F_EDGE_STATIC + F_EDGE_DYN, cfg.edge_hidden)
            for _ in range(cfg.controller_layers)])
        self.ln_x, self.ln_y = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ctx = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, d))
        self.price_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        if cfg.theta_family == "v6":
            from mkp_v6_model import V6Heads
            self.v6_heads = V6Heads(cfg)
            nn.init.zeros_(self.price_head[-1].weight)
            nn.init.zeros_(self.price_head[-1].bias)
            return
        if cfg.theta_sigma_head:
            if set(groups) != {"price", "alpha"}:
                raise ValueError("SMS proposal heads require the F0 price/alpha interface")
            self.sigma_u_head = nn.Linear(2*d, 1)
        scalar_count = len(self.scalar_names) + int(cfg.theta_sigma_head)
        self.scalar_head = nn.Linear(d, scalar_count) if scalar_count else None
        self._init_heads()

    def _init_heads(self):
        if self.cfg.theta_sigma_head:
            nn.init.zeros_(self.sigma_u_head.weight)
            nn.init.zeros_(self.sigma_u_head.bias)
        if self.cfg.untrained_random_heads:
            if self.cfg.theta_sigma_head:
                with torch.no_grad():
                    self.scalar_head.weight[-1].zero_()
                    self.scalar_head.bias[-1].zero_()
            return  # Preserve PyTorch's default random mean-head initialization for this control.
        nn.init.zeros_(self.price_head[-1].weight)
        nn.init.zeros_(self.price_head[-1].bias)
        if self.scalar_head is not None:
            nn.init.zeros_(self.scalar_head.weight)
            nn.init.zeros_(self.scalar_head.bias)
            with torch.no_grad():
                for index, name in enumerate(self.scalar_names):
                    if name == "alpha":
                        probability = (self.cfg.alpha_init - .5) / 2
                        self.scalar_head.bias[index] = math.log(probability / (1-probability))
                    else:
                        initial = self.cfg.block_gamma_init if name == "block" else self.cfg.lp_eta_init
                        self.scalar_head.bias[index] = math.log(math.expm1(initial))

    def representation(self, x, y, g, edge, avail):
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
        return x,y,ctx

    def forward(self, x, y, g, edge, avail):
        x,y,ctx = self.representation(x,y,g,edge,avail)
        cfg = self.cfg
        M = y.shape[1]
        if cfg.theta_family == "v6":
            with torch.autocast(device_type=y.device.type, enabled=False):
                features = torch.cat([y.float(),ctx.float()[:,None].expand(-1,M,-1)],-1)
                logits = self.price_head(features).squeeze(-1)
                return self.v6_heads(logits,y.float(),ctx.float())
        v = self.price_head(torch.cat([y, ctx[:, None, :].expand(-1, M, -1)], -1)).squeeze(-1).float()
        if cfg.theta_sigma_head:
            # New proposal heads keep their registered zero-init center exact
            # under AMP. The legacy head-disabled arithmetic stays unchanged.
            with torch.autocast(device_type=ctx.device.type, enabled=False):
                scalars = self.scalar_head(ctx.float())
        else:
            scalars = self.scalar_head(ctx).float() if self.scalar_head is not None else None
        z = (scalars[:, self.scalar_names.index("alpha")] if "alpha" in self.scalar_names else
             torch.full_like(v[:, 0], math.log(((cfg.alpha_init-.5)/2)/(1-(cfg.alpha_init-.5)/2))))
        g_pre = scalars[:, self.scalar_names.index("block")] if "block" in self.scalar_names else None
        h_pre = scalars[:, self.scalar_names.index("lp")] if "lp" in self.scalar_names else None
        theta = theta_from_pre(v, z, cfg.u_max, g_pre, h_pre)
        theta.update(u_pre=v, alpha_pre=z)
        if g_pre is not None:
            theta["gamma_pre"] = g_pre
        if h_pre is not None:
            theta["eta_pre"] = h_pre
        if cfg.theta_sigma_head:
            features = torch.cat([y, ctx[:, None, :].expand(-1, M, -1)], -1)
            with torch.autocast(device_type=ctx.device.type, enabled=False):
                su = self.sigma_u_head(features.float()).squeeze(-1)
            sa = scalars[:, -1]
            theta["sigma_u"] = (cfg.theta_sigma_u0 * torch.exp(1.5*torch.tanh(su))
                                if "u" in cfg.theta_sigma_dims else torch.zeros_like(su))
            theta["sigma_alpha"] = (cfg.theta_sigma_alpha0 * torch.exp(1.5*torch.tanh(sa))
                                    if "a" in cfg.theta_sigma_dims else torch.zeros_like(sa))
        return theta


# ======================================================================================
# Low-level weak decoder (parameter-free, interpretable)
# ======================================================================================
class WeakDecoder:
    """Transparent scorer with optional fixed blocking basis; no trainable parameters."""

    def __init__(self, model_cfg, env_cfg):
        self.normalize = model_cfg.score_normalization
        self.price_anchor = model_cfg.price_anchor
        self.lp_source = model_cfg.lp_source
        self.use_lp = "lp" in model_cfg.theta_groups
        self.price_floor = model_cfg.price_floor
        self.use_block = "block" in model_cfg.theta_groups
        self.block_chunk_size = model_cfg.block_chunk_size
        self.mode = getattr(env_cfg, "feasibility_mode", "hard_mask")
        self.beta = getattr(env_cfg, "soft_mask_beta", 5.0)
        if self.mode not in ("hard_mask", "perceive", "soft_mask"):
            raise ValueError(f"Unknown feasibility mode: {self.mode}")

    @staticmethod
    def moments(score, fits):
        count = fits.sum(1, keepdim=True).clamp_min(1)
        masked = score.masked_fill(~fits, 0.0)
        mean = masked.sum(1, keepdim=True) / count
        centered = (score - mean).masked_fill(~fits, 0.0)
        # Vector norm has a defined zero gradient at zero (unlike sqrt(var)).
        std = torch.linalg.vector_norm(centered, dim=1, keepdim=True) / count.sqrt()
        return mean, std

    def raw_scores(self, st: MKPState, theta: Dict[str, torch.Tensor]):
        anchor = st.mu if self.price_anchor == "root" else st.mu_res
        if anchor is None:
            raise ValueError("Residual LP anchor is not initialized")
        base = anchor + self.price_floor * anchor.mean(1, keepdim=True) + 1e-6
        lam = base * theta["u"].exp()
        cost = torch.einsum("bm,bmn->bn", lam, st.inst.r / (st.c_rem[:, :, None] + 1.0))
        p_norm = st.inst.p / st.inst.p.mean(1, keepdim=True).clamp_min(_EPS)
        score = theta["alpha"][:, None] * p_norm.clamp_min(_EPS).log() - cost.clamp_min(1e-9).log()
        if self.use_block:
            from mkp_basis import blocked_mass
            score = score - theta["gamma"][:, None] * blocked_mass(st, self.block_chunk_size)
        if self.use_lp:
            x_lp = st.x_lp_root if self.lp_source == "root" else st.x_lp
            if x_lp is None:
                raise ValueError("LP basis is not initialized")
            score = score + theta["eta"][:, None] * torch.log(x_lp + .01)
        return score, lam

    def __call__(self, st: MKPState, theta: Dict[str, torch.Tensor], return_raw_std=False):
        score, lam = self.raw_scores(st, theta)
        mean, std = self.moments(score.float(), st.fits)
        if self.normalize:
            score = (score.float() - mean) / (std + 1e-6)
        if self.mode == "soft_mask":
            over = (st.inst.r - st.c_rem[:, :, None]).clamp_min(0) / (st.inst.r + 1.0)
            viol = over.max(1).values
            score = score - self.beta * ((viol > 0).float() + viol)
        valid = st.fits if self.mode == "hard_mask" else st.available
        score = score.masked_fill(~valid, NEG)
        # Finished rows are no-ops; avoid a fully masked softmax in a mixed batch.
        logits = torch.where(st.done[:, None], torch.zeros_like(score), score)
        return (logits, lam, std) if return_raw_std else (logits, lam)


class ConstrainedMKPPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mcfg, self.ecfg = cfg.model, cfg.env
        if cfg.model.executor == "v6":
            from mkp_v6_model import V6Policy
            self.v6_engine = V6Policy(cfg)
            return
        if cfg.model.executor != "g0g1" or cfg.model.theta_family != "g0g1":
            raise ValueError("executor/theta_family must both select g0g1 or v6")
        if cfg.model.price_anchor not in ("root", "residual") or cfg.model.lp_source not in ("root", "residual"):
            raise ValueError("price_anchor and lp_source must be root or residual")
        if isinstance(cfg.model.lp_interval, bool) or not isinstance(cfg.model.lp_interval, int) or cfg.model.lp_interval < 1:
            raise ValueError("lp_interval must be a positive integer")
        if not .5 < cfg.model.alpha_init < 2.5 or not math.isfinite(cfg.model.lp_eta_init) or cfg.model.lp_eta_init <= 0:
            raise ValueError("alpha_init must be strictly inside (.5,2.5) and lp_eta_init positive")
        if cfg.model.theta_update not in ("step", "static"):
            raise ValueError("theta_update must be step or static")
        if cfg.model.temperature_mode not in ("learned", "fixed"):
            raise ValueError("temperature_mode must be learned or fixed")
        if not math.isfinite(cfg.model.temperature_init) or cfg.model.temperature_init <= 0:
            raise ValueError("temperature_init must be positive and finite")
        if not math.isfinite(cfg.model.u_max) or cfg.model.u_max < 0:
            raise ValueError("u_max must be finite and nonnegative")
        if isinstance(cfg.model.theta_interval, bool) or not isinstance(cfg.model.theta_interval, int) or cfg.model.theta_interval < 1:
            raise ValueError("theta_interval must be a positive integer")
        if not math.isfinite(cfg.model.block_gamma_init) or cfg.model.block_gamma_init <= 0 or cfg.model.block_chunk_size < 1:
            raise ValueError("Block gamma initialization and chunk size must be positive")
        if cfg.model.theta_sigma_dims not in ("u", "a", "ua"):
            raise ValueError("theta_sigma_dims must be u, a or ua")
        if any(not math.isfinite(v) or v <= 0 for v in (cfg.model.theta_sigma_u0,cfg.model.theta_sigma_alpha0)):
            raise ValueError("SMS base scales must be positive and finite")
        if cfg.model.theta_sigma_head and cfg.model.theta_update != "static":
            raise ValueError("T4 learned proposals use static controls")
        self.theta_keys = ("u", "alpha", "u_pre", "alpha_pre")
        if cfg.model.theta_sigma_head:
            self.theta_keys += ("sigma_u", "sigma_alpha")
        if "block" in cfg.model.theta_groups:
            self.theta_keys += ("gamma", "gamma_pre")
        if "lp" in cfg.model.theta_groups:
            self.theta_keys += ("eta", "eta_pre")
        self.encoder = StateEncoder(cfg.model)
        self.controller = StrategyController(cfg.model)
        self.decoder = WeakDecoder(cfg.model, cfg.env)
        initial = torch.tensor(math.log(cfg.model.temperature_init), dtype=torch.float32)
        if cfg.model.temperature_mode == "learned":
            self.log_temperature = nn.Parameter(initial)
        else:
            self.register_buffer("log_temperature", initial)

    @property
    def temperature(self):
        return self.log_temperature.exp().clamp_min(1e-6)

    def fixed_theta(self, st: MKPState):
        theta = {"u": torch.zeros_like(st.mu), "alpha": torch.full_like(st.profit, self.mcfg.alpha_init)}
        if "block" in self.mcfg.theta_groups:
            theta["gamma"] = torch.full_like(st.profit, self.mcfg.block_gamma_init)
        if "lp" in self.mcfg.theta_groups:
            theta["eta"] = torch.full_like(st.profit, self.mcfg.lp_eta_init)
        return theta

    def _step_core(self, x_s, y_s, edge_s, item_d, res_d, edge_d, glob, avail):
        with torch.profiler.record_function("t1r/controller_forward"):
            x, y, g = self.encoder.fuse_dynamic(x_s, y_s, item_d, res_d, glob)
            theta = self.controller(x, y, g, torch.cat([edge_s, edge_d], dim=-1), avail)
        return tuple(theta[key] for key in self.theta_keys)

    def proposal_parameters(self, env, inst, pre=None, amp=False):
        if self.mcfg.theta_update != "static":
            raise ValueError("Proposal readout requires a static controller")
        mu, ub = pre if pre is not None else env.precompute(inst)
        item_s,res_s,edge_s = env.static_features(inst,mu,ub)
        ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
               if amp and inst.p.device.type == "cuda" else nullcontext())
        with ctx:
            xs,ys = self.encoder.encode_static(item_s,res_s,edge_s)
            st = env.reset(inst,mu,ub)
            item_d,res_d,edge_d,glob = env.dynamic_features(st)
            values = self._step_core(xs,ys,edge_s,item_d,res_d,edge_d,glob,st.available)
        out = dict(zip(self.theta_keys,values))
        if not self.mcfg.theta_sigma_head:
            out["sigma_u"] = torch.full_like(out["u_pre"], self.mcfg.theta_sigma_u0 if "u" in self.mcfg.theta_sigma_dims else 0.)
            out["sigma_alpha"] = torch.full_like(out["alpha_pre"], self.mcfg.theta_sigma_alpha0 if "a" in self.mcfg.theta_sigma_dims else 0.)
        return {k:out[k] for k in ("u_pre","alpha_pre","sigma_u","sigma_alpha")}

    def policy(self,state):
        return self.v6_engine.policy(state)

    def value(self,state):
        return self.v6_engine.value(state)

    @property
    def feature_config(self):
        return self.v6_engine.exec_cfg

    @property
    def reuse_state_features(self):
        return False

    def forward(self, env: MKPEnv, inst: MKPInstances = None, num_samples: int = 1,
                decode: str = "sample", temperature: float = 1.0, amp: bool = False,
                pre: Optional[tuple] = None, return_trace: bool = False,
                theta_override=None, theta_noise: Optional[dict] = None, sigma_override=None):
        """Construct feasible maximal solutions, with one trajectory for primary greedy.

        theta_override='fixed' skips the network entirely. A dict supplies explicit
        u(B,M) and alpha(B,) coefficients, repeated across samples when needed.
        Static training computes theta at t=0 and retains its computation graph.
        theta_noise samples detached latent actions for a Gaussian score function;
        item choices are always greedy. Step controls refresh every theta_interval.
        """
        if self.mcfg.executor == "v6":
            if inst is None:
                return self.v6_engine.step_forward(env)
            if theta_noise is not None or sigma_override is not None:
                raise ValueError("T8 supports item-PG/PPO only, no theta-noise/SMS stages")
            return self.v6_engine.episode(inst,num_samples,decode,theta_override,return_trace)
        if theta_noise is not None:
            if theta_override is not None or self.decoder.mode != "hard_mask" or "alpha" not in self.mcfg.theta_groups:
                raise ValueError("Theta noise requires learned price/alpha controls and hard_mask")
            decode = "greedy"
        if num_samples < 1 or decode not in ("sample", "greedy"):
            raise ValueError("num_samples >= 1 and decode in {sample,greedy} required")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("sample temperature must be positive and finite")
        if isinstance(theta_override, str) and theta_override != "fixed":
            raise ValueError("Only the 'fixed' theta override is supported")
        dev, original_B, K = inst.p.device, inst.B, num_samples
        def autocast_context():
            return (torch.autocast("cuda", dtype=torch.bfloat16)
                    if amp and dev.type == "cuda" else nullcontext())
        def lp_clock():
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            return time.perf_counter()
        root_lp_time_s, lp_time_s, lp_refreshes = 0.0, 0.0, 0
        if pre is None:
            started = lp_clock()
            mu, ub = env.precompute(inst)
            root_lp_time_s = lp_clock() - started
        else:
            mu, ub = pre
        x_s = y_s = edge_s = None
        if theta_override is None:
            item_s, res_s, edge_s = env.static_features(inst, mu, ub)
            with autocast_context():
                x_s, y_s = self.encoder.encode_static(item_s, res_s, edge_s)
        static_means = None
        use_ckpt = self.training and self.mcfg.grad_checkpoint and torch.is_grad_enabled()
        if theta_noise is not None and self.mcfg.theta_update == "static":
            initial = env.reset(inst, mu, ub)
            if not bool(initial.done.all()):
                item_d, res_d, edge_d, glob = env.dynamic_features(initial)
                with autocast_context():
                    args = (x_s, y_s, edge_s, item_d, res_d, edge_d, glob, initial.available)
                    vals = (checkpoint(self._step_core, *args, use_reentrant=False) if use_ckpt
                            else self._step_core(*args))
                static_means = {key: val.repeat_interleave(K, 0) for key, val in zip(self.theta_keys, vals)}
        if K > 1:
            rep = lambda t: t.repeat_interleave(K, 0)
            if theta_override is None:
                x_s, y_s, edge_s = rep(x_s), rep(y_s), rep(edge_s)
            inst, mu, ub = inst.repeat(K), rep(mu), rep(ub)
        st = env.reset(inst, mu, ub)
        theta_static = None
        if theta_override is not None:
            if isinstance(theta_override, str):
                theta_static = self.fixed_theta(st)
            elif isinstance(theta_override, dict) and set(theta_override) == set(self.fixed_theta(st)):
                theta_static = {}
                for name, value in theta_override.items():
                    value = torch.as_tensor(value, dtype=torch.float32, device=dev)
                    if K > 1 and value.shape[0] == original_B:
                        value = value.repeat_interleave(K, 0)
                    theta_static[name] = value
                if theta_static["u"].shape != (inst.B, inst.M) or theta_static["alpha"].shape != (inst.B,):
                    raise ValueError("theta_override must have u(B,M) and alpha(B,) shapes")
                if not all(bool(v.isfinite().all()) for v in theta_static.values()):
                    raise ValueError("theta_override must be finite")
                if not torch.allclose(theta_static["u"].mean(1), torch.zeros(inst.B, device=dev), atol=1e-5):
                    raise ValueError("theta_override u must be centered")
                span = theta_static["u"].amax(1) - theta_static["u"].amin(1)
                if bool((span > 2 * self.mcfg.u_max + 1e-5).any()):
                    raise ValueError("theta_override u exceeds the centered tanh price range")
                if bool(((theta_static["alpha"] < 0.5) | (theta_static["alpha"] > 2.5)).any()):
                    raise ValueError("theta_override alpha must lie in [0.5,2.5]")
                for name in ("gamma", "eta"):
                    if name in theta_static and (theta_static[name].shape != (inst.B,) or bool((theta_static[name] < 0).any())):
                        raise ValueError(f"theta_override {name} must be nonnegative with shape (B,)")
            else:
                raise ValueError("theta_override must be 'fixed' or a u/alpha dict")
        logp_sum = torch.zeros(inst.B, device=dev)
        ent_sum, n_dec = torch.zeros_like(logp_sum), torch.zeros_like(logp_sum)
        theta_logp = torch.zeros_like(logp_sum)
        stats = {k: 0.0 for k in ("u_abs_mean", "alpha_mean", "alpha_lower_frac", "alpha_upper_frac", "u_boundary_frac", "raw_score_std")}
        if "block" in self.mcfg.theta_groups:
            stats["gamma_mean"] = 0.0
        if "lp" in self.mcfg.theta_groups:
            stats["eta_mean"] = 0.0
        sigma_stats = None
        n_stat = 0
        trace = [] if return_trace else None
        use_ckpt = self.training and self.mcfg.grad_checkpoint and torch.is_grad_enabled()
        # All arms share the same initial global T; fixed never reads learned parameters.
        global_T = (torch.tensor(self.mcfg.temperature_init, device=dev)
                    if theta_override is not None else self.temperature)

        theta = None
        pre_means = sampled_pre = None
        residual_needed = (self.mcfg.price_anchor == "residual" or
                           (self.decoder.use_lp and self.mcfg.lp_source == "residual"))
        while not bool(st.done.all()):
            if residual_needed and st.t > 0 and st.t % self.mcfg.lp_interval == 0:
                started = lp_clock()
                env.refresh_lp(st)
                lp_time_s += lp_clock() - started
                lp_refreshes += 1
            decision = (theta_static is None and
                        (theta is None or (self.mcfg.theta_update == "step" and
                                           st.t % self.mcfg.theta_interval == 0)))
            if decision:
                if static_means is not None:
                    controls = static_means
                else:
                    item_d, res_d, edge_d, glob = env.dynamic_features(st)
                    with autocast_context():
                        args = (x_s, y_s, edge_s, item_d, res_d, edge_d, glob, st.available)
                        values = (checkpoint(self._step_core, *args, use_reentrant=False)
                                  if use_ckpt else self._step_core(*args))
                    controls = dict(zip(self.theta_keys, values))
                if theta_noise is not None:
                    pre_means = {key: controls[key] for key in ("u_pre", "alpha_pre", "gamma_pre", "eta_pre")
                                 if key in controls}
                    sigmas = {"u_pre": theta_noise.get("sigma_u", .05),
                              "alpha_pre": theta_noise.get("sigma_alpha", .3),
                              "gamma_pre": theta_noise.get("sigma_gamma", .3),
                              "eta_pre": theta_noise.get("sigma_eta", .3)}
                    if self.mcfg.theta_sigma_head:
                        sigmas["u_pre"], sigmas["alpha_pre"] = controls["sigma_u"], controls["sigma_alpha"]
                    if sigma_override is not None:
                        sigmas["u_pre"] = torch.as_tensor(sigma_override["sigma_u"], device=dev).expand_as(pre_means["u_pre"])
                        sigmas["alpha_pre"] = torch.as_tensor(sigma_override["sigma_alpha"], device=dev).expand_as(pre_means["alpha_pre"])
                    sampled_pre, decision_logp = sample_theta_pre(
                        pre_means, sigmas, original_B, K,
                        antithetic=bool(theta_noise.get("antithetic", True)) and st.t == 0,
                        generator=theta_noise.get("generator"),
                        include_center=bool(theta_noise.get("include_center", False)) and st.t == 0)
                    theta = theta_from_pre(sampled_pre["u_pre"], sampled_pre["alpha_pre"],
                                           self.mcfg.u_max, sampled_pre.get("gamma_pre"), sampled_pre.get("eta_pre"))
                    theta_logp = theta_logp + decision_logp * (~st.done).float()
                    if self.mcfg.theta_sigma_head:
                        with torch.no_grad():
                            sigma_stats = {}
                            for field,key in (("sigma_u","u_pre"),("sigma_alpha","alpha_pre")):
                                value = torch.as_tensor(sigmas[key],device=dev).expand_as(pre_means[key])
                                for stat in ("mean","min","max"):
                                    sigma_stats[field+"_"+stat] = float(getattr(value,stat)())
                            mean_u = theta_from_pre(pre_means["u_pre"],pre_means["alpha_pre"],self.mcfg.u_max)["u"]
                            sigma_stats["proposal_u_abs_mean"] = float(mean_u.abs().mean())
                else:
                    theta = {key: controls[key] for key in ("u", "alpha", "gamma", "eta") if key in controls}
                if self.mcfg.theta_update == "static":
                    theta_static = theta
            elif theta_static is not None:
                theta = theta_static
            # Fixed overrides are a decision at t=0 for trace/statistics only.
            is_decision = decision or (theta_override is not None and st.t == 0)
            with torch.profiler.record_function("t1r/executor"):
                logits, lam, raw_std = self.decoder(st, theta, return_raw_std=True)
            valid = st.fits if self.decoder.mode == "hard_mask" else st.available
            if not bool(torch.isfinite(logits[valid]).all()):
                raise FloatingPointError("Non-finite scores on candidate items")
            # T is global, not a state-level control coefficient; greedy ranking ignores T.
            finite = torch.isfinite(logits)
            scaled = (logits.float().masked_fill(~finite, 0.0) / (global_T * temperature))
            scaled = scaled.masked_fill(~finite, NEG)
            logp_all = torch.log_softmax(scaled, -1)
            action = (logits.argmax(-1) if decode == "greedy" else
                      torch.multinomial(logp_all.exp(), 1).squeeze(-1))
            active = (~st.done).float()
            logp_sum = logp_sum + logp_all.gather(1, action[:, None]).squeeze(1) * active
            # Mask BEFORE multiplication to keep -inf entries out of entropy gradients.
            safe_logp = logp_all.masked_fill(~torch.isfinite(logp_all), 0.0)
            ent_sum = ent_sum - (logp_all.exp() * safe_logp).sum(-1) * active
            n_dec = n_dec + active
            with torch.no_grad():
                n_act = active.sum().clamp_min(1.0)
                stats["u_abs_mean"] += (theta["u"].abs().mean(1) * active).sum() / n_act
                stats["alpha_mean"] += (theta["alpha"] * active).sum() / n_act
                stats["alpha_lower_frac"] += ((theta["alpha"] < .52) * active).sum() / n_act
                stats["alpha_upper_frac"] += ((theta["alpha"] > 2.48) * active).sum() / n_act
                stats["u_boundary_frac"] += ((theta["u"].abs() > .95*self.mcfg.u_max).float().mean(1)*active).sum()/n_act
                if "gamma" in theta:
                    stats["gamma_mean"] += (theta["gamma"] * active).sum()/n_act
                if "eta" in theta:
                    stats["eta_mean"] += (theta["eta"] * active).sum()/n_act
                stats["raw_score_std"] += (raw_std.squeeze(1)*active).sum()/n_act
                n_stat += 1
                if return_trace:
                    trace.append({"action": action.detach().cpu(), "active": active.bool().cpu(),
                                  "u": theta["u"].detach().cpu(), "alpha": theta["alpha"].detach().cpu(),
                                  "temperature": global_T.detach().expand(inst.B).cpu(),
                                  "t": st.t, "theta_decision": is_decision,
                                  "lambda": lam.detach().cpu(),
                                  "fits_before": st.fits.gather(1, action[:, None]).squeeze(1).cpu()})
                    if "gamma" in theta:
                        trace[-1]["gamma"] = theta["gamma"].detach().cpu()
                    if "eta" in theta:
                        trace[-1]["eta"] = theta["eta"].detach().cpu()
                    if theta_noise is not None and is_decision:
                        trace[-1]["pre_mean"] = {k: v.detach().cpu() for k, v in pre_means.items()}
                        trace[-1]["pre_sample"] = {k: v.detach().cpu() for k, v in sampled_pre.items()}
            with torch.profiler.record_function("t1r/environment"):
                st = env.step(st, action)
        out = {"reward": env.get_reward(st), "profit": st.profit, "n_reject": st.n_reject,
               "log_likelihood": logp_sum, "entropy": ent_sum / n_dec.clamp_min(1),
               "solution": st.solution, "ub": st.ub, "steps": st.t,
               "u_abs_mean": float(stats["u_abs_mean"] / max(n_stat, 1)),
               "alpha_mean": float(stats["alpha_mean"] / max(n_stat, 1)),
               "temperature": float(global_T.detach()),
               "theta_dim": inst.M + int("alpha" in self.mcfg.theta_groups) + int("block" in self.mcfg.theta_groups) + int("lp" in self.mcfg.theta_groups),
               "lp_time_s": lp_time_s, "root_lp_time_s": root_lp_time_s, "lp_refreshes": lp_refreshes,
               "theta_logp": theta_logp}
        out.update({k: float(v / max(n_stat, 1)) for k, v in stats.items()})
        if sigma_stats is not None:
            out.update(sigma_stats)
        if return_trace:
            out["trace"] = trace
        return out


def build_model(cfg) -> ConstrainedMKPPolicy:
    return ConstrainedMKPPolicy(cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
