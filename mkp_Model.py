"""Item-resource relation encoder, resource-level theta controller and fixed executor.

T1 interface: M centered log-price corrections u and optional scalar alpha.
The encoder has item tokens, but no learned item-level output enters the score.
Static encoding is cached once; the original per-step representation is preserved.
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
        groups = list(cfg.theta_groups)
        if len(groups) != len(set(groups)) or "price" not in groups:
            raise ValueError("theta_groups must contain price exactly once")
        if set(groups) - {"price", "alpha"}:
            raise ValueError("T1 supports price/alpha only; block and lp belong to T2/T3")
        self.scalar_names = [name for name in ("alpha",) if name in groups]
        self.blocks = nn.ModuleList([
            DualAxisBlock(d, cfg.num_heads, cfg.ff_mult, cfg.dropout,
                          F_EDGE_STATIC + F_EDGE_DYN, cfg.edge_hidden)
            for _ in range(cfg.controller_layers)])
        self.ln_x, self.ln_y = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ctx = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, d))
        self.price_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.scalar_head = nn.Linear(d, len(self.scalar_names)) if self.scalar_names else None
        self._init_heads()

    def _init_heads(self):
        if self.cfg.untrained_random_heads:
            return  # Preserve PyTorch's default random head initialization for this control.
        nn.init.zeros_(self.price_head[-1].weight)
        nn.init.zeros_(self.price_head[-1].bias)
        if self.scalar_head is not None:
            nn.init.zeros_(self.scalar_head.weight)
            nn.init.constant_(self.scalar_head.bias, math.log(1.0 / 3.0))  # alpha=1

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
        M = y.shape[1]
        u = cfg.u_max * torch.tanh(self.price_head(
            torch.cat([y, ctx[:, None, :].expand(-1, M, -1)], -1)).squeeze(-1).float())
        u = u - u.mean(1, keepdim=True)
        alpha = (0.5 + 2.0 * torch.sigmoid(self.scalar_head(ctx).float()[:, 0])
                 if self.scalar_head is not None else torch.ones_like(u[:, 0]))
        return {"u": u, "alpha": alpha}


# ======================================================================================
# Low-level weak decoder (parameter-free, interpretable)
# ======================================================================================
class WeakDecoder:
    """Pure transparent T1 scorer; owns no trainable parameters or item embeddings."""

    def __init__(self, model_cfg, env_cfg):
        self.normalize = model_cfg.score_normalization
        self.price_floor = model_cfg.price_floor
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
        base = st.mu + self.price_floor * st.mu.mean(1, keepdim=True) + 1e-6
        lam = base * theta["u"].exp()
        cost = torch.einsum("bm,bmn->bn", lam, st.inst.r / (st.c_rem[:, :, None] + 1.0))
        p_norm = st.inst.p / st.inst.p.mean(1, keepdim=True).clamp_min(_EPS)
        score = theta["alpha"][:, None] * p_norm.clamp_min(_EPS).log() - cost.clamp_min(1e-9).log()
        return score, lam

    def __call__(self, st: MKPState, theta: Dict[str, torch.Tensor]):
        score, lam = self.raw_scores(st, theta)
        if self.normalize:
            mean, std = self.moments(score.float(), st.fits)
            score = (score.float() - mean) / (std + 1e-6)
        if self.mode == "soft_mask":
            over = (st.inst.r - st.c_rem[:, :, None]).clamp_min(0) / (st.inst.r + 1.0)
            viol = over.max(1).values
            score = score - self.beta * ((viol > 0).float() + viol)
        valid = st.fits if self.mode == "hard_mask" else st.available
        score = score.masked_fill(~valid, NEG)
        # Finished rows are no-ops; avoid a fully masked softmax in a mixed batch.
        return torch.where(st.done[:, None], torch.zeros_like(score), score), lam


class ConstrainedMKPPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mcfg, self.ecfg = cfg.model, cfg.env
        if cfg.model.price_anchor != "root":
            raise ValueError("T1 supports root price_anchor only; residual LP belongs to T3")
        if cfg.model.theta_update not in ("step", "static"):
            raise ValueError("theta_update must be step or static")
        if cfg.model.temperature_mode not in ("learned", "fixed"):
            raise ValueError("temperature_mode must be learned or fixed")
        if not math.isfinite(cfg.model.temperature_init) or cfg.model.temperature_init <= 0:
            raise ValueError("temperature_init must be positive and finite")
        if not math.isfinite(cfg.model.u_max) or cfg.model.u_max < 0:
            raise ValueError("u_max must be finite and nonnegative")
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

    @staticmethod
    def fixed_theta(st: MKPState):
        return {"u": torch.zeros_like(st.mu), "alpha": torch.ones_like(st.profit)}

    def _step_core(self, x_s, y_s, edge_s, item_d, res_d, edge_d, glob, avail):
        with torch.profiler.record_function("t1r/controller_forward"):
            x, y, g = self.encoder.fuse_dynamic(x_s, y_s, item_d, res_d, glob)
            theta = self.controller(x, y, g, torch.cat([edge_s, edge_d], dim=-1), avail)
        return theta["u"], theta["alpha"]

    def forward(self, env: MKPEnv, inst: MKPInstances, num_samples: int = 1,
                decode: str = "sample", temperature: float = 1.0, amp: bool = False,
                pre: Optional[tuple] = None, return_trace: bool = False,
                theta_override=None):
        """Construct feasible maximal solutions, with one trajectory for primary greedy.

        theta_override='fixed' skips the network entirely. A dict supplies explicit
        u(B,M) and alpha(B,) coefficients, repeated across samples when needed.
        Static training computes theta at t=0 and retains its computation graph.
        """
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
        mu, ub = pre if pre is not None else env.precompute(inst)
        x_s = y_s = edge_s = None
        if theta_override is None:
            item_s, res_s, edge_s = env.static_features(inst, mu, ub)
            with autocast_context():
                x_s, y_s = self.encoder.encode_static(item_s, res_s, edge_s)
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
            elif isinstance(theta_override, dict) and set(theta_override) == {"u", "alpha"}:
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
            else:
                raise ValueError("theta_override must be 'fixed' or a u/alpha dict")
        logp_sum = torch.zeros(inst.B, device=dev)
        ent_sum, n_dec = torch.zeros_like(logp_sum), torch.zeros_like(logp_sum)
        stats = {k: 0.0 for k in ("u_abs_mean", "alpha_mean", "alpha_lower_frac", "alpha_upper_frac", "u_boundary_frac", "raw_score_std")}
        n_stat = 0
        trace = [] if return_trace else None
        use_ckpt = self.training and self.mcfg.grad_checkpoint and torch.is_grad_enabled()
        # All arms share the same initial global T; fixed never reads learned parameters.
        global_T = (torch.tensor(self.mcfg.temperature_init, device=dev)
                    if theta_override is not None else self.temperature)

        while not bool(st.done.all()):
            if theta_static is None:
                item_d, res_d, edge_d, glob = env.dynamic_features(st)
                with autocast_context():
                    args = (x_s, y_s, edge_s, item_d, res_d, edge_d, glob, st.available)
                    values = (checkpoint(self._step_core, *args, use_reentrant=False)
                              if use_ckpt else self._step_core(*args))
                theta = dict(zip(("u", "alpha"), values))
                if self.mcfg.theta_update == "static":
                    theta_static = theta
            else:
                theta = theta_static
            with torch.profiler.record_function("t1r/executor"):
                logits, lam = self.decoder(st, theta)
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
                raw, _ = self.decoder.raw_scores(st, theta)
                _, raw_std = self.decoder.moments(raw, st.fits)
                stats["raw_score_std"] += (raw_std.squeeze(1)*active).sum()/n_act
                n_stat += 1
                if return_trace:
                    trace.append({"action": action.detach().cpu(), "active": active.bool().cpu(),
                                  "u": theta["u"].detach().cpu(), "alpha": theta["alpha"].detach().cpu(),
                                  "temperature": global_T.detach().expand(inst.B).cpu(),
                                  "lambda": lam.detach().cpu(),
                                  "fits_before": st.fits.gather(1, action[:, None]).squeeze(1).cpu()})
            with torch.profiler.record_function("t1r/environment"):
                st = env.step(st, action)
        out = {"reward": env.get_reward(st), "profit": st.profit, "n_reject": st.n_reject,
               "log_likelihood": logp_sum, "entropy": ent_sum / n_dec.clamp_min(1),
               "solution": st.solution, "ub": st.ub, "steps": st.t,
               "u_abs_mean": float(stats["u_abs_mean"] / max(n_stat, 1)),
               "alpha_mean": float(stats["alpha_mean"] / max(n_stat, 1)),
               "temperature": float(global_T.detach()),
               "theta_dim": inst.M + int("alpha" in self.mcfg.theta_groups)}
        out.update({k: float(v / max(n_stat, 1)) for k, v in stats.items()})
        if return_trace:
            out["trace"] = trace
        return out


def build_model(cfg) -> ConstrainedMKPPolicy:
    return ConstrainedMKPPolicy(cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
