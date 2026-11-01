"""Resource-valued controller and a parameter-free, differentiable executor.

Only prices[M] and criterion_weights[4] cross the learned/execution boundary.
Item embeddings are internal state representation, never learned item logits.
All dimensions are independent of N and support arbitrary positive M.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from mkp_config import ModelConfig
from mkp_env import MKPState


@dataclass
class Theta:
    prices: torch.Tensor
    weights: torch.Tensor

    def detached(self) -> "Theta":
        return Theta(self.prices.detach(), self.weights.detach())


def _active_criteria(config: ModelConfig, device) -> torch.Tensor:
    return torch.tensor([True, True, config.lp_score, config.pair_score], device=device)


def _price_anchor(state: MKPState, use_lp: bool, eps: float) -> torch.Tensor:
    # Capacity-scaled prices pair with w/C in the executor, preserving y*w.
    price = (state.dual * state.c).float() if use_lp else torch.ones(state.m, device=state.device)
    if float(price.sum()) <= eps:
        price = torch.ones_like(price)
    return price / price.sum()


def fixed_theta(state: MKPState, config: ModelConfig, use_lp: bool = False) -> Theta:
    """Transparent price-density baseline in exactly the same executor family."""
    return Theta(_price_anchor(state, use_lp, config.eps),
                 torch.tensor([1., 0., 0., 0.], device=state.device))


def _fixed_inputs(state: MKPState, config: ModelConfig):
    """State-relative, dimensionless fixed descriptors; no trainable features.

    Shapes: item[F,8], resource[M,12], ratio[F,M], global[8].
    LP-input ablation clears every dual/upper-bound input to the encoder.
    """
    key = ("inputs", config.lp_input, config.pair_features, config.pair_chunk_size,
           config.resource_log_clip, config.eps)
    if key in state._cache:
        return state._cache[key]
    if state.done:
        raise ValueError("The terminal state has no policy decision")
    eps = config.eps
    mask = state.feasible
    p, w = state.p[mask].float(), state.w[mask].float()
    rem, cap = state.remaining.float(), state.c.float()
    pscale = state.p.max().float().clamp_min(eps)
    total = state.p.sum().float().clamp_min(eps)
    pnorm = p / pscale
    ratio = w / rem.clamp_min(eps)
    remain_fraction = torch.where(cap > 0, rem / cap.clamp_min(eps), 1.)
    if config.pair_features:
        blocked_all, partner_all = state.pair_descriptors(config.pair_chunk_size)
        blocked, partner = blocked_all[mask].float(), partner_all[mask].float()
    else:
        blocked, partner = torch.zeros_like(p), torch.zeros_like(p)
    dual = state.dual.float()
    if config.lp_input:
        lpcost = (w @ dual) / pscale
        reduced = pnorm - lpcost
        dual_input = _price_anchor(state, True, eps)
        dual_remaining = dual * rem / total
        dual_share = dual * w.sum(0) / total
        upper = state.potential()
    else:
        lpcost, reduced = torch.zeros_like(p), torch.zeros_like(p)
        dual_input, dual_remaining, dual_share = [torch.zeros_like(cap) for _ in range(3)]
        upper = 0.0
    item = torch.stack((pnorm, ratio.mean(-1), ratio.max(-1).values,
                        ratio.std(-1, unbiased=False), blocked, partner,
                        reduced, lpcost), dim=-1)
    pweights = p / p.sum().clamp_min(eps)
    ffrac = float(mask.sum()) / state.n
    resource = torch.stack((
        remain_fraction,
        torch.full_like(cap, ffrac),
        ratio.mean(0), ratio.max(0).values, ratio.std(0, unbiased=False),
        (ratio * pweights[:, None]).sum(0),
        (ratio * blocked[:, None]).mean(0),
        (ratio * partner[:, None]).mean(0),
        dual_input, dual_remaining, dual_share,
        cap / state.w.sum(0).float().clamp_min(eps),
    ), dim=-1)
    global_features = torch.stack((
        cap.new_tensor(ffrac), cap.new_tensor(len(state.selected) / state.n),
        cap.new_tensor(state.objective) / total,
        remain_fraction.mean(), remain_fraction.min(), remain_fraction.max(),
        p.sum() / total, cap.new_tensor(upper),
    ))
    # Zero-weight resources can have arbitrarily large C/sum(w); log1p bounds
    # their dynamic range without encoding an order or absolute unit scale.
    resource[:, -1] = torch.log1p(resource[:, -1]).clamp_max(config.resource_log_clip)
    result = item, resource, ratio, global_features
    state._cache[key] = result
    return result


class ResidualEncoder(nn.Module):
    """Invariant item aggregation and equivariant resource message exchange.

    Mean and max pooling preserve broad mass and rare candidate signals.
    Consumption-weighted item aggregation assigns context to each resource.
    No learned per-item scalar is produced by the encoder or its readouts.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        h = config.hidden_dim
        self.item_embed = nn.Sequential(nn.Linear(8, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU())
        self.resource_embed = nn.Sequential(nn.Linear(12 + 2 * h, h), nn.SiLU(), nn.Linear(h, h))
        self.exchange = nn.ModuleList([
            nn.Sequential(nn.Linear(3 * h, h), nn.SiLU(), nn.Linear(h, h))
            for _ in range(config.message_rounds)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in self.exchange])
        self.global_embed = nn.Sequential(nn.Linear(2 * h + 8, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU())

    def forward(self, state: MKPState) -> tuple[torch.Tensor, torch.Tensor]:
        item, resource, ratio, global_features = _fixed_inputs(state, self.config)
        embedded = self.item_embed(item)
        weights = ratio / ratio.sum(0, keepdim=True).clamp_min(self.config.eps)
        resource_items = weights.transpose(0, 1) @ embedded
        critical_items = embedded.max(0).values.expand(state.m, -1)
        tokens = self.resource_embed(torch.cat((resource, resource_items, critical_items), -1))
        for exchange, norm in zip(self.exchange, self.norms):
            mean = tokens.mean(0).expand_as(tokens)
            maximum = tokens.max(0).values.expand_as(tokens)
            tokens = norm(tokens + exchange(torch.cat((tokens, mean, maximum), -1)))
        global_token = self.global_embed(torch.cat((tokens.mean(0), tokens.max(0).values, global_features)))
        return tokens, global_token


class ResourceController(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = ResidualEncoder(config)
        self.price_head = nn.Linear(config.hidden_dim, 1)
        self.criteria_head = nn.Linear(config.hidden_dim, 4)
        # A small nonzero readout allows gradients to reach the encoder at start.
        nn.init.normal_(self.price_head.weight, std=config.readout_init_std)
        nn.init.zeros_(self.price_head.bias)
        nn.init.normal_(self.criteria_head.weight, std=config.readout_init_std)
        with torch.no_grad():
            self.criteria_head.bias.copy_(torch.tensor(config.initial_criterion_bias))

    def forward(self, state: MKPState) -> Theta:
        resources, global_token = self.encoder(state)
        delta = self.config.price_log_bound * torch.tanh(self.price_head(resources).squeeze(-1))
        anchor = _price_anchor(state, self.config.lp_anchor, self.config.eps)
        if self.config.lp_anchor:
            # A configurable uniform share lets a root-zero-dual resource gain
            # price later. This is a trainable-interface hypothesis, not evidence
            # that the resulting controller improves objective quality.
            mix = self.config.lp_anchor_mix
            anchor = (1 - mix) * anchor + mix / state.m
        prices = torch.softmax(anchor.clamp_min(self.config.eps).log() + delta, dim=-1)
        logits = self.criteria_head(global_token)
        logits = logits.masked_fill(~_active_criteria(self.config, logits.device), -torch.inf)
        return Theta(prices, torch.softmax(logits, dim=-1))


class ResidualCritic(nn.Module):
    """Independent state value network; no actor parameters are shared."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = ResidualEncoder(config)
        self.value_head = nn.Linear(config.hidden_dim, 1)

    def forward(self, state: MKPState) -> torch.Tensor:
        if state.done:
            return self.value_head.weight.new_zeros(())
        _, global_token = self.encoder(state)
        return self.value_head(global_token).squeeze(-1)


class FixedExecutor(nn.Module):
    """Transparent score dictionary; this module has zero trainable parameters.

    d_i = log((p_i/pmax+eps)/(sum_j price_j*w_ij/C_j+eps)).
    phi = [d_i, p_i/pmax, (p_i-y*w_i)/pmax, -blocked_profit_fraction_i].
    Each column is centered/scaled over feasible candidates. An optional
    basis_clip clips standardized columns; default None preserves their order.
    score_i=sum_k weights_k*phi_ik; pi=softmax(score/temperature).
    Temperature is fixed configuration and cannot change a greedy ordering.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    def scores(self, state: MKPState, theta: Theta) -> torch.Tensor:
        if state.done:
            raise ValueError("The terminal state has no executable action")
        if theta.prices.shape != (state.m,) or theta.weights.shape != (4,):
            raise ValueError("Theta must contain prices[M] and weights[4]")
        eps = self.config.eps
        p = state.p.float() / state.p.max().float().clamp_min(eps)
        consumption = state.w.float() / state.c.float().clamp_min(eps)
        cost = consumption @ theta.prices
        density = torch.log(p + eps) - torch.log(cost + eps)
        reduced = ((state.p - state.w @ state.dual) / state.p.max().clamp_min(eps)).float()
        if self.config.pair_score:
            blocked = state.pair_descriptors(self.config.pair_chunk_size)[0].float()
        else:
            blocked = torch.zeros_like(p)
        basis = torch.stack((density, p, reduced, -blocked), dim=-1)
        feasible_basis = basis[state.feasible]
        mean = feasible_basis.mean(0)
        scale = ((feasible_basis - mean).square().mean(0) + eps).sqrt()
        basis = (basis - mean) / scale
        if self.config.basis_clip is not None:
            basis = basis.clamp(-self.config.basis_clip, self.config.basis_clip)
        active = _active_criteria(self.config, theta.weights.device)
        weights = theta.weights * active
        weights = weights / weights.sum().clamp_min(eps)
        score = basis @ weights
        return score.masked_fill(~state.feasible, -torch.inf)

    def distribution(self, state: MKPState, theta: Theta) -> Categorical:
        return Categorical(logits=self.scores(state, theta) / self.config.temperature)


@torch.no_grad()
def run_episode(controller: ResourceController, state: MKPState, mode="dynamic",
                sample=False, generator: torch.Generator | None = None) -> dict:
    """One pure construction trajectory; caller controls repetitions/budgets.

    static freezes theta at the initial state; fixed/lp_dual freeze transparent
    price-density coefficients. untrained is dynamic with a caller-supplied
    initial model. All modes share the same executor, masks, and score family.
    Greedy argmax breaks exactly tied scores by the first original item index;
    score/probability equivariance does not imply trajectory equivariance at ties.
    """
    if mode not in {"dynamic", "static", "fixed", "lp_dual", "untrained"}:
        raise ValueError(f"Unknown policy mode: {mode}")
    config = controller.config
    executor = FixedExecutor(config)
    root_theta = None
    if not state.done:
        if mode == "static":
            root_theta = controller(state)
        elif mode in {"fixed", "lp_dual"}:
            root_theta = fixed_theta(state, config, use_lp=(mode == "lp_dual"))
    trace, changes = [], []
    previous = None
    while not state.done:
        theta = root_theta if root_theta is not None else controller(state)
        if sample:
            probs = executor.distribution(state, theta).probs
            action = int(torch.multinomial(probs, 1, generator=generator).item())
        else:
            action = int(executor.scores(state, theta).argmax().item())
        current = torch.cat((theta.prices, theta.weights))
        if previous is not None:
            changes.append(float((current - previous).abs().mean()))
        previous = current
        trace.append({"step": len(state.selected), "action": action,
                      "prices": theta.prices.cpu().tolist(), "weights": theta.weights.cpu().tolist()})
        state.step(action)
    result = state.solution_metrics()
    result["theta_change_mean"] = sum(changes) / max(len(changes), 1)
    result["theta_trace"] = trace
    if trace:
        all_prices = torch.tensor([step["prices"] for step in trace])
        result["theta_price_mean"] = float(all_prices.mean())
        result["theta_price_std"] = float(all_prices.std(unbiased=False))
    else:
        result.update(theta_price_mean=0.0, theta_price_std=0.0)
    return result
