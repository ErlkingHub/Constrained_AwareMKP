# Ported from D:/PhD_Doctor/Study/Codes/RL_ModelInhance/mkp_Model.py, commit 4f935551e62d91d55d998fa6255a67f9b0fafeb7.
# StrategyTheta, CandidatePolicy and WeakDecoder copied without arithmetic changes.
import math
from dataclasses import dataclass, fields
import torch
from mkp_v6_config import ModelConfig
from mkp_v6_lp import lp_relaxation
from mkp_v6_basis import interaction_loads, companion_basis
from mkp_v6_cache import cached_features

THETA_STATISTICS = ('profit_exponent','scarcity_exponent','bottleneck_weight','temperature','price_std','opportunity_weight','reserve_weight','lp_primal_weight','lp_dual_weight','interaction_weight_sum','companion_weight_sum')

@dataclass
class StrategyTheta:
    resource_prices: torch.Tensor  # [B, M], strictly positive, mean 1
    profit_exponent: torch.Tensor # [B]
    scarcity_exponent: torch.Tensor
    bottleneck_weight: torch.Tensor
    temperature: torch.Tensor
    opportunity_weight: torch.Tensor  # [B], penalty for lost feasible profit mass
    reserve_weight: torch.Tensor      # [B], smooth loss of future capacity margins
    lp_primal_weight: torch.Tensor    # [B], affinity for items the residual LP keeps
    lp_dual_weight: torch.Tensor      # [B], LP reduced cost under shadow prices
    resource_interactions: torch.Tensor  # [B,M,M], symmetric; upper-triangle coefficients counted once
    companion_weights: torch.Tensor      # [B,K], nonnegative fixed pair-basis coefficients

    def detach(self):
        return StrategyTheta(*(getattr(self, field.name).detach() for field in fields(self)))

    def statistics(self):
        """[B, len(THETA_STATISTICS)] in that order."""
        return torch.stack((self.profit_exponent, self.scarcity_exponent, self.bottleneck_weight,
                            self.temperature, self.resource_prices.std(-1, unbiased=False),
                            self.opportunity_weight, self.reserve_weight,
                            self.lp_primal_weight, self.lp_dual_weight,
                            self.resource_interactions.triu().sum((-1, -2)), self.companion_weights.sum(-1)), -1)


class CandidatePolicy:
    """Ragged categorical distribution. Only actual candidates are stored.

    No dense logits with -inf and no invalid-action rejection loop. Segment
    reductions support different candidate counts and already terminal rows.
    """
    def __init__(self, owners, items, logits, batch_size, score_terms=None):
        self.owners, self.items, self.logits = owners, items, logits
        self.batch_size = batch_size
        self.score_terms = score_terms
        maxima = logits.new_full((batch_size,), -torch.inf)
        maxima = maxima.scatter_reduce(0, owners, logits, reduce="amax", include_self=True)
        centered = logits - maxima[owners]
        total = logits.new_zeros(batch_size).scatter_add(0, owners, centered.exp())
        self.log_probs = centered - total[owners].log()

    def choose(self, stochastic=False):
        values = self.log_probs
        if stochastic:
            uniform = torch.rand_like(values).clamp(1e-7, 1 - 1e-7)
            values = values - (-uniform.log()).log()
        maxima = values.new_full((self.batch_size,), -torch.inf)
        maxima.scatter_reduce_(0, self.owners, values, reduce="amax", include_self=True)
        # Exact ties use the smallest original item index; empty rows receive -1.
        sentinel = torch.iinfo(torch.long).max
        chosen = torch.where(values == maxima[self.owners], self.items, sentinel)
        result = self.items.new_full((self.batch_size,), sentinel)
        result.scatter_reduce_(0, self.owners, chosen, reduce="amin", include_self=True)
        return torch.where(result == sentinel, -1, result)

    def log_prob(self, actions):
        selected = self.items == actions[self.owners]
        return self.logits.new_zeros(self.batch_size).scatter_add(
            0, self.owners[selected], self.log_probs[selected])

    def entropy(self):
        return self.logits.new_zeros(self.batch_size).scatter_add(
            0, self.owners, -self.log_probs.exp() * self.log_probs)

    def normalized_entropy(self):
        count = self.logits.new_zeros(self.batch_size).scatter_add(
            0, self.owners, torch.ones_like(self.logits))
        return self.entropy() / count.clamp_min(2).log()


class WeakDecoder:
    """No nn.Module, trainable weight, pointer, search, local repair or solver.

    q_ij = N * w_ij / C_j ; s_j = C_j / max(c_j, eps)
    cost_i = mean_j(price_j * q_ij * s_j ** beta)
    score_i = alpha * log(p_i / mean(p)) - log(cost_i)
              - kappa * max_j(w_ij / max(c_j, eps))
              - opportunity_weight * blocked_profit_mass_i
              - reserve_weight * smooth_margin_loss_i
              + lp_primal_weight * log(x_lp_i + floor)
              + lp_dual_weight * reduced_cost_i
              - sum_{r<=s} interaction_rs * z_ir * z_is
              + sum_k companion_weight_k * companion_basis_ik
    pi(i|s) = softmax(score_i / temperature) over feasible candidates.
    """
    def __init__(self, config: ModelConfig):
        self.config = config

    @torch.no_grad()
    def opportunity_features(self, state, owners, items):
        """Transparent pairwise capacity arithmetic, not a rollout or learned Q head.

        Exclude the chosen item and already infeasible items. Normalize both losses
        by the same pre-decision feasible profit mass. Chunk candidate x other-item
        tensors to bound peak storage; comparisons retain float64 accounting.
        """
        feasible = state.feasible
        mass = state.profits * feasible
        total = mass.sum(-1).clamp_min(self.config.eps)
        baseline = ((state.remaining[:, None] - state.weights)
                    / state.remaining[:, None].clamp_min(self.config.eps)).amin(-1)
        baseline = torch.sigmoid(baseline.float() / self.config.opportunity_bandwidth)
        blocked, reserve = [], []
        item_ids = torch.arange(state.profits.shape[1], device=items.device)
        for row, col in zip(owners.split(self.config.opportunity_chunk_size),
                            items.split(self.config.opportunity_chunk_size)):
            after = state.remaining[row] - state.weights[row, col]
            margin = after[:, None] - state.weights[row]
            lost = (margin < 0).any(-1)
            soft = torch.sigmoid((margin / state.remaining[row, None].clamp_min(self.config.eps))
                                 .amin(-1).float() / self.config.opportunity_bandwidth)
            other_mass = mass[row] * (item_ids[None] != col[:, None])
            blocked.append(((lost * other_mass).sum(-1) / total[row]).float())
            reserve.append((((baseline[row] - soft).clamp_min(0) * other_mass).sum(-1)
                            / total[row]).float())
        if not blocked:
            empty = state.profits.new_empty(0, dtype=torch.float32)
            return empty, empty
        return torch.cat(blocked), torch.cat(reserve)

    def __call__(self, state, theta):
        owners, items = state.feasible.nonzero(as_tuple=True)
        cached = cached_features(state, self.config)
        eps = self.config.eps
        p, w, c, rem = (x.float() for x in
                        (state.profits, state.weights, state.capacities, state.remaining))
        # Calculate only feasible-item scores; Actor attention does not define legality.
        candidate_w = w[owners, items]
        scale = state.profits.shape[1]
        log_scarcity = (c[owners].clamp_min(eps).log() - rem[owners].clamp_min(eps).log())
        # Log-space cost avoids overflow when a resource is exhausted (zero loads are legal).
        zero_cost = (candidate_w == 0).all(-1)
        log_load = (scale * candidate_w / c[owners]).clamp_min(eps).log()
        # -inf here represents a zero summand in a resource cost, never an item logit.
        log_load = torch.where(candidate_w > 0, log_load, -torch.inf)
        log_components = (theta.resource_prices[owners].log() + log_load
                          + theta.scarcity_exponent[owners, None] * log_scarcity)
        log_components = torch.where(zero_cost[:, None], torch.zeros_like(log_components), log_components)
        log_cost = torch.logsumexp(log_components, -1) - math.log(c.shape[-1])
        log_cost = torch.where(zero_cost, log_cost.new_full(log_cost.shape, math.log(eps)), log_cost)
        profit = p[owners, items] / p.mean(-1)[owners].clamp_min(eps)
        bottleneck = (candidate_w / rem[owners].clamp_min(eps)).amax(-1)
        scores = theta.profit_exponent[owners] * profit.clamp_min(eps).log() - log_cost
        scores = scores - theta.bottleneck_weight[owners] * bottleneck
        if self.config.opportunity_weight_max or self.config.reserve_weight_max:
            blocked, reserve = (self.opportunity_features(state, owners, items) if cached is None else
                                (cached["blocked"][owners, items], cached["reserve"][owners, items]))
            scores = (scores - theta.opportunity_weight[owners] * blocked
                      - theta.reserve_weight[owners] * reserve)
        if self.config.lp_primal_weight_max or self.config.lp_dual_weight_max:
            if cached is None:
                lp = lp_relaxation(state, self.config)
                primal, reduced = lp.primal[owners, items], lp.reduced[owners, items]
            else:
                primal, reduced = cached["lp_primal"][owners, items], cached["lp_reduced"][owners, items]
            # x_lp is a fixed relaxation value; floor bounds the penalty for LP-rejected items.
            scores = (scores + theta.lp_primal_weight[owners] * (primal + self.config.lp_primal_floor).log()
                      + theta.lp_dual_weight[owners] * reduced.clamp(-self.config.feature_clip,
                                                                     self.config.feature_clip))
        base = scores
        penalty, bonus = torch.zeros_like(scores), torch.zeros_like(scores)
        if self.config.interaction_weight_max:
            z = (interaction_loads(state, owners, items, eps) if cached is None else
                 cached["interaction"][owners, items])
            penalty = (theta.resource_interactions[owners].triu() * z[:, :, None] * z[:, None, :]).sum((-1, -2))
        if self.config.companion_weight_max:
            basis = (companion_basis(state, owners, items, self.config) if cached is None else
                     cached["companion"][owners, items])
            bonus = (theta.companion_weights[owners] * basis).sum(-1)
        scores = base - penalty + bonus
        return CandidatePolicy(owners, items, scores / theta.temperature[owners], p.shape[0],
                               dict(base=base, interaction_penalty=penalty, companion_bonus=bonus, score=scores))
