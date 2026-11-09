# Ported from D:/PhD_Doctor/Study/Codes/RL_ModelInhance/mkp_Trainer.py, commit 4f935551e62d91d55d998fa6255a67f9b0fafeb7.
# Numerical rollout/GAE/shaping/PPO recipe retained; new-framework root context carried with states.
import math
import time
import torch
from torch.nn import functional as F
from mkp_v6_env import MKPEnv, MKPState
from mkp_v6_lp import lp_potential
from mkp_v6_basis import COMBINATION_METRICS, decoder_statistics
from mkp_v6_ranking import (collect_ranking_batch, ranking_objective, ranking_kl,
                            paired_terminal_returns, return_targets)
from mkp_v6_cache import build_rollout_cache, prepare_state

def shaped_rewards(rewards, potentials, active):
    """r_t + Phi(s_{t+1}) - Phi(s_t); Phi is the residual LP bound, 0 when terminal.

    The sum telescopes to objective - Phi(s_0), an instance constant, so (gamma=1)
    every policy's return shifts equally. Each step is charged the upper-bound loss
    its choice causes: wasted capacity is penalized immediately, not at episode end.
    """
    return (rewards + potentials[1:] - potentials[:-1]) * active


def generalized_advantage(rewards, values, dones, active, gamma, gae_lambda):
    """[T,B] complete episodes; padding and true terminations never bootstrap."""
    advantages = torch.zeros_like(rewards)
    next_value = torch.zeros_like(values[0])
    gae = torch.zeros_like(next_value)
    for t in reversed(range(rewards.shape[0])):
        continuation = (~dones[t]).float()
        delta = rewards[t] + gamma * next_value * continuation - values[t]
        gae = (delta + gamma * gae_lambda * continuation * gae) * active[t]
        advantages[t] = gae
        next_value = values[t]
    return advantages, advantages + values


@torch.no_grad()
def collect_rollout(model, data):
    env = MKPEnv(*data)
    records = {key: [] for key in ("remaining", "selected", "actions", "log_probs",
                                  "values", "rewards", "dones", "active", "potentials")}
    model.eval()  # no dropout anywhere: collection and PPO recomputation agree
    combination_sum = data[0].new_zeros(len(COMBINATION_METRICS), dtype=torch.float32)
    transition_count = data[0].new_zeros((), dtype=torch.float32)
    theta_sum = data[0].new_zeros(11,dtype=torch.float32)
    for _ in range(data[0].shape[1]):
        active = env.state.active
        if not active.any():
            break
        policy, value, theta, _ = model(env.state)
        # The LP solve is memoized on this state; the potential reuses it.
        records["potentials"].append(lp_potential(env.state, model.feature_config))
        combination = torch.cat((theta.statistics()[:, -2:], decoder_statistics(policy)), -1)
        combination_sum += (combination * active[:, None]).sum(0)
        transition_count += active.sum()
        theta_sum += (theta.statistics()*active[:,None]).sum(0)
        actions = policy.choose(stochastic=True)
        records["remaining"].append(env.state.remaining.clone())
        records["selected"].append(env.state.selected.clone())
        records["actions"].append(actions)
        records["log_probs"].append(policy.log_prob(actions))
        records["values"].append(value)
        records["active"].append(active)
        reward, done = env.step(actions)
        records["rewards"].append(reward)
        records["dones"].append(done)
    if not records["actions"]:
        raise ValueError("Synthetic batch has no feasible item; adjust the generator")
    if env.state.active.any():
        raise RuntimeError("Rollout exceeded N steps")
    # Terminal rows have no feasible item, so their LP bound is exactly zero.
    records["potentials"].append(torch.zeros_like(records["potentials"][0]))
    result = {k: torch.stack(v) for k, v in records.items()}
    result["root_mu"], result["root_ub"] = env.state._root
    from mkp_v6exec import THETA_STATISTICS
    result["theta_statistics"] = dict(zip(THETA_STATISTICS,(theta_sum/transition_count.clamp_min(1)).tolist()))
    result["lp_bound_initial"] = records["potentials"][0]
    result["objective"] = env.objective
    result["utilization"] = (1 - env.state.remaining / env.state.capacities).mean()
    result["combination_metrics"] = dict(zip(COMBINATION_METRICS,
                                              (combination_sum / transition_count.clamp_min(1)).tolist()))
    return result


class PPOUpdate:
    def __init__(self, config, device):
        self.config, self.device = config, device

    def ppo_update(self, model, optimizer, data, rollout, update=0):
        c = self.config
        # No parameter update has occurred since collection: this is the frozen
        # behavior policy. Branches never enter the PPO buffer or critic targets.
        ranking, ranking_stats = collect_ranking_batch(model, data, rollout, c,
            seed=c.seed + 1000003 * update + data[2].shape[-1])
        rewards = (shaped_rewards(rollout["rewards"], rollout["potentials"], rollout["active"].float())
                   if c.lp_reward_shaping else rollout["rewards"])
        advantages, returns = generalized_advantage(rewards, rollout["values"],
            rollout["dones"], rollout["active"], c.gamma, c.gae_lambda)
        times, batches = rollout["active"].nonzero(as_tuple=True)
        ppo_began = time.perf_counter()
        state_bank, cache_stats = build_rollout_cache(model, data, rollout, times, batches, c)
        if ranking is not None and model.reuse_state_features:
            prepare_state(ranking.state, model.feature_config)
        adv = advantages[times, batches]
        normalized_adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)
        targets = returns[times, batches]
        old_values = rollout["values"][times, batches]
        old_log_probs = rollout["log_probs"][times, batches]
        actions = rollout["actions"][times, batches]
        model.train()
        metric_names = ("policy_loss", "value_loss", "constraint_aux_loss", "entropy",
                        "approx_kl", "clip_fraction", "grad_norm")
        sums = torch.zeros(len(metric_names), device=self.device)
        minibatches, observations, stopped = 0, 0, False
        ranking_sums = torch.zeros(2, device=self.device)
        ranking_updates, max_ranking_kl = 0, 0.
        for _ in range(c.ppo_epochs):
            order = torch.randperm(times.numel(), device=self.device)
            for ids in order.split(c.minibatch_size):
                if state_bank is not None:
                    state = state_bank.index(ids)
                else:
                    b, t = batches[ids], times[ids]
                    state = MKPState(data[0][b], data[1][b], data[2][b],
                                     rollout["remaining"][t, b], rollout["selected"][t, b], _root=(rollout["root_mu"][b], rollout["root_ub"][b]))
                policy, values, _, constraint_logits = model(state)
                log_ratio = policy.log_prob(actions[ids]) - old_log_probs[ids]
                ratio = log_ratio.exp()
                approximate_kl = ((ratio - 1) - log_ratio).mean()
                if not torch.isfinite(approximate_kl):
                    raise FloatingPointError("Non-finite PPO probability ratio")
                full_ranking_kl = ranking_kl(model, ranking).item() if ranking is not None else 0.
                if not math.isfinite(full_ranking_kl):
                    raise FloatingPointError("Non-finite ranking KL")
                max_ranking_kl = max(max_ranking_kl, full_ranking_kl)
                if max(approximate_kl.item(), full_ranking_kl) > c.target_kl:
                    stopped = True
                    break
                surrogate = ratio * normalized_adv[ids]
                clipped = ratio.clamp(1 - c.clip_ratio, 1 + c.clip_ratio) * normalized_adv[ids]
                policy_loss = -torch.minimum(surrogate, clipped).mean()
                clipped_values = old_values[ids] + (values - old_values[ids]).clamp(-c.value_clip, c.value_clip)
                value_loss = 0.5 * torch.maximum((values - targets[ids]).square(),
                                                (clipped_values - targets[ids]).square()).mean()
                constraint_loss = F.binary_cross_entropy_with_logits(constraint_logits, state.feasible.float())
                entropy = policy.normalized_entropy().mean()
                loss = (policy_loss + c.value_coefficient * value_loss
                        + c.constraint_aux_coefficient * constraint_loss - c.entropy_coefficient * entropy)
                if ranking is not None:
                    rank_ids = torch.randperm(ranking.left.numel(), device=self.device)[:c.ranking_minibatch_size]
                    rank_loss, _, rank_accuracy = ranking_objective(model, ranking.index(rank_ids), c)
                    loss = loss + c.ranking_coefficient * rank_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite PPO loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c.max_grad_norm,
                                                          error_if_nonfinite=True)
                optimizer.step()
                if ranking is not None:
                    ranking_sums += torch.stack((rank_loss.detach(), rank_accuracy.detach()))
                    ranking_updates += 1
                metrics = (policy_loss, value_loss, constraint_loss, entropy, approximate_kl,
                           ((ratio - 1).abs() > c.clip_ratio).float().mean(), grad_norm)
                sums += torch.stack([val.detach() for val in metrics]) * ids.numel()
                observations += ids.numel()
                minibatches += 1
            if stopped:
                break
        variance = targets.var(unbiased=False)
        explained = (1 - (targets - old_values).var(unbiased=False) / variance).item() if variance > 1e-12 else float("nan")
        result = dict(zip(metric_names, (sums / max(1, observations)).tolist()))
        result.update(minibatches=minibatches, kl_stopped=int(stopped), explained_variance=explained,
                      shaped_return=rewards.sum(0).mean().item())
        if ranking is not None:
            final_kl = ranking_kl(model, ranking).item()
            with torch.no_grad():
                post_accuracy = float(ranking_objective(model, ranking, c)[2])
            if not math.isfinite(final_kl):
                raise FloatingPointError("Non-finite final ranking KL")
            max_ranking_kl = max(max_ranking_kl, final_kl)
        else:
            post_accuracy = None
        result.update(ranking_stats, **dict(zip(("ranking_loss", "ranking_accuracy"),
                      (ranking_sums / max(1, ranking_updates)).tolist())),
                      ranking_kl=max_ranking_kl, ranking_minibatches=ranking_updates)
        result.update(cache_stats, ppo_seconds=time.perf_counter() - ppo_began)
        result["ranking_post_accuracy"] = post_accuracy
        # Observation only: a fitted score sign is not evidence that a frozen
        # continuation label survives the policy update. Re-evaluate terminal
        # returns periodically, outside PPO/critic targets and outside autograd.
        result.update(ranking_return_retention=None, ranking_return_sign_retention=None,
                      ranking_post_greedy_aligned_return_gap=None, ranking_diagnostic_trajectories=0)
        if ranking is not None and update % 25 == 0:
            was_training = model.training
            try:
                model.eval()
                with torch.no_grad():
                    generator = torch.Generator(device=self.device).manual_seed(c.seed + 7000003 * update + 99)
                    greedy = paired_terminal_returns(model, ranking.state, ranking.left, ranking.right, 1, False)
                    sampled = paired_terminal_returns(model, ranking.state, ranking.left, ranking.right,
                                                      c.ranking_repeats, True, generator)
                    delta, _, keep = return_targets(greedy, sampled, c)
                    same_sign = delta * ranking.delta > 0
                    result["ranking_return_sign_retention"] = same_sign.float().mean().item()
                    result["ranking_return_retention"] = (same_sign & keep).float().mean().item()
                    result["ranking_post_greedy_aligned_return_gap"] = ((greedy[:,0,0]-greedy[:,0,1]) * ranking.delta.sign()).mean().item()
                    result["ranking_diagnostic_trajectories"] = ranking.left.numel() * 2 * (1+c.ranking_repeats)
            finally:
                model.train(was_training)
        return result
