"""Full-episode PPO on synthetic MKP; Chu is never read by this module.

The actor produces deterministic resource prices / criterion weights. PPO's
sampled action is the item drawn from the fixed executor's induced distribution.
The independent critic predicts reward-to-go under the configured reward and
shaping, not item Q values. Terminal reward includes the selected prefix profit;
incremental reward predicts remaining profit. Both optimize the same total at gamma=1.
"""
import argparse
import csv
import itertools
import json
import logging
import random
import shutil
import time

import numpy as np
import torch

from mkp_config import ExperimentConfig, load_config
from mkp_data import generate_instance, solve_root_lp
from mkp_env import MKPState
from mkp_Model import ResourceController, ResidualCritic, FixedExecutor, run_episode
from mkp_utils import (atomic_torch_save, load_checkpoint, resolve_device, resolve_path,
                       runtime_manifest, save_source_snapshot, seed_everything,
                       sha256_file, timestamp, write_json)


def compute_gae(rewards, values, gamma=1.0, gae_lambda=1.0):
    """A complete episode: V(terminal)=0, no cross-instance/bootstrap leakage."""
    if len(rewards) != len(values):
        raise ValueError("Reward and value lengths differ")
    advantages = np.zeros(len(rewards), dtype=np.float64)
    carry = 0.0
    for i in reversed(range(len(rewards))):
        next_value = values[i + 1] if i + 1 < len(values) else 0.0
        delta = rewards[i] + gamma * next_value - values[i]
        carry = delta + gamma * gae_lambda * carry
        advantages[i] = carry
    return advantages, advantages + np.asarray(values, dtype=np.float64)


def normalized_reward(profit, final_objective, terminal, upper_bound, before_potential,
                      after_potential, config):
    base = (final_objective if terminal else 0.0) if config.reward_mode == "terminal" else profit
    base /= max(upper_bound, 1e-8)
    return base + config.shaping_coef * (config.gamma * after_potential - before_potential)


def _cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _append_csv(path, row):
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _generate(config, rng, prefix, index, spec=None):
    d = config.data
    if spec is None:
        spec = (int(rng.choice(d.train_n)), int(rng.choice(d.train_m)),
                str(rng.choice(d.train_distributions)), float(rng.choice(d.capacity_ratios)))
    n, m, family, ratio = spec
    instance_seed = int(rng.integers(0, 2**63 - 1))
    instance = generate_instance(np.random.default_rng(instance_seed), n, m, family, ratio,
                                 f"{prefix}_{index:08d}", coefficient_max=d.coefficient_max,
                                 correlation_noise=d.correlation_noise, correlation_offset=d.correlation_offset)
    record = {"id": instance.id, "instance_seed": instance_seed, "n": n, "m": m,
              "distribution": family, "capacity_ratio": ratio, "sha256": instance.fingerprint()}
    return instance, record


class MKPTrainer:
    def __init__(self, config, resume=None):
        self.config = config.validate()
        t = config.train
        torch.set_num_threads(t.torch_threads)
        self.device = resolve_device(t.device)
        seed_everything(t.seed)
        self.rng = np.random.default_rng(t.seed)
        self.actor = ResourceController(config.model).to(self.device)
        self.initial_actor = _cpu_state(self.actor)
        self.critic = ResidualCritic(config.model).to(self.device)
        self.executor = FixedExecutor(config.model)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=t.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=t.critic_lr)
        self.run_id = f"{timestamp()}_seed{t.seed}"
        self.run_dir = resolve_path(t.log_root) / t.version / self.run_id
        self.checkpoint_dir = resolve_path(t.checkpoint_root) / t.version / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=False)
        self.logger = logging.getLogger(f"mkp.{self.run_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        formatter = logging.Formatter("%(asctime)s %(message)s")
        for handler in (logging.StreamHandler(), logging.FileHandler(self.run_dir / "train.log", encoding="utf-8")):
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.start_update = 0
        self.best_score = -float("inf")
        self.best_update = 0
        self.total_decisions = 0
        self.optimizer_steps = 0
        self.parent_checkpoint = None
        self.validation = []
        val_rng = np.random.default_rng(config.data.validation_seed)
        cells = list(itertools.product(config.data.train_n, config.data.train_m,
                                      config.data.train_distributions, config.data.capacity_ratios))
        validation_records = []
        for i in range(config.data.validation_instances):
            instance, record = _generate(config, val_rng, "validation", i, cells[i % len(cells)])
            self.validation.append((instance, solve_root_lp(instance)))
            validation_records.append(record)
        write_json(self.run_dir / "validation_instances.json", validation_records)
        if resume:
            self._restore(resume, validation_records)
        manifest = runtime_manifest(config.to_dict())
        manifest.update({"run_id": self.run_id, "parent_checkpoint": self.parent_checkpoint,
                         "selection_rule": "maximum mean objective/root_LP_upper_bound on fixed synthetic validation",
                         "validation_instances": validation_records,
                         "validation_root_lp_seconds": sum(lp.solve_seconds for _, lp in self.validation),
                         "evidence_status": "implementation; no convergence or generalization claim",
                         "credit_assignment": "complete episodes; terminal bootstrap zero; independent critic",
                         "clock_boundary": "update includes generation, LP, rollout and optimization; validation policy time separate; validation root LPs solved once at initialization"})
        write_json(self.run_dir / "manifest.json", manifest)
        write_json(self.run_dir / "config.json", config.to_dict())
        save_source_snapshot(self.run_dir / "source")
        self.validation_records = validation_records
        self.logger.info("run=%s device=%s checkpoints=%s", self.run_id, self.device, self.checkpoint_dir)

    def _restore(self, path, validation_records):
        path = resolve_path(path)
        payload = load_checkpoint(path, "cpu")
        restored = ExperimentConfig.from_dict(payload["config"]).to_dict()
        requested = self.config.to_dict()
        # Budget and hardware may change on resume; the experiment may not.
        for key in ("updates", "device"):
            restored["train"][key] = requested["train"][key]
        if restored != requested:
            raise ValueError("Resume config differs from checkpoint; only updates/device may change")
        if payload.get("validation_instances") != validation_records:
            raise ValueError("Resume requires identical synthetic validation instances")
        self.actor.load_state_dict(payload["actor"])
        self.initial_actor = payload["initial_actor"]
        self.critic.load_state_dict(payload["critic"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self.start_update = int(payload["update"])
        self.best_score = float(payload["best_score"])
        self.best_update = int(payload["best_update"])
        self.total_decisions = int(payload["total_decisions"])
        self.optimizer_steps = int(payload["optimizer_steps"])
        self.rng.bit_generator.state = payload["rng"]["generator"]
        random.setstate(payload["rng"]["python"])
        np.random.set_state(payload["rng"]["numpy"])
        torch.set_rng_state(payload["rng"]["torch"].cpu())
        if self.device.type == "cuda" and payload["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
        old_best = path.parent / "best.pt"
        if not old_best.is_file():
            raise FileNotFoundError(f"Resume needs the sibling synthetic-selected best.pt: {old_best}")
        shutil.copy2(old_best, self.checkpoint_dir / "best.pt")
        self.parent_checkpoint = {"path": str(path), "sha256": sha256_file(path)}

    @torch.no_grad()
    def collect(self, update):
        self.actor.eval()
        self.critic.eval()
        records, episode_ratios, lengths = [], [], []
        raw_advantages, returns, credits = [], [], [[], [], []]
        lp_seconds = 0.0
        t = self.config.train
        for episode in range(t.episodes_per_update):
            instance, provenance = _generate(self.config, self.rng, f"train_seed{t.seed}",
                                             (update - 1) * t.episodes_per_update + episode)
            lp = solve_root_lp(instance)
            lp_seconds += lp.solve_seconds
            state = MKPState(instance, lp, self.device)
            ep_records, rewards, values = [], [], []
            while not state.done:
                snapshot = state.clone()
                theta = self.actor(snapshot)
                dist = self.executor.distribution(snapshot, theta)
                action = dist.sample()
                value = float(self.critic(snapshot).item())
                before = float(state.potential()) if t.shaping_coef else 0.0
                profit = float(state.step(int(action.item())))
                after = float(state.potential()) if t.shaping_coef else 0.0
                rewards.append(normalized_reward(profit, state.objective, state.done,
                                                 lp.upper_bound, before, after, t))
                values.append(value)
                ep_records.append((snapshot, int(action.item()), float(dist.log_prob(action).item()), value))
            advantages, targets = compute_gae(rewards, values, t.gamma, t.gae_lambda)
            for i, value in enumerate(advantages):
                credits[min(2, 3 * i // max(1, len(advantages)))].append(abs(float(value)))
            records.extend(ep_records)
            raw_advantages.extend(advantages.tolist())
            returns.extend(targets.tolist())
            episode_ratios.append(state.objective / max(lp.upper_bound, self.config.model.eps))
            lengths.append(len(ep_records))
            provenance.update({"update": update, "objective": state.objective, "lp_upper_bound": lp.upper_bound,
                               "selected": state.selected, "episode_reward": float(sum(rewards)), "steps": len(ep_records)})
            _append_jsonl(self.run_dir / "episodes.jsonl", provenance)
        if not records:
            raise RuntimeError("All training instances had no feasible actions; change the synthetic capacity/size settings")
        advantages = torch.tensor(raw_advantages, device=self.device, dtype=torch.float32)
        adv_std = advantages.std(unbiased=False)
        # Across the full rollout, never normalize one sparse terminal episode alone.
        advantages = (advantages - advantages.mean()) / adv_std.clamp_min(self.config.model.eps)
        targets = torch.tensor(returns, device=self.device, dtype=torch.float32)
        self.total_decisions += len(records)
        stats = {"instances": len(lengths), "decisions": len(records), "mean_steps": float(np.mean(lengths)),
                 "train_objective_over_lp": float(np.mean(episode_ratios)),
                 "instance_return_std": float(np.std(episode_ratios)), "raw_advantage_std": float(adv_std),
                 "early_abs_advantage": float(np.mean(credits[0])) if credits[0] else 0.0,
                 "middle_abs_advantage": float(np.mean(credits[1])) if credits[1] else 0.0,
                 "late_abs_advantage": float(np.mean(credits[2])) if credits[2] else 0.0,
                 "lp_seconds": lp_seconds}
        return records, advantages, targets, stats

    def optimize(self, records, advantages, targets):
        t = self.config.train
        self.actor.train()
        self.critic.train()
        metrics = []
        stopped = False
        for epoch in range(t.ppo_epochs):
            order = self.rng.permutation(len(records))
            for start in range(0, len(order), t.minibatch_size):
                batch = order[start:start + t.minibatch_size]
                new_logps, old_logps, entropies, values = [], [], [], []
                for index in batch:
                    state, action, old_logp, _ = records[index]
                    dist = self.executor.distribution(state, self.actor(state))
                    new_logps.append(dist.log_prob(torch.tensor(action, device=self.device)))
                    entropies.append(dist.entropy())
                    values.append(self.critic(state).reshape(()))
                    old_logps.append(old_logp)
                log_ratio = torch.stack(new_logps) - torch.tensor(old_logps, device=self.device)
                ratio = log_ratio.exp()
                indices = torch.as_tensor(batch, device=self.device)
                approx_kl = ((ratio - 1) - log_ratio).mean()
                if not torch.isfinite(approx_kl):
                    raise FloatingPointError("Nonfinite PPO ratio / KL")
                # Stop before applying a minibatch whose stale-policy drift is too large.
                if float(approx_kl.detach()) > t.target_kl:
                    stopped = True
                    break
                surrogate = torch.minimum(ratio * advantages[indices],
                                          ratio.clamp(1 - t.clip_ratio, 1 + t.clip_ratio) * advantages[indices])
                policy_loss = -surrogate.mean()
                entropy = torch.stack(entropies).mean()
                value_loss = (torch.stack(values) - targets[indices]).square().mean()
                actor_loss = policy_loss - t.entropy_coef * entropy
                critic_loss = t.value_coef * value_loss
                if not torch.isfinite(actor_loss + critic_loss):
                    raise FloatingPointError("Nonfinite training loss")
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), t.max_grad_norm, error_if_nonfinite=True)
                self.actor_optimizer.step()
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), t.max_grad_norm, error_if_nonfinite=True)
                self.critic_optimizer.step()
                self.optimizer_steps += 1
                metrics.append([float(policy_loss.detach()), float(value_loss.detach()), float(entropy.detach()),
                                float(approx_kl.detach()), float(((ratio - 1).abs() > t.clip_ratio).float().mean()),
                                float(actor_norm)])
            if stopped:
                break
        mean = np.mean(metrics, axis=0).tolist() if metrics else [0.0] * 6
        result = dict(zip(("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "actor_grad_norm"), mean))
        result.update({"minibatch_updates": len(metrics), "kl_early_stop": int(stopped), "optimizer_steps_total": self.optimizer_steps})
        return result

    @torch.no_grad()
    def validate(self, update):
        self.actor.eval()
        ratios = []
        _sync(self.device)
        started = time.perf_counter()
        for instance, lp in self.validation:
            result = run_episode(self.actor, MKPState(instance, lp, self.device), mode="dynamic")
            ratio = result["objective"] / max(lp.upper_bound, self.config.model.eps)
            ratios.append(ratio)
            _append_jsonl(self.run_dir / "validation_solutions.jsonl", {"update": update, "id": instance.id,
                          "objective": result["objective"], "lp_upper_bound": lp.upper_bound,
                          "selected": result["selected"], "objective_over_lp": ratio})
        _sync(self.device)
        score = float(np.mean(ratios))
        _append_csv(self.run_dir / "validation.csv", {"update": update, "instances": len(ratios),
                    "objective_over_lp": score, "lp_gap_pct": 100 * (1 - score),
                    "worst_lp_gap_pct": 100 * (1 - min(ratios)), "seconds": time.perf_counter() - started})
        self.logger.info("validation update=%d mean LP-gap=%.6f%% (synthetic only)", update, 100 * (1 - score))
        return score

    def save(self, update, improved=False):
        payload = {"schema_version": 1, "config": self.config.to_dict(), "run_id": self.run_id,
                   "seed": self.config.train.seed, "update": update, "actor": _cpu_state(self.actor),
                   "initial_actor": self.initial_actor, "critic": _cpu_state(self.critic),
                   "actor_optimizer": self.actor_optimizer.state_dict(), "critic_optimizer": self.critic_optimizer.state_dict(),
                   "best_score": self.best_score, "best_update": self.best_update,
                   "total_decisions": self.total_decisions, "optimizer_steps": self.optimizer_steps,
                   "validation_instances": self.validation_records,
                   "selection_rule": "synthetic validation mean objective/root LP UB",
                   "source_manifest": str(self.run_dir / "manifest.json"),
                   "rng": {"generator": self.rng.bit_generator.state, "python": random.getstate(),
                           "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                           "cuda": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None}}
        last_path, best_path = self.checkpoint_dir / "last.pt", self.checkpoint_dir / "best.pt"
        atomic_torch_save(payload, last_path)
        if improved:
            atomic_torch_save(payload, best_path)
        metadata = {"run_id": self.run_id, "update": update, "best_update": self.best_update,
                    "best_score": self.best_score, "best_checkpoint": str(best_path), "last_checkpoint": str(last_path),
                    "best_sha256": sha256_file(best_path), "last_sha256": sha256_file(last_path),
                    "selection_rule": payload["selection_rule"], "run_dir": str(self.run_dir)}
        write_json(self.checkpoint_dir / "checkpoints.json", metadata)
        write_json(self.checkpoint_dir.parent / "latest.json", metadata)

    def train(self):
        try:
            return self._train()
        finally:
            for handler in list(self.logger.handlers):
                self.logger.removeHandler(handler)
                handler.close()

    def _train(self):
        t = self.config.train
        if self.start_update >= t.updates:
            raise ValueError(f"--updates is cumulative; use a target above checkpoint update {self.start_update}")
        if not self.parent_checkpoint:
            self.best_score = self.validate(0)
            self.save(0, improved=True)
        for update in range(self.start_update + 1, t.updates + 1):
            _sync(self.device)
            started = time.perf_counter()
            records, advantages, targets, stats = self.collect(update)
            rollout_done = time.perf_counter()
            optimization = self.optimize(records, advantages, targets)
            _sync(self.device)
            elapsed = time.perf_counter() - started
            row = {"update": update, **stats, **optimization, "total_decisions": self.total_decisions,
                   "rollout_seconds": rollout_done - started, "update_seconds": elapsed,
                   "decisions_per_second": len(records) / max(elapsed, 1e-8)}
            _append_csv(self.run_dir / "metrics.csv", row)
            self.logger.info("update=%d/%d instances=%d decisions=%d ratio=%.5f policy=%.5f value=%.5f sec=%.2f",
                             update, t.updates, stats["instances"], stats["decisions"],
                             stats["train_objective_over_lp"], optimization["policy_loss"], optimization["value_loss"], elapsed)
            improved = False
            if update % t.validate_every == 0 or update == t.updates:
                score = self.validate(update)
                if score > self.best_score:
                    self.best_score, self.best_update, improved = score, update, True
            if improved or update % t.checkpoint_every == 0 or update == t.updates:
                self.save(update, improved)
        write_json(self.run_dir / "completion.json", {"status": "budget_complete", "update": t.updates,
                   "best_update": self.best_update, "best_validation_score": self.best_score,
                   "convergence_claim": False, "benchmark_used_for_selection": False})
        self.logger.info("Finished. synthetic-selected checkpoint: %s", self.checkpoint_dir / "best.pt")
        return self.checkpoint_dir / "best.pt"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON overrides of mkp_config.py defaults")
    parser.add_argument("--resume", help="Trusted local last.pt; updates is a cumulative target")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--version")
    parser.add_argument("--write-default-config", metavar="PATH")
    args = parser.parse_args(argv)
    if args.resume:
        if args.config or args.seed is not None or args.version:
            parser.error("Resume restores full configuration; only --updates and --device can override it")
        config = ExperimentConfig.from_dict(load_checkpoint(resolve_path(args.resume), "cpu")["config"])
    else:
        config = load_config(args.config)
    for name in ("updates", "seed", "device", "version"):
        value = getattr(args, name)
        if value is not None:
            setattr(config.train, name, value)
    config.validate()
    if args.write_default_config:
        write_json(resolve_path(args.write_default_config), config.to_dict())
        return
    MKPTrainer(config, args.resume).train()


if __name__ == "__main__":
    main()
