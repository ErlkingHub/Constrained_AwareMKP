"""
mkp_Trainer.py
==============
REINFORCE trainer with a shared multi-sample baseline (POMO-style, Kwon et al. 2020).

For each training instance K trajectories are sampled; the baseline of a trajectory is the
mean reward of the K trajectories of the same instance, so no critic network is needed:

    A_k = R_k - mean_k' R_k'          (optionally divided by the batch std)
    L   = - E[ A_k * log pi(tau_k) ] - entropy_coef * H

Reward (hard-mask main method): profit / LP-bound.

Logs   : runs/<timestamp>/{config.json, log.txt, train_log.csv, val_log.csv, tensorboard events}
Ckpts  : checkpoints/<timestamp>/{last.pt, best.pt}  and  checkpoints/{last.pt, best.pt}
         (the latter two are what Benchmark_zeroShot.py loads by default)
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
import time
from datetime import datetime

import numpy as np
import torch

from mkp_config import Config, resolve_path, config_from_dict
from mkp_env import MKPEnv, MKPInstances, generate_instances, greedy_dual_ratio
from mkp_Model import build_model, count_parameters
from mkp_artifacts import instance_sha, sha256_file, source_provenance, write_json
from mkp_reference import milp_reference


CHECKPOINT_SCHEMA = "weak_decoder_t1_v1"


def _check_checkpoint(ck, expected_model=None):
    """Reject legacy item-residual models instead of silently discarding config fields."""
    model_cfg = ck.get("config", {}).get("model", {})
    legacy = {"residual_max", "primitives", "init_primitive", "init_primitive_logit",
              "kappa_init", "kappa_min", "price_log_range"}.intersection(model_cfg)
    legacy_weights = any(any(part in key for part in ("res_head", "prim_head", "kappa_head"))
                         for key in ck.get("model", {}))
    if legacy or legacy_weights:
        raise ValueError("Legacy delta/primitive checkpoint is incompatible with the T1 weak decoder; "
                         "use the original main_v0 code for the T0 diagnostic")
    if ck.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"Expected checkpoint schema {CHECKPOINT_SCHEMA!r}; got {ck.get('schema')!r}")
    if expected_model is not None:
        expected = vars(expected_model)
        differences = [name for name in expected if model_cfg.get(name) != expected[name]]
        if differences:
            raise ValueError("Checkpoint model config differs from requested training config: " +
                             ", ".join(differences))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _CSVLogger:
    def __init__(self, path):
        self.path = path
        self.fields = None
        if os.path.isfile(path) and os.path.getsize(path):
            with open(path, newline="", encoding="utf-8") as stream:
                self.fields = next(csv.reader(stream))

    def log(self, row: dict):
        new = self.fields is None
        if new:
            self.fields = list(row.keys())
        elif set(row) != set(self.fields):
            raise ValueError(f"CSV schema changed while appending to {self.path}")
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)


class MKPTrainer:
    def __init__(self, cfg: Config, device: torch.device):
        self.cfg, self.device = cfg, device
        tc = cfg.train
        if tc.batch_size < 1 or tc.num_samples < 2 or tc.steps_per_epoch < 1:
            raise ValueError("REINFORCE needs batch_size>=1, num_samples>=2 and steps_per_epoch>=1")
        if tc.val_instances < 1 or not tc.val_sizes or not tc.val_dims or tc.val_every < 1:
            raise ValueError("A nonempty synthetic validation set and val_every>=1 are required")
        if tc.validation_reference not in ("milp", "lp"):
            raise ValueError("train.validation_reference must be milp or lp")
        self.selection_rule = ("minimum mean single-trajectory greedy gap (%) on fixed generated "
                               f"validation instances to {tc.validation_reference} reference; "
                               "benchmark data excluded; ties retain the earlier checkpoint")
        self.source = source_provenance()
        self.run_name = tc.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = resolve_path(os.path.join(tc.runs_dir, self.run_name))
        self.ckpt_root = resolve_path(tc.ckpt_dir)
        self.ckpt_dir = os.path.join(self.ckpt_root, self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        cfg.save(os.path.join(self.run_dir, "config.json"))

        self.env = MKPEnv(cfg.env)
        self.model = build_model(cfg).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
        self.total_steps = tc.epochs * tc.steps_per_epoch
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_lambda)
        self.epoch, self.global_step, self.best_val = 0, 0, float("inf")
        self.inherited_best, self.best_checkpoint_path = None, None
        self.amp = cfg.device.amp and device.type == "cuda"

        self.train_csv = _CSVLogger(os.path.join(self.run_dir, "train_log.csv"))
        self.val_csv = _CSVLogger(os.path.join(self.run_dir, "val_log.csv"))
        self.tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(self.run_dir)
        except Exception:
            pass

        if tc.resume:
            self._load(resolve_path(tc.resume), full=True)
        elif tc.init_from:
            self._load(resolve_path(tc.init_from), full=False)
        self._build_val_set()
        write_json(os.path.join(self.run_dir, "meta.json"), {
            "schema": CHECKPOINT_SCHEMA, "config": cfg.to_dict(), "source": self.source,
            "seed": cfg.device.seed, "selection_rule": self.selection_rule,
            "validation_manifest": self.validation_manifest,
            "inherited_best": self.inherited_best,
            "resume": self._input_checkpoint_meta(tc.resume),
            "init_from": self._input_checkpoint_meta(tc.init_from),
        })
        self.log(f"run {self.run_name} | device {device} | params {count_parameters(self.model):,}")
        self.log(f"logs -> {self.run_dir} | checkpoints -> {self.ckpt_dir}")

    # ------------------------------------------------------------------ utils
    def _lr_lambda(self, step):
        tc = self.cfg.train
        if step < tc.warmup_steps:
            return (step + 1) / max(tc.warmup_steps, 1)
        prog = min(1.0, (step - tc.warmup_steps) / max(self.total_steps - tc.warmup_steps, 1))
        floor = tc.lr_min / tc.lr
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog))

    def log(self, msg: str):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(os.path.join(self.run_dir, "log.txt"), "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _build_val_set(self):
        tc, ec = self.cfg.train, self.cfg.env
        g = torch.Generator().manual_seed(tc.val_seed)
        self.val_sets = []
        self.validation_manifest = []
        for n in tc.val_sizes:
            for m in tc.val_dims:
                generated = generate_instances(tc.val_instances, n, m, ec.train_tightness, ec.weight_max,
                                               ec.profit_noise, device="cpu", generator=g)
                inst = generated.to(self.device)
                mu, ub = self.env.precompute(inst)
                manifest = {"name": f"N{n}_M{m}", "data_sha256": instance_sha(generated),
                            "seed": tc.val_seed, "count": inst.B,
                            "reference": tc.validation_reference}
                if tc.validation_reference == "milp":
                    reference = milp_reference(generated, tc.val_milp_time_limit, tc.val_mip_rel_gap,
                                               tc.reference_cache, log=self.log)
                    ref_obj = torch.as_tensor(reference["obj"], dtype=torch.float32, device=self.device)
                    manifest.update({"reference_path": reference["path"],
                                     "reference_sha256": reference["sha256"],
                                     "proven_optimal": int(reference["optimal"].sum()),
                                     "solver_settings": reference["settings"]})
                else:
                    ref_obj = ub
                base = greedy_dual_ratio(inst, mu)
                base_gap = float(((1 - base / ref_obj) * 100).mean())
                self.val_sets.append({"name": manifest["name"], "inst": inst, "pre": (mu, ub),
                                      "reference": ref_obj, "base_gap": base_gap})
                self.validation_manifest.append(manifest)
        self.log(f"synthetic validation sets (dual-ratio greedy gap to {tc.validation_reference} reference): " +
                 ", ".join(f"{v['name']}={v['base_gap']:.3f}%" for v in self.val_sets))

    @staticmethod
    def _input_checkpoint_meta(path):
        if not path:
            return None
        resolved = resolve_path(path)
        return {"path": resolved, "sha256": sha256_file(resolved)}

    # ------------------------------------------------------------------ checkpoints
    def _state(self):
        return {"schema": CHECKPOINT_SCHEMA,
                "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(), "epoch": self.epoch,
                "global_step": self.global_step, "best_val": self.best_val,
                "config": self.cfg.to_dict(), "run_name": self.run_name,
                "selection_rule": self.selection_rule, "source": self.source,
                "validation_manifest": self.validation_manifest,
                "inherited_best": self.inherited_best,
                "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}

    def save(self, is_best: bool):
        st = self._state()
        last = os.path.join(self.ckpt_dir, "last.pt")
        torch.save(st, last)
        shutil.copyfile(last, os.path.join(self.ckpt_root, "last.pt"))
        meta = {"schema": CHECKPOINT_SCHEMA, "sha256": sha256_file(last), "epoch": self.epoch,
                "global_step": self.global_step, "best_val": self.best_val,
                "selection_rule": self.selection_rule, "validation_manifest": self.validation_manifest}
        write_json(last + ".meta.json", meta)
        write_json(os.path.join(self.ckpt_root, "last.pt.meta.json"), meta)
        if is_best:
            best = os.path.join(self.ckpt_dir, "best.pt")
            shutil.copyfile(last, best)
            shutil.copyfile(last, os.path.join(self.ckpt_root, "best.pt"))
            write_json(best + ".meta.json", meta)
            write_json(os.path.join(self.ckpt_root, "best.pt.meta.json"), meta)
            self.best_checkpoint_path = best

    def _inherit_best(self, checkpoint, resumed_path):
        """Keep the selected historical best when a resumed run has no improvement."""
        candidate = os.path.join(os.path.dirname(os.path.abspath(resumed_path)), "best.pt")
        try:
            if not os.path.isfile(candidate):
                raise ValueError("the resumed checkpoint has no sibling best.pt")
            selected = torch.load(candidate, map_location="cpu", weights_only=False)
            _check_checkpoint(selected, expected_model=self.cfg.model)
            same = (selected.get("run_name") == checkpoint.get("run_name") and
                    selected.get("best_val") == checkpoint.get("best_val") and
                    selected.get("epoch", -1) <= checkpoint.get("epoch", -1) and
                    selected.get("validation_manifest") == checkpoint.get("validation_manifest"))
            if not same:
                raise ValueError("sibling best.pt does not match resumed run/epoch/value/validation")
            digest = sha256_file(candidate)
            sidecar = candidate + ".meta.json"
            if os.path.isfile(sidecar):
                with open(sidecar, encoding="utf-8") as stream:
                    recorded = json.load(stream)
                if recorded.get("sha256") != digest:
                    raise ValueError("sibling best.pt hash disagrees with its metadata")
            local = os.path.join(self.ckpt_dir, "best.pt")
            if os.path.abspath(local) != os.path.abspath(candidate):
                shutil.copyfile(candidate, local)
            self.inherited_best = {"available": True, "source_path": candidate,
                                   "sha256": digest, "source_run": selected.get("run_name"),
                                   "epoch": selected.get("epoch"), "best_val": selected["best_val"],
                                   "local_path": local}
            write_json(local + ".meta.json", {"schema": CHECKPOINT_SCHEMA, "sha256": digest,
                                              "inherited_best": self.inherited_best,
                                              "selection_rule": selected.get("selection_rule")})
            self.best_checkpoint_path = local
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            self.inherited_best = {"available": False, "source_path": candidate, "reason": str(exc)}
            self.log(f"inherited best checkpoint unavailable: {exc}; a new best requires validation improvement")

    def _load(self, path, full: bool):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        _check_checkpoint(ck, expected_model=self.cfg.model)
        self.model.load_state_dict(ck["model"])
        if full:
            old_train = ck["config"]["train"]
            selection_fields = ("validation_reference", "val_seed", "val_instances", "val_sizes", "val_dims",
                                "val_milp_time_limit", "val_mip_rel_gap")
            changed = [name for name in selection_fields if old_train.get(name) != getattr(self.cfg.train, name)]
            old_env = ck["config"]["env"]
            changed.extend("env." + name for name in ("train_tightness", "weight_max", "profit_noise",
                                                       "dual_iters", "dual_lr", "feasibility_mode",
                                                       "reject_penalty", "soft_mask_beta")
                           if old_env.get(name) != getattr(self.cfg.env, name))
            if changed:
                raise ValueError("Resume cannot reuse best_val after changing validation: " + ", ".join(changed))
            self.optimizer.load_state_dict(ck["optimizer"])
            self.scheduler.load_state_dict(ck["scheduler"])
            self.epoch = ck.get("epoch", 0)
            self.global_step = ck.get("global_step", 0)
            self.best_val = ck.get("best_val", float("inf"))
            self._inherit_best(ck, path)
            rng = ck.get("rng")
            if rng:
                random.setstate(rng["python"])
                np.random.set_state(rng["numpy"])
                torch.set_rng_state(rng["torch"].cpu())
                if torch.cuda.is_available() and rng.get("cuda") is not None:
                    torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
        print(f"[trainer] loaded {'checkpoint' if full else 'weights'} from {path}")

    # ------------------------------------------------------------------ training
    def train_step(self):
        tc = self.cfg.train
        n = random.choice(self.cfg.env.train_sizes)
        m = random.choice(self.cfg.env.train_dims)
        B, K = tc.batch_size, tc.num_samples
        generated = self.env.generate(B, n, m, device="cpu")
        data_sha = instance_sha(generated)
        inst = generated.to(self.device)
        mu, ub = self.env.precompute(inst)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        per_chunk = max(1, tc.max_rollouts_per_chunk // K)
        agg = {"loss": 0.0, "reward": 0.0, "best_reward": 0.0, "gap_ub": 0.0, "reject_rate": 0.0,
               "entropy": 0.0, "u_abs_mean": 0.0, "alpha_mean": 0.0, "temperature": 0.0}
        for s in range(0, B, per_chunk):
            e = min(B, s + per_chunk)
            sub = MKPInstances(inst.p[s:e], inst.r[s:e], inst.c[s:e])
            out = self.model(self.env, sub, num_samples=K, decode="sample", amp=self.amp,
                             pre=(mu[s:e], ub[s:e]))
            R = out["reward"].view(e - s, K)
            adv = R - R.mean(1, keepdim=True)
            if tc.adv_normalize:
                adv = adv / (adv.std() + 1e-8)
            ll = out["log_likelihood"].view(e - s, K)
            loss = -(adv.detach() * ll).mean() - tc.entropy_coef * out["entropy"].mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite REINFORCE loss at step {self.global_step}")
            frac = (e - s) / B
            (loss * frac).backward()
            with torch.no_grad():
                profit = out["profit"].view(e - s, K)
                ubv = out["ub"].view(e - s, K)
                agg["loss"] += loss.item() * frac
                agg["reward"] += R.mean().item() * frac
                agg["best_reward"] += R.max(1).values.mean().item() * frac
                agg["gap_ub"] += ((1 - profit.max(1).values / ubv[:, 0]) * 100).mean().item() * frac
                agg["reject_rate"] += (out["n_reject"] / max(out["steps"], 1)).mean().item() * frac
                agg["entropy"] += out["entropy"].mean().item() * frac
                for key in ("u_abs_mean", "alpha_mean", "temperature"):
                    agg[key] += float(out[key]) * frac
        gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip)
        finite_gradients = bool(torch.isfinite(gn))
        if finite_gradients:
            self.optimizer.step()
        else:                                              # never let a bad batch poison the weights
            self.log(f"warning: non-finite grad norm at step {self.global_step}, update skipped")
            self.optimizer.zero_grad(set_to_none=True)
        if finite_gradients:
            self.scheduler.step()
        self.global_step += 1
        agg.update({"N": n, "M": m, "grad_norm": float(gn), "finite_gradients": int(finite_gradients),
                    "lr": self.scheduler.get_last_lr()[0], "data_sha256": data_sha})
        return agg

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        res = {}
        gaps = []
        for v in self.val_sets:
            out = self.model(self.env, v["inst"], num_samples=1, decode="greedy", amp=self.amp, pre=v["pre"])
            gap = float(((1 - out["profit"] / v["reference"]) * 100).mean())
            res[f"gap_{v['name']}"] = gap
            res[f"rej_{v['name']}"] = float(out["n_reject"].mean())
            gaps.append(gap)
        res["gap_mean"] = float(np.mean(gaps))
        return res

    def fit(self):
        tc = self.cfg.train
        start_epoch = self.epoch
        for ep in range(start_epoch, tc.epochs):
            self.epoch = ep
            t0 = time.time()
            run = {}
            for it in range(tc.steps_per_epoch):
                st = self.train_step()
                for k, val in st.items():
                    if isinstance(val, (int, float)):
                        run[k] = run.get(k, 0.0) + val
                row = {"epoch": ep, "step": self.global_step, **st}
                self.train_csv.log(row)
                if self.tb is not None:
                    for k, val in st.items():
                        if isinstance(val, (int, float)) and k not in ("N", "M"):
                            self.tb.add_scalar(f"train/{k}", val, self.global_step)
                if (it + 1) % tc.log_every == 0:
                    msg = (f"ep {ep} it {it + 1}/{tc.steps_per_epoch} | N{st['N']} M{st['M']} "
                           f"| loss {st['loss']:+.4f} | R {st['reward']:.4f} | gapUB(best of K) "
                           f"{st['gap_ub']:.3f}% | rej {st['reject_rate']:.3f} | H {st['entropy']:.3f} "
                           f"| |u| {st['u_abs_mean']:.3f} | alpha {st['alpha_mean']:.3f} "
                           f"| T {st['temperature']:.3f} | grad {st['grad_norm']:.3f} "
                           f"| lr {st['lr']:.2e} | {time.time() - t0:.0f}s")
                    self.log(msg)
            is_best = False
            if (ep + 1) % tc.val_every == 0 or ep == tc.epochs - 1:
                val = self.validate()
                is_best = val["gap_mean"] < self.best_val
                if is_best:
                    self.best_val = val["gap_mean"]
                self.val_csv.log({"epoch": ep, "step": self.global_step, **val, "best": self.best_val})
                if self.tb is not None:
                    for k, x in val.items():
                        self.tb.add_scalar(f"val/{k}", x, self.global_step)
                self.log(f"== epoch {ep} synthetic validation (greedy, gap to {tc.validation_reference} reference): " +
                         ", ".join(f"{k}={x:.4f}" for k, x in val.items()) +
                         f" | best {self.best_val:.4f}{' *' if is_best else ''} | epoch time {time.time() - t0:.0f}s")
            self.epoch = ep + 1
            self.save(is_best)
        if self.tb is not None:
            self.tb.close()
        best_description = (self.best_checkpoint_path if self.best_checkpoint_path is not None else
                            "unavailable: no validated inherited file and no new validation improvement")
        self.log(f"training finished. best synthetic val gap({tc.validation_reference}) {self.best_val:.4f}% | "
                 f"checkpoint: {best_description}")


def load_policy(path: str, device, cfg_override=None):
    """Load a trained policy + its config from a checkpoint (used by the benchmark)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    _check_checkpoint(ck)
    cfg = config_from_dict(ck.get("config", {}))
    if cfg_override is not None:
        cfg.eval = cfg_override.eval
        cfg.device = cfg_override.device
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck
