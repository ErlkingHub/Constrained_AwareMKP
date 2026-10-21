"""
mkp_Trainer.py
==============
REINFORCE trainer with a shared multi-sample baseline (POMO-style, Kwon et al. 2020).

For each training instance K trajectories are sampled; the baseline of a trajectory is the
mean reward of the K trajectories of the same instance, so no critic network is needed:

    A_k = R_k - mean_k' R_k'          (optionally divided by the batch std)
    L   = - E[ A_k * log pi(tau_k) ] - entropy_coef * H

Reward (see mkp_env.MKPEnv.get_reward):  profit / LP-bound  -  reject_penalty * #rejected / N

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

    def log(self, row: dict):
        new = self.fields is None
        if new:
            self.fields = list(row.keys())
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)


class MKPTrainer:
    def __init__(self, cfg: Config, device: torch.device):
        self.cfg, self.device = cfg, device
        tc = cfg.train
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
        g = torch.Generator().manual_seed(20240607)
        self.val_sets = []
        for n in tc.val_sizes:
            for m in tc.val_dims:
                inst = generate_instances(tc.val_instances, n, m, ec.train_tightness, ec.weight_max,
                                          ec.profit_noise, device="cpu", generator=g).to(self.device)
                mu, ub = self.env.precompute(inst)
                base = greedy_dual_ratio(inst, mu)
                base_gap = float(((1 - base / ub) * 100).mean())
                self.val_sets.append({"name": f"N{n}_M{m}", "inst": inst, "pre": (mu, ub), "base_gap": base_gap})
        self.log("validation sets (gap to LP bound of the dual-ratio greedy reference): " +
                 ", ".join(f"{v['name']}={v['base_gap']:.3f}%" for v in self.val_sets))

    # ------------------------------------------------------------------ checkpoints
    def _state(self):
        return {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(), "epoch": self.epoch,
                "global_step": self.global_step, "best_val": self.best_val,
                "config": self.cfg.to_dict(), "run_name": self.run_name}

    def save(self, is_best: bool):
        st = self._state()
        last = os.path.join(self.ckpt_dir, "last.pt")
        torch.save(st, last)
        shutil.copyfile(last, os.path.join(self.ckpt_root, "last.pt"))
        if is_best:
            best = os.path.join(self.ckpt_dir, "best.pt")
            shutil.copyfile(last, best)
            shutil.copyfile(last, os.path.join(self.ckpt_root, "best.pt"))

    def _load(self, path, full: bool):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        if full:
            self.optimizer.load_state_dict(ck["optimizer"])
            self.scheduler.load_state_dict(ck["scheduler"])
            self.epoch = ck.get("epoch", 0)
            self.global_step = ck.get("global_step", 0)
            self.best_val = ck.get("best_val", float("inf"))
        print(f"[trainer] loaded {'checkpoint' if full else 'weights'} from {path}")

    # ------------------------------------------------------------------ training
    def train_step(self):
        tc = self.cfg.train
        n = random.choice(self.cfg.env.train_sizes)
        m = random.choice(self.cfg.env.train_dims)
        B, K = tc.batch_size, tc.num_samples
        inst = self.env.generate(B, n, m, device="cpu").to(self.device)
        mu, ub = self.env.precompute(inst)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        per_chunk = max(1, tc.max_rollouts_per_chunk // K)
        agg = {"loss": 0.0, "reward": 0.0, "best_reward": 0.0, "gap_ub": 0.0, "reject_rate": 0.0,
               "entropy": 0.0, "kappa": 0.0}
        prim_w = None
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
                agg["kappa"] += out["kappa"] * frac
                pw = torch.tensor(out["prim_w"]) * frac
                prim_w = pw if prim_w is None else prim_w + pw
        gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip)
        if torch.isfinite(gn):
            self.optimizer.step()
        else:                                              # never let a bad batch poison the weights
            self.log(f"warning: non-finite grad norm at step {self.global_step}, update skipped")
            self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        self.global_step += 1
        agg.update({"N": n, "M": m, "grad_norm": float(gn), "lr": self.scheduler.get_last_lr()[0]})
        for k, name in enumerate(self.cfg.model.primitives):
            agg[f"w_{name}"] = float(prim_w[k])
        return agg

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        res = {}
        gaps = []
        for v in self.val_sets:
            out = self.model(self.env, v["inst"], num_samples=1, decode="greedy", amp=self.amp, pre=v["pre"])
            gap = float(((1 - out["profit"] / out["ub"]) * 100).mean())
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
                           f"| kappa {st['kappa']:.2f} | lr {st['lr']:.2e} | {time.time() - t0:.0f}s")
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
                self.log(f"== epoch {ep} validation (greedy, gap to LP bound): " +
                         ", ".join(f"{k}={x:.4f}" for k, x in val.items()) +
                         f" | best {self.best_val:.4f}{' *' if is_best else ''} | epoch time {time.time() - t0:.0f}s")
            self.epoch = ep + 1
            self.save(is_best)
        if self.tb is not None:
            self.tb.close()
        self.log(f"training finished. best val gap(LP) {self.best_val:.4f}% | "
                 f"checkpoint: {os.path.join(self.ckpt_root, 'best.pt')}")


def load_policy(path: str, device, cfg_override=None):
    """Load a trained policy + its config from a checkpoint (used by the benchmark)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_dict(ck.get("config", {}))
    if cfg_override is not None:
        cfg.eval = cfg_override.eval
        cfg.device = cfg_override.device
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck
