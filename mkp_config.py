"""
mkp_config.py
=============
Central, configurable hyper-parameters for every module of the Constrained-Aware
MKP project (environment / model / trainer / zero-shot benchmark).

Usage
-----
    from mkp_config import get_config
    cfg = get_config()                       # defaults
    cfg = get_config(["--train.lr", "5e-5"])  # CLI-style overrides

Every field can be overridden on the command line of `train_ConstrainedMKP.py`
and `Benchmark_zeroShot.py` with `--<section>.<field> <value>`, e.g.

    python train_ConstrainedMKP.py --train.epochs 200 --model.embed_dim 192
    python Benchmark_zeroShot.py --eval.num_samples 512 --eval.ls_enable false

Lists are given comma separated: `--env.train_sizes 50,100,150`.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------------------
# Hardware
# --------------------------------------------------------------------------------------
@dataclass
class DeviceConfig:
    # Prefer GPU 1 where available; get_device falls back to GPU 0 or CPU.
    cuda_device: int = 1
    force_cpu: bool = False
    # bf16 autocast for the attention stacks (env / decoder / rewards always stay fp32).
    amp: bool = True
    seed: int = 1234


# --------------------------------------------------------------------------------------
# Dynamic MKP environment + random instance generator
# --------------------------------------------------------------------------------------
@dataclass
class EnvConfig:
    # Training instances follow the Chu & Beasley (1998) generator, i.e. the same
    # *distribution* as Benchmarks_Chu.mat (the benchmark instances themselves are never
    # seen in training -> zero-shot evaluation).
    train_sizes: List[int] = field(default_factory=lambda: [50, 100, 150])   # N (items)
    train_dims: List[int] = field(default_factory=lambda: [5, 10])           # M (resources)
    train_tightness: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])  # alpha
    weight_max: int = 1000           # r_ij ~ U{0..weight_max}
    profit_noise: float = 500.0      # p_j = sum_i r_ij / M + profit_noise * U(0,1)

    # The main method selects only feasible items; the other modes are named ablations.
    #   "perceive"  : NO feasibility mask. An infeasible pick is *rejected* by the env (the
    #                 item is discarded, capacities untouched) and penalised. Because
    #                 capacities only shrink, a rejected item could never fit again, so the
    #                 rejection is loss-free for solution quality; the agent must perceive
    #                 feasibility from the state to avoid wasting steps / penalties.
    #   "soft_mask" : additionally add -soft_mask_beta * violation to the logits (ablation).
    #   "hard_mask" : -inf mask on infeasible items (main method).
    feasibility_mode: str = "hard_mask"
    soft_mask_beta: float = 5.0
    reject_penalty: float = 0.1      # used only in perceive/soft_mask ablations

    # Batched Lagrangian-dual (LP relaxation) approximation used for features + bound.
    dual_iters: int = 300
    dual_lr: float = 0.05
    residual_lp_iters: int = 300          # 30 failed required residual LP accuracy checks
    residual_lp_lr: float = 0.02


# --------------------------------------------------------------------------------------
# Model: State encoder -> High-level strategy controller -> Low-level weak decoder
# --------------------------------------------------------------------------------------
@dataclass
class ModelConfig:
    executor: str = "g0g1"             # g0g1 | v6 | cert; legacy path unchanged
    theta_family: str = "g0g1"         # g0g1 | v6 | cert
    ctrl_residual_lp_features: bool = False
    v6_critic: bool = False             # independent critic/feasibility head, T8-B only
    ctrl_basis_features: bool = False   # T9: five bases + rank/core + global certificate
    v6_frozen_groups: List[str] = field(default_factory=list)
    lp_warm_start: bool = False         # reserved; cold solver retained, no speed optimization
    cert_weights: List[float] = field(default_factory=lambda: [.2]*5)
    cert_temperature: float = 0.2
    cert_temperature_min: float = 0.01
    cert_temperature_max: float = 2.0
    embed_dim: int = 128
    num_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.0
    # Static bipartite (item <-> resource) graph encoder, run once per instance.
    encoder_layers: int = 3
    # Per-step dual-axis Transformer of the high-level controller (re-encodes the
    # *current* state at every decision step -> step-wise state/constraint awareness).
    controller_layers: int = 2
    # Edge-feature MLP hidden size for the item-resource attention biases.
    edge_hidden: int = 32

    # G0+G1 by default; opt-in block group is prepared for gated T2 experiments.
    theta_groups: List[str] = field(default_factory=lambda: ["price", "alpha"])
    price_anchor: str = "root"
    lp_source: str = "root"
    lp_interval: int = 8
    lp_eta_init: float = 0.5
    alpha_init: float = 1.0
    u_max: float = 3.0                   # tanh bound before centering (sum_j u_j = 0)
    theta_update: str = "step"           # step | static (static must be trained separately)
    theta_interval: int = 1              # decision times 0,k,2k,...; T1-P dynamic uses 8
    block_gamma_init: float = 0.1
    block_chunk_size: int = 64
    score_normalization: bool = True    # T1-R feasible-set z-score; old checkpoints retain False
    temperature_mode: str = "fixed"    # one global scalar, independent of state and N
    temperature_init: float = 1.0 / 3.0
    untrained_random_heads: bool = False # True only for the untrained control arm
    price_floor: float = 0.05             # floor (relative to mean dual price) for non-binding dims
    grad_checkpoint: bool = True          # checkpoint the per-step controller (saves memory)
    theta_sigma_head: bool = False
    theta_sigma_u0: float = 0.05
    theta_sigma_alpha0: float = 0.3
    theta_sigma_dims: str = "ua"


# --------------------------------------------------------------------------------------
# Training (REINFORCE with shared multi-sample baseline)
# --------------------------------------------------------------------------------------
@dataclass
class TrainConfig:
    signal: str = "theta_pg"             # theta_pg | theta_maxk | item_pg | ppo_gae
    theta_pg_sigma_u: float = 0.05
    theta_pg_sigma_alpha: float = 0.3
    theta_pg_sigma_gamma: float = 0.3
    theta_pg_sigma_eta: float = 0.3
    theta_pg_antithetic: bool = True
    initial_validation: bool = True
    gate_early_stop: bool = False
    gate_margin_pp: float = 0.3
    epochs: int = 100
    steps_per_epoch: int = 250
    batch_size: int = 64                  # instances per step
    num_samples: int = 16                 # rollouts per instance (shared baseline, POMO-style)
    max_rollouts_per_chunk: int = 1024    # split B*K rollouts into chunks if larger (memory)
    lr: float = 1e-4
    lr_min: float = 1e-6
    warmup_steps: int = 500
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    entropy_coef: float = 0.01
    adv_normalize: bool = True

    # Fixed synthetic validation only; Chu instances never select checkpoints.
    validation_reference: str = "milp"   # milp | lp (LP is an explicit diagnostic)
    val_seed: int = 20240607
    val_noise_seed: int = 20261106        # fixed common random draws for max@K validation
    val_milp_time_limit: float = 10.0
    val_mip_rel_gap: float = 1e-9
    reference_cache: str = "cache"
    val_instances: int = 64               # per (N, M) pair
    val_sizes: List[int] = field(default_factory=lambda: [100])
    val_dims: List[int] = field(default_factory=lambda: [5, 10])
    val_every: int = 1                    # epochs

    log_every: int = 25                   # steps
    runs_dir: str = "runs"
    ckpt_dir: str = "checkpoints"
    resume: str = ""                      # path to a checkpoint to resume from ("" = scratch)
    init_from: str = ""                   # load weights only (fine-tune), "" = none
    run_name: str = ""                    # default: timestamp
    ppo_epochs: int = 3
    minibatch_size: int = 128
    clip_ratio: float = 0.2
    target_kl: float = 0.03
    gae_lambda: float = 0.95
    value_coefficient: float = 0.5
    value_clip: float = 0.2
    constraint_aux_coefficient: float = 0.05
    ranking_coefficient: float = 0.1
    ranking_states: int = 16
    ranking_pool_size: int = 64
    ranking_candidates: int = 8
    ranking_repeats: int = 4
    ranking_minibatch_size: int = 8
    ranking_min_gap: float = 0.001
    ranking_se_multiplier: float = 1.0
    ranking_return_scale: float = 0.02
    ranking_score_scale: float = 0.2
    ranking_greedy_weight: float = 0.5


# --------------------------------------------------------------------------------------
# Zero-shot benchmark
# --------------------------------------------------------------------------------------
@dataclass
class EvalConfig:
    mat_file: str = os.path.join("MKP_dataset", "Benchmarks_Chu.mat")
    dims: List[int] = field(default_factory=lambda: [5, 10, 30]) # full Chu270; training/validation remain M5/10
    sizes: List[int] = field(default_factory=lambda: [100, 250, 500])
    max_instances: int = 0                # 0 = all matching instances
    checkpoint: str = os.path.join("checkpoints", "best.pt")
    results_dir: str = "results"

    num_samples: int = 256                # exact K for sampleK (greedy is a separate stage)
    sample_temperature: float = 1.0
    max_rollouts_per_chunk: int = 2048
    sms_dims: str = "ua"
    sms_scale: float = 1.0
    sms_include_center: bool = True

    # External refinement stages; equal ls_pool startup budgets apply to every arm.
    ls_enable: bool = False
    ls_pool: int = 16
    ls_max_iters: int = 200
    ls_drop_k: int = 32                   # candidate items for the drop-and-refill move
    ils_rounds: int = 100                 # iterated perturbation rounds (0 = plain LS)
    ils_perturb_max: int = 4              # items dropped per perturbation (2..max)
    ls_batch: int = 16                    # solutions refined in parallel


@dataclass
class Config:
    device: DeviceConfig = field(default_factory=DeviceConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _parse_value(raw: str, current):
    if isinstance(current, bool):
        return raw.strip().lower() in ("1", "true", "yes", "y", "on")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        items = [s for s in raw.split(",") if s.strip() != ""]
        if current and isinstance(current[0], int):
            return [int(s) for s in items]
        if current and isinstance(current[0], float):
            return [float(s) for s in items]
        # empty list or string list: try numeric first
        try:
            return [int(s) for s in items]
        except ValueError:
            try:
                return [float(s) for s in items]
            except ValueError:
                return [s.strip() for s in items]
    return raw


def config_from_dict(d: dict) -> Config:
    """Rebuild a Config from a (possibly partial / older) dictionary."""
    cfg = Config()
    for section, values in (d or {}).items():
        if not hasattr(cfg, section) or not isinstance(values, dict):
            continue
        sub = getattr(cfg, section)
        valid = {f.name for f in dataclasses.fields(sub)}
        for k, v in values.items():
            if k in valid:
                setattr(sub, k, v)
    if d and "model" in d and "score_normalization" not in d["model"]:
        cfg.model.score_normalization = False
    if d and "signal" not in d.get("train", {}):
        cfg.train.signal = "item_pg"
    return cfg


def apply_overrides(cfg: Config, overrides: List[str]) -> Config:
    """Apply `--section.field value` pairs to cfg (returns a modified copy)."""
    cfg = copy.deepcopy(cfg)
    i = 0
    while i < len(overrides):
        key = overrides[i]
        if not key.startswith("--"):
            raise ValueError(f"Unexpected argument '{key}', expected --section.field value")
        key = key[2:]
        if "=" in key:
            key, raw = key.split("=", 1)
            i += 1
        else:
            if i + 1 >= len(overrides):
                raise ValueError(f"Missing value for --{key}")
            raw = overrides[i + 1]
            i += 2
        if "." not in key:
            raise ValueError(f"Override '--{key}' must look like --section.field")
        section, name = key.split(".", 1)
        sub = getattr(cfg, section, None)
        if sub is None or not hasattr(sub, name):
            raise ValueError(f"Unknown config key '{section}.{name}'")
        setattr(sub, name, _parse_value(raw, getattr(sub, name)))
    return cfg


def get_config(argv: Optional[List[str]] = None) -> Config:
    return apply_overrides(Config(), list(argv or []))


def resolve_path(path: str) -> str:
    """Relative paths are interpreted relative to the project root."""
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def get_device(cfg: Config):
    import torch
    if cfg.device.force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        idx = cfg.device.cuda_device
        if idx >= n:
            print(f"[config] cuda:{idx} not available ({n} visible GPU(s)); falling back to cuda:0")
            idx = 0
        torch.cuda.set_device(idx)
        return torch.device(f"cuda:{idx}")
    print("[config] CUDA not available, running on CPU")
    return torch.device("cpu")


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=description + "  Any config field can be overridden with --section.field value.",
    )
    p.add_argument("--print_config", action="store_true", help="print the resolved config and exit")
    return p


if __name__ == "__main__":
    print(json.dumps(get_config().to_dict(), indent=2))
