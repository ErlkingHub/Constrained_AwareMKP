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
    # The local machine has two RTX 4090s; GPU 0 is busy -> use GPU 1.
    cuda_device: int = 1
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

    # How infeasible picks are treated (see README "Constraint awareness without masks"):
    #   "perceive"  : NO feasibility mask. An infeasible pick is *rejected* by the env (the
    #                 item is discarded, capacities untouched) and penalised. Because
    #                 capacities only shrink, a rejected item could never fit again, so the
    #                 rejection is loss-free for solution quality; the agent must perceive
    #                 feasibility from the state to avoid wasting steps / penalties.
    #   "soft_mask" : additionally add -soft_mask_beta * violation to the logits (ablation).
    #   "hard_mask" : classic -inf mask on infeasible items (ablation / upper reference).
    feasibility_mode: str = "perceive"
    soft_mask_beta: float = 5.0
    reject_penalty: float = 0.1      # reward -= reject_penalty * (#rejected picks) / N

    # Batched Lagrangian-dual (LP relaxation) approximation used for features + bound.
    dual_iters: int = 300
    dual_lr: float = 0.05


# --------------------------------------------------------------------------------------
# Model: State encoder -> High-level strategy controller -> Low-level weak decoder
# --------------------------------------------------------------------------------------
@dataclass
class ModelConfig:
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

    # Strategy primitives of the weak decoder (subset/order is configurable):
    #   profit | efficiency | price_ratio | slack
    primitives: List[str] = field(default_factory=lambda: ["profit", "efficiency", "price_ratio", "slack"])
    init_primitive: str = "price_ratio"   # primitive favoured at initialisation (warm start)
    init_primitive_logit: float = 2.0
    price_log_range: float = 3.0          # learned prices lambda_t = base * exp(+-range)
    price_floor: float = 0.05             # floor (relative to mean dual price) for non-binding dims
    kappa_init: float = 3.0               # initial inverse temperature of the weak decoder
    kappa_min: float = 0.1
    residual_max: float = 3.0             # |delta_j| bound of the per-item learned residual
    grad_checkpoint: bool = True          # checkpoint the per-step controller (saves memory)


# --------------------------------------------------------------------------------------
# Training (REINFORCE with shared multi-sample baseline)
# --------------------------------------------------------------------------------------
@dataclass
class TrainConfig:
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

    # validation on a fixed set of generated instances (greedy decoding, gap vs LP bound)
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


# --------------------------------------------------------------------------------------
# Zero-shot benchmark
# --------------------------------------------------------------------------------------
@dataclass
class EvalConfig:
    mat_file: str = os.path.join("MKP_dataset", "Benchmarks_Chu.mat")
    dims: List[int] = field(default_factory=lambda: [5, 10])     # add 30 to also test 30-dim
    sizes: List[int] = field(default_factory=lambda: [100, 250, 500])
    max_instances: int = 0                # 0 = all matching instances
    checkpoint: str = os.path.join("checkpoints", "best.pt")
    results_dir: str = "results"

    num_samples: int = 256                # sampled rollouts per instance (+1 greedy)
    sample_temperature: float = 1.0
    max_rollouts_per_chunk: int = 2048

    # Post-hoc interpretable refinement (classic MKP moves, applied to the best
    # `ls_pool` distinct policy solutions). Disable for a pure-policy measurement;
    # the CSV always reports policy-only gaps too.
    ls_enable: bool = True
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
