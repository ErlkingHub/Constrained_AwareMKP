"""All experiment settings. Paths are resolved relative to this project, not cwd.

Use a JSON config containing any subset of these nested sections; unknown keys
are errors. Checkpoints store the fully expanded configuration.
"""
from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path


@dataclass
class ModelConfig:
    hidden_dim: int = 64
    message_rounds: int = 2
    price_log_bound: float = 2.0
    temperature: float = 0.25
    lp_input: bool = True
    lp_anchor: bool = True
    lp_anchor_mix: float = 0.05  # Uniform floor for resources with zero root dual.
    lp_score: bool = True
    pair_features: bool = True
    pair_score: bool = True
    pair_chunk_size: int = 64
    readout_init_std: float = 0.01
    initial_criterion_bias: tuple = (2.0, 0.0, -1.0, -1.0)
    basis_clip: float | None = None
    resource_log_clip: float = 20.0
    eps: float = 1e-8


@dataclass
class DataConfig:
    train_n: tuple = (100,)
    train_m: tuple = (5, 10)
    train_distributions: tuple = ("weakly_correlated",)
    capacity_ratios: tuple = (0.25, 0.5, 0.75)
    coefficient_max: int = 1000
    correlation_noise: int = 100
    correlation_offset: int = 100
    validation_seed: int = 104729
    validation_instances: int = 24


@dataclass
class TrainConfig:
    version: str = "main_g1"
    seed: int = 42
    device: str = "auto"
    torch_threads: int = 1
    updates: int = 2000  # Cumulative target, including updates before --resume.
    episodes_per_update: int = 8
    ppo_epochs: int = 3
    minibatch_size: int = 64  # Decision states, not independent instances.
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    clip_ratio: float = 0.2
    target_kl: float = 0.03
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    gamma: float = 1.0
    gae_lambda: float = 1.0  # Full remaining episode return at lambda=gamma=1.
    reward_mode: str = "terminal"  # terminal or incremental, normalized by root LP UB.
    shaping_coef: float = 0.0  # Independent LP reward-path ablation.
    validate_every: int = 25
    checkpoint_every: int = 25
    log_root: str = "runs"
    checkpoint_root: str = "checkpoints"
    research_question: str = "Dynamic resource valuation and long-horizon credit assignment"


@dataclass
class EvalConfig:
    checkpoint: str | None = None
    dataset_path: str = "MKP_dataset/Benchmarks_Chu.mat"
    output_root: str = "results"
    arms: tuple = ("dynamic", "static", "fixed", "untrained", "lp_dual")
    best_of_k: int = 0
    seed: int = 20261031
    device: str = "auto"
    torch_threads: int = 1
    limit: int | None = None
    filter_n: tuple = ()
    filter_m: tuple = ()
    synthetic: bool = False
    synthetic_n: tuple = (50, 100, 250, 500)
    synthetic_m: tuple = (5, 10, 30)
    synthetic_distributions: tuple = ("uncorrelated", "weakly_correlated", "strongly_correlated", "subset_sum")
    synthetic_ratios: tuple = (0.25, 0.5, 0.75)
    synthetic_instances_per_group: int = 5
    synthetic_coefficient_max: int = 1000
    synthetic_correlation_noise: int = 100
    synthetic_correlation_offset: int = 100


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        unknown = set(value) - {"model", "data", "train", "eval"}
        if unknown:
            raise ValueError(f"Unknown config sections: {sorted(unknown)}")
        result = cls()
        for section, config_type in (("model", ModelConfig), ("data", DataConfig), ("train", TrainConfig), ("eval", EvalConfig)):
            supplied = value.get(section, {})
            defaults = config_type()
            valid = {f.name for f in fields(defaults)}
            if set(supplied) - valid:
                raise ValueError(f"Unknown {section} settings: {sorted(set(supplied) - valid)}")
            converted = {k: tuple(v) if isinstance(getattr(defaults, k), tuple) else v for k, v in supplied.items()}
            setattr(result, section, config_type(**converted))
        result.validate()
        return result

    def validate(self):
        m, d, t, e = self.model, self.data, self.train, self.eval
        for section in (m, d, t, e):
            defaults = type(section)()
            for f in fields(section):
                value, default = getattr(section, f.name), getattr(defaults, f.name)
                if isinstance(default, bool) and not isinstance(value, bool):
                    raise ValueError(f"{f.name} must be boolean")
                if isinstance(default, int) and not isinstance(default, bool):
                    if not isinstance(value, int) or isinstance(value, bool):
                        raise ValueError(f"{f.name} must be an integer")
                if isinstance(default, float):
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError(f"{f.name} must be a finite number")
        for name, value in (("hidden_dim", m.hidden_dim), ("message_rounds", m.message_rounds), ("pair_chunk_size", m.pair_chunk_size),
                            ("updates", t.updates), ("episodes_per_update", t.episodes_per_update), ("ppo_epochs", t.ppo_epochs),
                            ("minibatch_size", t.minibatch_size), ("validate_every", t.validate_every), ("checkpoint_every", t.checkpoint_every),
                            ("validation_instances", d.validation_instances), ("torch_threads", t.torch_threads), ("eval.torch_threads", e.torch_threads),
                            ("coefficient_max", d.coefficient_max), ("synthetic_coefficient_max", e.synthetic_coefficient_max),
                            ("synthetic_instances_per_group", e.synthetic_instances_per_group)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < m.temperature or not 0 < m.eps or m.price_log_bound < 0:
            raise ValueError("temperature/eps must be positive; price_log_bound nonnegative")
        if not 0 <= m.lp_anchor_mix <= 1 or m.readout_init_std < 0 or m.resource_log_clip <= 0:
            raise ValueError("Invalid model initialization/anchor settings")
        if len(m.initial_criterion_bias) != 4 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in m.initial_criterion_bias):
            raise ValueError("initial_criterion_bias requires four finite numbers")
        if m.basis_clip is not None and (isinstance(m.basis_clip, bool) or not isinstance(m.basis_clip, (float, int)) or not math.isfinite(m.basis_clip) or m.basis_clip <= 0):
            raise ValueError("basis_clip must be null or a positive finite number")
        if t.gamma != 1.0:
            raise ValueError("MKP total-profit objective requires gamma=1 (no preference for shorter episodes)")
        if not 0 <= t.gae_lambda <= 1 or t.reward_mode not in ("terminal", "incremental"):
            raise ValueError("Invalid credit assignment settings")
        if not 0 < t.clip_ratio < 1 or min(t.actor_lr, t.critic_lr, t.max_grad_norm, t.target_kl) <= 0:
            raise ValueError("Invalid PPO optimizer settings")
        if min(t.entropy_coef, t.value_coef, t.shaping_coef, d.correlation_noise, d.correlation_offset,
               e.synthetic_correlation_noise, e.synthetic_correlation_offset) < 0:
            raise ValueError("Loss/shaping/correlation settings must be nonnegative")
        if not t.version or Path(t.version).name != t.version or t.version in (".", ".."):
            raise ValueError("version must be one directory name")
        if t.device not in ("auto", "cpu", "cuda") or e.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")
        if t.seed == d.validation_seed or min(t.seed, d.validation_seed, e.seed) < 0 or max(t.seed, d.validation_seed, e.seed) >= 2**32:
            raise ValueError("Seeds must be integers in [0, 2**32); training and validation seeds must differ")
        families = {"uncorrelated", "weakly_correlated", "strongly_correlated", "subset_sum"}
        for name, sizes in (("train_n", d.train_n), ("train_m", d.train_m), ("synthetic_n", e.synthetic_n), ("synthetic_m", e.synthetic_m)):
            if not sizes or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in sizes):
                raise ValueError(f"{name} requires positive integer sizes")
        for values in (d.capacity_ratios, e.synthetic_ratios):
            if not values or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not 0 < x <= 1 for x in values):
                raise ValueError("Capacity ratios must be in (0, 1]")
        for values in (d.train_distributions, e.synthetic_distributions):
            if not values or not set(values) <= families:
                raise ValueError(f"Supported synthetic distributions: {sorted(families)}")
        if not e.arms or len(set(e.arms)) != len(e.arms) or not set(e.arms) <= {"dynamic", "static", "fixed", "untrained", "lp_dual"}:
            raise ValueError("Invalid or duplicate evaluation arms")
        if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in (*e.filter_n, *e.filter_m)):
            raise ValueError("Evaluation filters require positive integer sizes")
        if e.best_of_k < 0 or (e.limit is not None and (not isinstance(e.limit, int) or isinstance(e.limit, bool) or e.limit <= 0)):
            raise ValueError("best_of_k must be nonnegative and limit positive")
        return self


def load_config(path=None):
    if path is None:
        return ExperimentConfig().validate()
    path = Path(path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return ExperimentConfig.from_dict(json.loads(path.read_text(encoding="utf-8-sig")))
