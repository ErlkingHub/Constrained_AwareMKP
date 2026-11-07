"""Activated v6 theta arithmetic and injection; imports no project modules.

The caller must bootstrap a separate legacy-only Python process before invoking
get_default. All values returned here are detached CPU float32 tensors with a
single leading batch row. Nothing writes to the legacy source/checkpoint tree.
"""
from __future__ import annotations

import math
import sys

sys.dont_write_bytecode = True

import torch
from torch import nn


THETA_FIELDS = (
    "resource_prices", "profit_exponent", "scarcity_exponent", "bottleneck_weight",
    "temperature", "opportunity_weight", "reserve_weight", "lp_primal_weight",
    "lp_dual_weight", "resource_interactions", "companion_weights",
)
ABLATION_FIELDS = {
    "prices": "resource_prices", "profit_exponent": "profit_exponent",
    "scarcity": "scarcity_exponent", "bottleneck": "bottleneck_weight",
    "opportunity": "opportunity_weight", "reserve": "reserve_weight",
    "lp_primal": "lp_primal_weight", "lp_dual": "lp_dual_weight",
    "interactions": "resource_interactions", "companion": "companion_weights",
}
VALIDATION_ATOL = 1e-5


def _config(cfg, name):
    return cfg[name] if isinstance(cfg, dict) else getattr(cfg, name)


def theta_to_dict(theta):
    """Copy one v6 StrategyTheta (or its serialized dictionary) without activation."""
    if isinstance(theta, dict):
        missing, extra = set(THETA_FIELDS) - set(theta), set(theta) - set(THETA_FIELDS)
        if missing or extra:
            raise ValueError(f"Invalid theta fields: missing={sorted(missing)}, extra={sorted(extra)}")
    result = {}
    for name in THETA_FIELDS:
        value = theta[name] if isinstance(theta, dict) else getattr(theta, name)
        value = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float32).clone()
        if value.ndim < 1 or value.shape[0] != 1:
            raise ValueError(f"{name} must retain exactly one leading batch row")
        result[name] = value
    return result


def validate_theta(theta, cfg, allow_neutral=False):
    """Assert the v6 closed activation ranges, convex gate budgets and shapes.

    Neutral zeros for penalty/basis groups are inside their closed ranges. The
    profit neutral is one (never zero); allow_neutral also permits it when a
    caller's configured alpha interval happens not to contain one.
    """
    values = theta_to_dict(theta)
    prices = values["resource_prices"]
    if prices.ndim != 2 or prices.shape[1] < 1:
        raise ValueError("resource_prices must have shape (1,M), M>=1")
    m, k = prices.shape[1], len(_config(cfg, "companion_scales"))
    shapes = {"resource_prices": (1, m), "resource_interactions": (1, m, m),
              "companion_weights": (1, k)}
    for name, value in values.items():
        expected = shapes.get(name, (1,))
        if tuple(value.shape) != expected or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite with shape {expected}")
    tolerance = VALIDATION_ATOL
    floor = float(_config(cfg, "price_floor"))
    upper = floor + (1 - floor) * m
    if bool(((prices < floor - tolerance) | (prices > upper + tolerance)).any()):
        raise ValueError("resource_prices violate the v6 softmax floor/upper bounds")
    if not torch.allclose(prices.mean(-1), torch.ones(1), rtol=0, atol=tolerance):
        raise ValueError("resource_prices must have mean one")
    bounds = {
        "profit_exponent": (_config(cfg, "profit_exponent_min"), _config(cfg, "profit_exponent_max")),
        "scarcity_exponent": (0., _config(cfg, "scarcity_exponent_max")),
        "bottleneck_weight": (0., _config(cfg, "bottleneck_weight_max")),
        "temperature": (_config(cfg, "temperature_min"), _config(cfg, "temperature_max")),
        "opportunity_weight": (0., _config(cfg, "opportunity_weight_max")),
        "reserve_weight": (0., _config(cfg, "reserve_weight_max")),
        "lp_primal_weight": (0., _config(cfg, "lp_primal_weight_max")),
        "lp_dual_weight": (0., _config(cfg, "lp_dual_weight_max")),
    }
    for name, (low, high) in bounds.items():
        value = values[name]
        neutral_alpha = (allow_neutral and name == "profit_exponent" and
                         bool(torch.allclose(value, torch.ones_like(value), rtol=0, atol=tolerance)))
        if not neutral_alpha and bool(((value < float(low) - tolerance) | (value > float(high) + tolerance)).any()):
            raise ValueError(f"{name} is outside [{low}, {high}]")
    if not bool((values["temperature"] > 0).all()) or not bool((prices > 0).all()):
        raise ValueError("temperature and resource prices must remain strictly positive")
    interaction = values["resource_interactions"]
    companion = values["companion_weights"]
    if not torch.allclose(interaction, interaction.transpose(-1, -2), rtol=0, atol=tolerance):
        raise ValueError("resource_interactions must be symmetric")
    if bool((interaction < -tolerance).any()) or bool((companion < -tolerance).any()):
        raise ValueError("Combination coefficients must be nonnegative")
    # The symmetric lower triangle duplicates unordered resource coefficients.
    # The decoder and the interaction gate count the upper triangle exactly once.
    if float(interaction.triu().sum()) > float(_config(cfg, "interaction_weight_max")) + tolerance:
        raise ValueError("resource_interactions exceed the upper-triangle gate budget")
    if float(companion.sum()) > float(_config(cfg, "companion_weight_max")) + tolerance:
        raise ValueError("companion_weights exceed the companion gate budget")
    return True


def mean_theta(thetas, cfg):
    """Average activated decision records; each input record has equal weight.

    G/C callers supply every active decision, rather than first averaging each
    trajectory (which would incorrectly give long and short episodes equal mass).
    """
    records = [theta_to_dict(theta) for theta in thetas]
    if not records:
        raise ValueError("Cannot average an empty theta collection")
    for record in records:
        validate_theta(record, cfg)
    shapes = {name: tuple(records[0][name].shape) for name in THETA_FIELDS}
    if any(tuple(record[name].shape) != shapes[name] for record in records for name in THETA_FIELDS):
        raise ValueError("A theta mean cannot mix resource/basis dimensions")
    result = {name: torch.stack([record[name] for record in records]).double().mean(0).float()
              for name in THETA_FIELDS}
    price_mean = result["resource_prices"].mean(-1, keepdim=True)
    if not torch.allclose(price_mean, torch.ones_like(price_mean), rtol=0, atol=VALIDATION_ATOL):
        raise ValueError("Averaging activated v6 prices unexpectedly lost their mean-one invariant")
    result["resource_prices"] = result["resource_prices"] / price_mean
    interaction = result["resource_interactions"]
    result["resource_interactions"] = (interaction + interaction.transpose(-1, -2)) / 2
    validate_theta(result, cfg)
    return result


def ablated(theta, term, cfg):
    """One neutralization; never change decoder/LP configuration or temperature."""
    aliases = {**ABLATION_FIELDS, **{value: value for value in ABLATION_FIELDS.values()}}
    if term not in aliases:
        raise ValueError(f"Unknown v6 ablation term: {term}")
    result = theta_to_dict(theta)
    validate_theta(result, cfg, allow_neutral=True)
    name = aliases[term]
    result[name] = (torch.ones_like(result[name]) if name in ("resource_prices", "profit_exponent")
                    else torch.zeros_like(result[name]))
    validate_theta(result, cfg, allow_neutral=True)
    return result


def injected_actor(theta, theta_class):
    """Return a buffer-only actor; expand one activated theta to the state's batch."""
    stored = theta_to_dict(theta)

    class InjectedActor(nn.Module):
        def __init__(self):
            super().__init__()
            for name, value in stored.items():
                self.register_buffer(name, value)

        def forward(self, state):
            batch, n = state.profits.shape
            values = {name: getattr(self, name).to(device=state.profits.device, dtype=torch.float32)
                      .expand(batch, *getattr(self, name).shape[1:]) for name in THETA_FIELDS}
            feasibility = torch.zeros((batch, n), dtype=torch.float32, device=state.profits.device)
            return theta_class(**values), feasibility

    return InjectedActor()


@torch.no_grad()
def get_default(model_cfg, state):
    """Read D from two newly initialized old actors and require exact equality."""
    # This import belongs to the caller's isolated legacy process. This helper
    # must never be used after loading the new project's same-named modules.
    from mkp_Model import HighLevelController

    device = state.profits.device
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    results = []
    with torch.random.fork_rng(devices=cuda_devices):
        for seed in (314159, 271828):
            torch.manual_seed(seed)
            actor = HighLevelController(model_cfg).to(device).eval().requires_grad_(False)
            theta, _ = actor(state)
            results.append(theta_to_dict(theta))
            del actor
    for result in results:
        validate_theta(result, model_cfg)
    for name in THETA_FIELDS:
        if not torch.equal(results[0][name], results[1][name]):
            raise AssertionError(f"Fresh-actor default theta depends on initialization seed: {name}")
    return results[0]
