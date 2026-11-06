"""Static strategy-space candidates with prefix-stable, independent latent draws.

One CPU generator per theta index draws exactly M+1 normals in a fixed order.
Neither the requested K nor rollout chunking participates in its seed. All theta
values are generated before rollouts. There is no antithetic coupling.
"""
from __future__ import annotations

import math

import torch

from mkp_evaluation import paired_seed
from mkp_Model import theta_from_pre


def _f0_parameters(model, inst, ec):
    mcfg = model.mcfg
    alpha_p = (mcfg.alpha_init - .5) / 2.
    if not 0 < alpha_p < 1:
        raise ValueError("SMS requires alpha_init strictly between .5 and 2.5")
    dims, scale = getattr(ec, "sms_dims", "ua"), float(getattr(ec, "sms_scale", 1.))
    if getattr(mcfg, "theta_sigma_head", False):
        # A matched T4-b fixed control is the trained run's registered isotropic
        # proposal, not a fresh default T4-a scale/dims choice.
        dims = mcfg.theta_sigma_dims
        sigma_u, sigma_a = mcfg.theta_sigma_u0, mcfg.theta_sigma_alpha0
    else:
        sigma_u, sigma_a = .05 * scale, .3 * scale
    return {"u_pre": torch.zeros(1, inst.M), "alpha_pre": torch.tensor([math.log(alpha_p / (1 - alpha_p))]),
            "sigma_u": torch.full((1, inst.M), sigma_u if "u" in dims else 0.),
            "sigma_alpha": torch.tensor([sigma_a if "a" in dims else 0.])}


@torch.no_grad()
def proposal_candidates(model, env, inst, pre, ec, theta_k, arm, seed, instance_id, tag=0, amp=False):
    """Return K static theta overrides and proposal details for one instance."""
    if inst.B != 1 or theta_k < 1:
        raise ValueError("SMS evaluation requires one instance and positive theta_k")
    if (model.mcfg.theta_groups != ["price", "alpha"] or model.mcfg.price_anchor != "root"
            or env.cfg.feasibility_mode != "hard_mask"):
        raise ValueError("SMS uses the F0 price/alpha executor with a root price anchor and hard mask")
    dims, scale = getattr(ec, "sms_dims", "ua"), float(getattr(ec, "sms_scale", 1.))
    if dims not in ("u", "a", "ua") or not math.isfinite(scale) or scale < 0:
        raise ValueError("SMS dims must be u/a/ua and scale finite/nonnegative")
    if arm == "fixed":
        params = _f0_parameters(model, inst, ec)
    elif arm in ("learned_static", "learned_mu_only"):
        if model.mcfg.theta_update != "static" or not getattr(model.mcfg, "theta_sigma_head", False):
            raise ValueError("Learned SMS requires a static proposal checkpoint with theta_sigma_head=True")
        params = model.proposal_parameters(env, inst, pre=pre, amp=amp)
        params = {k: v.detach().cpu().float() for k, v in params.items()}
        if arm == "learned_mu_only":
            base_dims = model.mcfg.theta_sigma_dims
            params["sigma_u"] = torch.full_like(params["u_pre"], model.mcfg.theta_sigma_u0 if "u" in base_dims else 0.)
            params["sigma_alpha"] = torch.full_like(params["alpha_pre"], model.mcfg.theta_sigma_alpha0 if "a" in base_dims else 0.)
    else:
        raise ValueError("SMS supports fixed, learned_static and learned_mu_only proposal arms")
    u_mean = params["u_pre"].reshape(inst.M).cpu().float()
    a_mean = params["alpha_pre"].reshape(()).cpu().float()
    su = params["sigma_u"].reshape(inst.M).cpu().float()
    sa = params["sigma_alpha"].reshape(()).cpu().float()
    if not all(bool(torch.isfinite(t).all()) for t in (u_mean, a_mean, su, sa)) or bool((su < 0).any()) or bool(sa < 0):
        raise ValueError("Proposal means and scales must be finite, with nonnegative scales")
    us, alphas, latent_u, latent_a = [], [], [], []
    include_center = bool(getattr(ec, "sms_include_center", True))
    for index in range(theta_k):
        generator = torch.Generator(device="cpu").manual_seed(paired_seed(seed, instance_id, "sms", f"{tag}:{index}"))
        eps = torch.randn(inst.M + 1, generator=generator)
        if index == 0 and include_center:
            eps.zero_()
        v, z = u_mean + su * eps[:inst.M], a_mean + sa * eps[-1]
        # Activate one row at a time too: the latent and transformed prefixes do
        # not depend on vector-kernel choices for different requested K values.
        theta = theta_from_pre(v[None], z[None], model.mcfg.u_max)
        us.append(theta["u"])
        alphas.append(theta["alpha"])
        latent_u.append(v)
        latent_a.append(z)
    theta = {"u": torch.cat(us).to(inst.p.device), "alpha": torch.cat(alphas).to(inst.p.device)}
    details = {"u_pre": torch.stack(latent_u), "alpha_pre": torch.stack(latent_a),
               "mean_u_pre": u_mean, "mean_alpha_pre": a_mean,
               "sigma_u": su, "sigma_alpha": sa, "include_center": include_center,
               "rng": "per-theta-index SHA256 seed, M+1 normals, no antithetic"}
    return theta, details


@torch.no_grad()
def sms_rollouts(model, env, inst, pre, ec, spec, arm, seed, instance_id, *, amp=False, tag=0,
                 return_trace=False):
    """Yield (offset, model output) in theta-major order; first candidate is greedy.

    Pure theta stages use the configured rollout chunks. Hybrid item-sampling
    trajectories use their own seeds and one rollout each to make item draws as
    well as latent prefixes independent of chunk size and larger K_theta.
    """
    theta, details = proposal_candidates(model, env, inst, pre, ec, spec["theta_k"], arm,
                                         seed, instance_id, tag, amp)
    item_k, budget = spec["item_k"], spec["budget"]
    expanded = {k: t.repeat_interleave(item_k, 0) for k, t in theta.items()}
    include_center = details["include_center"]
    offset = 0
    while offset < budget:
        greedy_center = offset == 0 and include_center
        # The center is separated even in theta stages. Thus theta1 uses exactly
        # the same one-row arithmetic as the center of every larger pool.
        count = 1 if spec["decode"] == "sample" or greedy_center else min(ec.max_rollouts_per_chunk, budget - offset)
        decode = "greedy" if greedy_center else spec["decode"]
        if decode == "sample":
            item_seed = paired_seed(seed, instance_id, "sms_items", f"{tag}:{offset // item_k}:{offset % item_k}")
            torch.manual_seed(item_seed)
            if inst.p.is_cuda:
                torch.cuda.manual_seed_all(item_seed)
        override = {k: value[offset:offset + count] for k, value in expanded.items()}
        out = model(env, inst, num_samples=count, decode=decode, temperature=ec.sample_temperature,
                    amp=amp, pre=pre, theta_override=override, return_trace=return_trace)
        out["sms_proposal"] = details if offset == 0 else None
        out["sms_theta_override"] = override
        yield offset, out
        offset += count
