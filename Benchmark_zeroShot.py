"""Matched weak-decoder benchmark. Greedy is the primary search-free stage.

--arm fixed,untrained,learned_static,learned_dynamic --checkpoint <dynamic.pt>
--static_checkpoint <separately-trained-static.pt> --stages greedy,sample64,sample256,greedy+ILS,sampleK+ILS

sampleK is exactly K sampled trajectories, excluding the separate greedy rollout.
"""
from __future__ import annotations

import math
import copy
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

from mkp_artifacts import sha256_file, source_provenance, write_json
from mkp_config import build_arg_parser, get_config, get_device, resolve_path
from mkp_env import MKPEnv, instances_from_dicts
from mkp_evaluation import ARMS, paired_report, paired_seed, parse_stages, summarize, write_csv, jsonable
from mkp_Model import build_model
from mkp_refine import greedy_fill, is_feasible, iterated_local_search, _ratio
from mkp_Trainer import load_policy, set_seed
from read_mat import read_mat


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _audit(solution, inst):
    rep = inst.repeat(solution.shape[0])
    if not bool(is_feasible(solution, rep.r, rep.c).all()):
        raise RuntimeError("Infeasible benchmark solution")
    slack = rep.c - torch.einsum("bmn,bn->bm", rep.r, solution)
    if bool(((solution < .5) & (rep.r <= slack[:, :, None] + 1e-6).all(1)).any()):
        raise RuntimeError("Non-maximal benchmark solution")


def _decision_statistics(traces):
    """Summarize active trajectories at controller decision points, not held steps."""
    u_sum, u_count, alpha_sum, alpha_sq, decisions, gamma_sum, gamma_count = 0., 0, 0., 0., 0, 0., 0
    eta_sum, eta_count = 0., 0
    for trace in traces:
        for step in trace:
            if not step.get("theta_decision", True) or "u" not in step or "alpha" not in step:
                continue
            u, alpha = torch.as_tensor(step["u"]), torch.as_tensor(step["alpha"])
            active = torch.as_tensor(step.get("active", torch.ones(alpha.shape, dtype=torch.bool))).bool()
            u, alpha = u[active], alpha[active]
            u_sum += float(u.abs().sum())
            u_count += u.numel()
            alpha_sum += float(alpha.sum())
            alpha_sq += float(alpha.square().sum())
            decisions += alpha.numel()
            if "gamma" in step:
                gamma = torch.as_tensor(step["gamma"])[active]
                gamma_sum += float(gamma.sum())
                gamma_count += gamma.numel()
            if "eta" in step:
                eta = torch.as_tensor(step["eta"])[active]
                eta_sum += float(eta.sum())
                eta_count += eta.numel()
    alpha_mean = alpha_sum / max(decisions, 1)
    return {"theta_decisions": decisions, "decision_u_abs_mean": u_sum / max(u_count, 1),
            "decision_alpha_mean": alpha_mean,
            "decision_alpha_std": math.sqrt(max(0., alpha_sq / max(decisions, 1) - alpha_mean ** 2)),
            "decision_gamma_mean": gamma_sum / gamma_count if gamma_count else "",
            "decision_eta_mean": eta_sum / eta_count if eta_count else ""}


@torch.no_grad()
def _historical_rollout(env, inst, pre, k, decode, temperature):
    """Historical static-c ratio, plus an explicitly named stochastic extension."""
    mu, ub = pre
    rp, prices = inst.repeat(k), mu.repeat_interleave(k, 0)
    score = _ratio(rp.p, rp.r, rp.c, prices).clamp_min(1e-6).log()
    if decode == "greedy":
        sol = greedy_fill(torch.zeros_like(rp.p), rp.p, rp.r, rp.c, score)
    else:
        st = env.reset(rp, prices, ub.repeat_interleave(k, 0))
        while not bool(st.done.all()):
            logits = (score / temperature).masked_fill(~st.fits, -1e9)
            logits = torch.where(st.done[:, None], torch.zeros_like(logits), logits)
            st = env.step(st, torch.multinomial(logits.softmax(-1), 1).squeeze(-1))
        sol = st.solution
    return {"solution": sol, "profit": (sol * rp.p).sum(1), "trace": []}


@torch.no_grad()
def evaluate_stage(model, env, inst, ec, device, amp, pre, stage, arm, seed, instance_id,
                   root_lp_seconds=0.0, artifact_sink=None):
    spec = parse_stages([stage], ec.num_samples)[0]
    rng_seed = paired_seed(seed, instance_id, stage)
    set_seed(rng_seed)
    gen = torch.Generator(device="cpu").manual_seed(rng_seed)
    mu, ub = pre
    sols, objs, traces = [], [], []
    residual_lp_seconds = 0.0
    _sync(device)
    start, remaining, offset, artifact_seconds = time.perf_counter(), spec["budget"], 0, 0.0
    while remaining:
        k = min(remaining, ec.max_rollouts_per_chunk)
        if arm == "fixed_dualgreedy":
            out = _historical_rollout(env, inst, pre, k, spec["decode"], ec.sample_temperature)
        else:
            kwargs = {"theta_override": "fixed"} if arm == "fixed" else {}
            out = model(env, inst, num_samples=k, decode=spec["decode"],
                        temperature=ec.sample_temperature, amp=amp, pre=pre,
                        return_trace=True, **kwargs)
        residual_lp_seconds += float(out.get("lp_time_s", 0.0))
        sols.append(out["solution"])
        objs.append(out["profit"])
        traces.append(out.get("trace", []))
        if artifact_sink is not None:
            _sync(device)
            astart = time.perf_counter()
            artifact_sink({"arm": arm, "stage": stage, "sample_offset": offset,
                           "solutions": out["solution"], "objectives": out["profit"],
                           "theta_trace": out.get("trace", [])})
            artifact_seconds += time.perf_counter() - astart
        remaining -= k
        offset += k
    _sync(device)
    construction_seconds = time.perf_counter() - start - artifact_seconds - residual_lp_seconds
    solution, objective = torch.cat(sols), torch.cat(objs)
    theta_stats = _decision_statistics(traces)
    _audit(solution, inst)
    idx = int(objective.argmax())
    best_solution, best_obj = solution[idx:idx + 1], float(objective[idx])
    refine_seconds, pool_size = 0.0, 0
    if spec["refine"]:
        # Retain duplicates to avoid unequal numbers of ILS starts between arms.
        pool_size = min(ec.ls_pool, spec["budget"])
        pool = solution[objective.argsort(descending=True)[:pool_size]]
        gen.manual_seed(paired_seed(seed, instance_id, stage, "ILS"))
        _sync(device)
        start, artifact_seconds = time.perf_counter(), 0.0
        for offset in range(0, pool_size, ec.ls_batch):
            x = pool[offset:offset + ec.ls_batch]
            rp = inst.repeat(x.shape[0])
            bx, bo = iterated_local_search(x, rp.p, rp.r, rp.c, mu.repeat_interleave(x.shape[0], 0),
                rounds=ec.ils_rounds, perturb_max=ec.ils_perturb_max,
                max_iters=ec.ls_max_iters, drop_k=ec.ls_drop_k, generator=gen)
            idx = int(bo.argmax())
            if float(bo[idx]) > best_obj:
                best_obj, best_solution = float(bo[idx]), bx[idx:idx + 1]
            if artifact_sink is not None:
                _sync(device)
                astart = time.perf_counter()
                artifact_sink({"arm": arm, "stage": stage, "refine_start_offset": offset,
                               "refined_solutions": bx, "refined_objectives": bo})
                artifact_seconds += time.perf_counter() - astart
        _sync(device)
        refine_seconds = time.perf_counter() - start - artifact_seconds
        _audit(best_solution, inst)
    return {"stage": stage, "budget": spec["budget"], "objective": best_obj,
            "ub_lp": float(ub[0]), "time_lp_s": root_lp_seconds + residual_lp_seconds,
            "time_root_lp_s": root_lp_seconds, "time_residual_lp_s": residual_lp_seconds,
            "time_construct_s": construction_seconds, "time_refine_s": refine_seconds,
            "time_total_s": root_lp_seconds + residual_lp_seconds + construction_seconds + refine_seconds,
            "search_starts": pool_size, "ils_rounds": ec.ils_rounds if spec["refine"] else 0,
            "ls_max_iters": ec.ls_max_iters if spec["refine"] else 0,
            "ls_drop_k": ec.ls_drop_k if spec["refine"] else 0,
            "ils_perturb_max": ec.ils_perturb_max if spec["refine"] else 0,
            "sample_temperature": ec.sample_temperature, "rng_seed": rng_seed,
            "solution": best_solution[0].to(torch.int8).cpu().tolist(), "feasible": True, "maximal": True,
            **theta_stats}


def _load_arms(cfg, arms, checkpoint, static_checkpoint, device):
    models, configs, checkpoints = {}, {}, {}
    reference_cfg, training_seed = cfg, cfg.device.seed
    load_names = [a for a in arms if a.startswith("learned_")]
    if not load_names and checkpoint:
        load_names = ["reference"]
    for name in load_names:
        path = static_checkpoint if name == "learned_static" and static_checkpoint else checkpoint
        path = resolve_path(path or cfg.eval.checkpoint)
        model, mc, ck = load_policy(path, device, cfg_override=cfg)
        if name.startswith("learned_"):
            expected = "static" if name == "learned_static" else "step"
            if mc.model.theta_update != expected:
                raise ValueError(f"{name} requires a separately trained theta_update={expected} checkpoint; "
                                 f"{path} was trained with {mc.model.theta_update}")
        ck_seed = int(ck.get("config", {}).get("device", {}).get("seed", cfg.device.seed))
        if checkpoints:
            if ck_seed != training_seed:
                raise ValueError("Matched arms must use the same training seed")
            lhs, rhs = reference_cfg.to_dict(), mc.to_dict()
            for conf in (lhs, rhs):
                for key in ("theta_update", "theta_interval", "untrained_random_heads"):
                    conf["model"].pop(key, None)
            if lhs["model"] != rhs["model"] or lhs["env"] != rhs["env"]:
                raise ValueError("Matched checkpoints must share encoder, theta groups and executor")
            for key in ("epochs", "steps_per_epoch", "batch_size", "num_samples"):
                if getattr(reference_cfg.train, key) != getattr(mc.train, key):
                    raise ValueError(f"Static/dynamic training budgets differ: {key}")
        reference_cfg, training_seed = mc, ck_seed
        models[name], configs[name] = model, mc
        checkpoints[name] = {"path": path, "sha256": sha256_file(path), "epoch": ck.get("epoch"), "global_step": ck.get("global_step"),
                             "selection_rule": ck.get("selection_rule", "not_recorded"),
                             "config": ck.get("config"), "training_seed": ck_seed}
    for name in arms:
        if name.startswith("learned_"):
            continue
        mc = copy.deepcopy(reference_cfg)
        # The untrained primary control always uses per-step theta, independent
        # of the order in which static/dynamic checkpoints were requested.
        mc.model.theta_update = "step"
        mc.model.theta_interval = 1
        mc.model.untrained_random_heads = name == "untrained"
        set_seed(training_seed)
        models[name] = None if name == "fixed_dualgreedy" else build_model(mc).to(device).eval()
        configs[name] = mc
    models.pop("reference", None)
    configs.pop("reference", None)
    return models, configs, checkpoints, training_seed


def _protocol(cfg, include_schedule):
    model = cfg.to_dict()["model"]
    omitted = () if include_schedule else ("theta_update", "theta_interval", "untrained_random_heads")
    for key in omitted:
        model.pop(key, None)
    return {"formula": "weak_decoder_t3_lp_v1" if ("lp" in cfg.model.theta_groups or cfg.model.price_anchor == "residual") else
                "weak_decoder_t2_block_v1" if "block" in cfg.model.theta_groups else
                ("weak_decoder_t1r_zscore_v1" if cfg.model.score_normalization else "weak_decoder_t1_g0g1_v1"),
                "model": model, "env": cfg.to_dict()["env"],
                "sampling": "exact-K-excluding-greedy", "search": "top-pool-retaining-duplicates",
                "global_temperature": "initial-for-controls-fitted-for-learned"}


def protocol_sha256(cfg):
    """Comparison signature (historical helper API): allow the declared schedule ablation."""
    return hashlib.sha256(json.dumps(_protocol(cfg, include_schedule=False), sort_keys=True).encode()).hexdigest()


def arm_protocol_sha256(cfg):
    """Full method signature, including checkpoint-defined theta_update and interval."""
    return hashlib.sha256(json.dumps(_protocol(cfg, include_schedule=True), sort_keys=True).encode()).hexdigest()


def main(argv=None):
    parser = build_arg_parser("Matched zero-shot weak-decoder evaluation.")
    parser.add_argument("--checkpoint", "--dynamic_checkpoint", dest="checkpoint", default=None)
    parser.add_argument("--static_checkpoint", default=None)
    parser.add_argument("--beta", type=float, default=None, help="shared inverse temperature for ALL arms; requires normalized scores")
    parser.add_argument("--arm", default="learned_dynamic", help="comma-separated subset of " + ",".join(ARMS))
    parser.add_argument("--stages", default="greedy")
    parser.add_argument("--baseline_only", action="store_true", help="alias for --arm fixed_dualgreedy")
    args, overrides = parser.parse_known_args(argv)
    cfg = get_config(overrides)
    if args.checkpoint:
        cfg.eval.checkpoint = args.checkpoint
    arms = ["fixed_dualgreedy"] if args.baseline_only else args.arm.split(",")
    if not arms or len(set(arms)) != len(arms) or any(a not in ARMS for a in arms):
        raise ValueError(f"--arm must be a unique subset of {ARMS}")
    if any(arg.split("=", 1)[0] == "--model.theta_interval" for arg in overrides):
        if args.checkpoint or args.static_checkpoint or any(a.startswith("learned_") for a in arms):
            raise ValueError("theta_interval is inherited from each checkpoint; inference overrides are not allowed")
        if cfg.model.theta_interval != 1:
            raise ValueError("The fixed/untrained controls use theta_interval=1")
    stages = [s["name"] for s in parse_stages(args.stages.split(","), cfg.eval.num_samples)]
    stages.sort(key=lambda name: name != "greedy")
    if args.print_config:
        print(json.dumps({**cfg.to_dict(), "arms": arms, "stages": stages}, indent=2))
        return
    ec = cfg.eval
    if min(ec.max_rollouts_per_chunk, ec.ls_pool, ec.ls_batch) < 1:
        raise ValueError("rollout chunk, search pool and search batch must be positive")
    if ec.sample_temperature <= 0 or min(ec.ils_rounds, ec.ls_max_iters, ec.ls_drop_k) < 0:
        raise ValueError("temperature must be positive and search budgets nonnegative")
    device = get_device(cfg)
    models, configs, checkpoints, training_seed = _load_arms(cfg, arms, args.checkpoint, args.static_checkpoint, device)
    if args.beta is not None:
        if not math.isfinite(args.beta) or args.beta <= 0 or ec.sample_temperature != 1.0:
            raise ValueError("Shared beta requires finite positive beta and sample_temperature=1")
        for a in arms:
            if not configs[a].model.score_normalization:
                raise ValueError("Shared T1-R beta requires normalized checkpoints")
            configs[a].model.temperature_mode = "fixed"
            configs[a].model.temperature_init = 1.0 / args.beta
            if models[a] is not None:
                with torch.no_grad():
                    models[a].log_temperature.fill_(math.log(1.0/args.beta))
    envs = {a: MKPEnv(configs[a].env) for a in arms}
    if any(configs[a].env.feasibility_mode != "hard_mask" for a in arms):
        raise ValueError("Main arms require hard_mask; rejection-mode results need a named ablation")
    path = resolve_path(ec.mat_file)
    data = [d for d in read_mat(path) if d["M"] in ec.dims and d["N"] in ec.sizes]
    data = data[:ec.max_instances] if ec.max_instances > 0 else data
    if not data:
        raise RuntimeError("No benchmark instances match eval.dims/sizes")
    out_dir = Path(resolve_path(ec.results_dir)) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir.mkdir(parents=True)
    cfg.save(str(out_dir / "config.json"))
    source = source_provenance()
    (out_dir / "source_diff.patch").write_text(source.get("diff", ""), encoding="utf-8")
    meta = {"schema": "weak_decoder_benchmark_t1_v1", "status": "running", "source": source,
            "config": cfg.to_dict(), "arm_configs": {a: configs[a].to_dict() for a in arms},
            "dataset": path, "data_sha256": sha256_file(path), "instance_ids": [d["id"] for d in data],
            "checkpoints": checkpoints, "training_seed": training_seed,
            "evaluation_seed": cfg.device.seed, "arms": arms, "stages": stages,
            "protocol_sha256": {a: arm_protocol_sha256(configs[a]) for a in arms},
            "comparison_protocol_sha256": {a: protocol_sha256(configs[a]) for a in arms},
            "theta_schedules": {a: {"theta_update": configs[a].model.theta_update,
                                     "theta_interval": configs[a].model.theta_interval} for a in arms},
            "device": str(device), "amp": cfg.device.amp and device.type == "cuda", "primary_stage": "greedy",
            "primary_stage_evaluated": "greedy" in stages,
            "reference": "Chu best-known values; not proven optima",
            "timing": "GPU synchronized; root LP shared per instance, charged once to each arm/stage. "
                      "Residual LP refresh time is reported separately and included in total LP time. "
                      "Construction excludes residual LP but includes per-step theta trace capture/CPU copies. Decision-point theta "
                      "statistics, JSON serialization, artifact output, audits and model loading excluded.",
            "protocol": "sampleK is K samples excluding greedy. ILS retains duplicate starts for matched budgets. "
                        "Stage RNG resets by evaluation seed, instance ID and stage, independently of arm.",
            "schedule_comparison": "Each learned arm inherits its trained theta_interval without an inference "
                                   "override. Full protocol SHA includes update mode and interval. Comparison "
                                   "SHA excludes only theta_update, theta_interval and untrained_random_heads "
                                   "to permit the declared static/dynamic control comparison; paired reports "
                                   "retain both full SHAs and schedules. Fixed/untrained controls use k=1.",
            "historical_reference": "fixed_dualgreedy uses static capacities/root dual. Its sampling extension "
                                    "softmaxes log ratio; this arm is not a matched-executor control.",
            "shared_beta_override": args.beta,
            "temperature_control": "Fixed-mode normalized checkpoints share configured beta; --beta overrides every arm. Learned-mode comparisons: fixed/untrained use the configured initial global T; learned arms use "
                                   "their fitted global T. Greedy is invariant to T. Sampling comparisons "
                                   "therefore include the effect of learning a global temperature.",
            "evidence": "One training seed per invocation; current T1-P protocol specifies seed 1234. "
                        "Wilcoxon pairs are instances within this seed, not evidence across training seeds. "
                        "This evaluator never selects weights."}
    write_json(out_dir / "meta.json", meta)
    rows = []
    try:
        with (out_dir / "trajectories.jsonl").open("w", encoding="utf-8") as artifact_file:
            for index, d in enumerate(data):
                inst = instances_from_dicts([d], device=device)
                _sync(device)
                start = time.perf_counter()
                pre = envs[arms[0]].precompute(inst)
                _sync(device)
                lp_seconds = time.perf_counter() - start
                alpha = round(float(inst.c.sum() / inst.r.sum()), 2)
                for arm in arms:
                    for stage in stages:
                        def sink(record):
                            artifact_file.write(json.dumps(jsonable({"id": d["id"], "seed": training_seed, **record})) + "\n")
                        rec = evaluate_stage(models[arm], envs[arm], inst, ec, device, meta["amp"], pre,
                                             stage, arm, cfg.device.seed, d["id"], lp_seconds, sink)
                        sol = rec.pop("solution")
                        row = {"id": d["id"], "M": d["M"], "N": d["N"], "alpha": alpha,
                               "arm": arm, "seed": training_seed, "evaluation_seed": cfg.device.seed,
                               "data_sha256": meta["data_sha256"], "protocol_sha256": meta["protocol_sha256"][arm],
                               "comparison_protocol_sha256": meta["comparison_protocol_sha256"][arm],
                               "theta_update": configs[arm].model.theta_update,
                               "theta_interval": configs[arm].model.theta_interval,
                               "training_signal": configs[arm].train.signal if arm.startswith("learned_") else "none",
                               "best_known": float(d["best"]), **rec}
                        row["gap_pct"] = (row["best_known"] - rec["objective"]) / row["best_known"] * 100
                        rows.append(row)
                        sink({"arm": arm, "stage": stage, "best_solution": sol, "objective": row["objective"]})
                        print(f"[{index + 1}/{len(data)}] id={d['id']} {arm} {stage} "
                              f"gap={row['gap_pct']:.4f}% time={row['time_total_s']:.2f}s", flush=True)
                write_csv(out_dir / "per_instance.csv", rows)
                artifact_file.flush()
        write_csv(out_dir / "summary.csv", summarize(rows))
        pairs, statistics = paired_report(rows)
        write_csv(out_dir / "paired_per_instance.csv", pairs)
        write_csv(out_dir / "paired_summary.csv", statistics)
        meta["status"], meta["row_count"] = "complete", len(rows)
    except Exception as exc:
        meta["status"], meta["error"] = "failed", repr(exc)
        raise
    finally:
        write_json(out_dir / "meta.json", meta)
    print(f"Results: {out_dir}")
    return out_dir


if __name__ == "__main__":
    main(sys.argv[1:])
