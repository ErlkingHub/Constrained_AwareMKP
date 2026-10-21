"""
Benchmark_zeroShot.py
=====================
Zero-shot evaluation of a trained policy on Benchmarks_Chu.mat (never seen in training).

For every instance:
  1. greedy rollout of the policy                         -> gap_greedy
  2. `num_samples` sampled rollouts (+ greedy), best      -> gap_policy  (pure learned policy)
  3. optional refinement of the best `ls_pool` distinct
     policy solutions (mkp_refine: swap / drop-refill / ILS) -> gap_final
  Reference: static LP-dual ratio greedy (Pirkul / Chu & Beasley)  -> gap_dual_greedy

gap = (best_known - obj) / best_known * 100

Outputs (results/<timestamp>/):
  per_instance.csv   one row per instance
  summary.csv        means grouped by (M, N, alpha), by (M, N) and overall
  config.json        configuration used

    python Benchmark_zeroShot.py
    python Benchmark_zeroShot.py --checkpoint checkpoints/best.pt --eval.num_samples 512
    python Benchmark_zeroShot.py --eval.dims 5,10,30 --eval.ls_enable false
    python Benchmark_zeroShot.py --baseline_only      # no model: dual-greedy (+ refinement)
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch

from mkp_config import get_config, get_device, build_arg_parser, resolve_path
from mkp_env import MKPEnv, instances_from_dicts, greedy_dual_ratio
from mkp_refine import iterated_local_search, is_feasible, greedy_fill, _ratio
from mkp_Trainer import load_policy, set_seed
from read_mat import read_mat


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _top_distinct(sol: torch.Tensor, obj: torch.Tensor, k: int):
    uniq, inv = torch.unique(sol, dim=0, return_inverse=True)
    best_obj = torch.full((uniq.shape[0],), -1.0, device=obj.device).scatter_reduce(
        0, inv, obj, reduce="amax")
    order = best_obj.argsort(descending=True)[:k]
    return uniq[order], best_obj[order]


@torch.no_grad()
def evaluate_instance(model, env, inst, ec, device, amp, gen):
    """inst: MKPInstances with B == 1. Returns a dict of metrics."""
    rec = {}
    _sync(device); t0 = time.time()
    mu, ub = env.precompute(inst)
    rec["ub_lp"] = float(ub[0])
    rec["dual_greedy_obj"] = float(greedy_dual_ratio(inst, mu)[0])
    N = inst.N
    sols, objs = [], []
    if model is not None:
        g = model(env, inst, num_samples=1, decode="greedy", amp=amp, pre=(mu, ub))
        rec["greedy_obj"] = float(g["profit"][0])
        rec["greedy_rejects"] = float(g["n_reject"][0])
        rec["greedy_steps"] = int(g["steps"])
        rec["kappa"] = g["kappa"]
        for name, w in zip(g["primitives"], g["prim_w"]):
            rec[f"w_{name}"] = w
        rec["abs_residual"] = g["abs_residual"]
        rec["abs_log_price"] = g["log_price_abs"]
        sols.append(g["solution"]); objs.append(g["profit"])
        S = ec.num_samples
        rej, steps = [], []
        while S > 0:
            k = min(S, ec.max_rollouts_per_chunk)
            o = model(env, inst, num_samples=k, decode="sample", temperature=ec.sample_temperature,
                      amp=amp, pre=(mu, ub))
            sols.append(o["solution"]); objs.append(o["profit"])
            rej.append(o["n_reject"]); S -= k
        allrej = torch.cat(rej) if rej else torch.zeros(1, device=device)
        rec["sample_mean_rejects"] = float(allrej.mean())
        rec["sample_feasible_pick_rate"] = float(1.0 - (allrej / max(N, 1)).mean())
    else:
        # baseline: dual greedy + noisy dual greedy starts
        T = max(ec.ls_pool, 1)
        rp = inst.repeat(T)
        sc = torch.log(_ratio(rp.p, rp.r, rp.c, mu.repeat(T, 1)))
        noise = 0.3 * torch.rand(sc.shape, generator=gen, device="cpu").to(device)
        noise[0] = 0
        x0 = greedy_fill(torch.zeros(T, N, device=device), rp.p, rp.r, rp.c, sc + noise)
        sols.append(x0); objs.append((x0 * rp.p).sum(1))
    sol = torch.cat(sols, 0); obj = torch.cat(objs, 0)
    assert bool(is_feasible(sol, inst.r.expand(sol.shape[0], -1, -1), inst.c.expand(sol.shape[0], -1)).all())
    rec["policy_obj"] = float(obj.max())
    rec["policy_mean_obj"] = float(obj.mean())
    _sync(device); rec["time_policy_s"] = time.time() - t0

    final = rec["policy_obj"]
    t1 = time.time()
    if ec.ls_enable:
        pool, _ = _top_distinct(sol, obj, ec.ls_pool)
        for s in range(0, pool.shape[0], ec.ls_batch):
            x = pool[s:s + ec.ls_batch]
            b = x.shape[0]
            rp = inst.repeat(b)
            bx, bo = iterated_local_search(x, rp.p, rp.r, rp.c, mu.repeat(b, 1), rounds=ec.ils_rounds,
                                           perturb_max=ec.ils_perturb_max, max_iters=ec.ls_max_iters,
                                           drop_k=ec.ls_drop_k, generator=gen)
            assert bool(is_feasible(bx, rp.r, rp.c).all())
            final = max(final, float(bo.max()))
    _sync(device)
    rec["time_refine_s"] = time.time() - t1
    rec["final_obj"] = final
    rec["time_total_s"] = time.time() - t0
    return rec


def main(argv=None):
    parser = build_arg_parser("Zero-shot benchmark on Benchmarks_Chu.mat.")
    parser.add_argument("--checkpoint", type=str, default=None, help="overrides eval.checkpoint")
    parser.add_argument("--baseline_only", action="store_true",
                        help="no model: dual-ratio greedy starts (+ refinement if enabled)")
    args, overrides = parser.parse_known_args(argv)
    cfg = get_config(overrides)
    if args.checkpoint:
        cfg.eval.checkpoint = args.checkpoint
    if args.print_config:
        print(json.dumps(cfg.to_dict(), indent=2)); return
    set_seed(cfg.device.seed)
    device = get_device(cfg)
    ec = cfg.eval

    model, mcfg = None, cfg
    ckpt_name = "baseline"
    if not args.baseline_only:
        ck_path = resolve_path(ec.checkpoint)
        if not os.path.isfile(ck_path):
            raise FileNotFoundError(f"checkpoint not found: {ck_path} (train first, or use --baseline_only)")
        model, mcfg, ck = load_policy(ck_path, device, cfg_override=cfg)
        ckpt_name = f"{ck.get('run_name', 'model')}_ep{ck.get('epoch', '?')}"
        print(f"[bench] loaded {ck_path} (run {ck.get('run_name')}, epoch {ck.get('epoch')}, "
              f"best val {ck.get('best_val')})")
    env = MKPEnv(mcfg.env)
    amp = cfg.device.amp and device.type == "cuda"
    gen = torch.Generator().manual_seed(cfg.device.seed)

    data = read_mat(resolve_path(ec.mat_file))
    data = [d for d in data if d["M"] in ec.dims and d["N"] in ec.sizes]
    if ec.max_instances > 0:
        data = data[:ec.max_instances]
    if not data:
        raise RuntimeError("no benchmark instance matches eval.dims / eval.sizes")

    out_dir = resolve_path(os.path.join(ec.results_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{ckpt_name}"))
    os.makedirs(out_dir, exist_ok=True)
    cfg.save(os.path.join(out_dir, "config.json"))
    per_path = os.path.join(out_dir, "per_instance.csv")
    print(f"[bench] {len(data)} instances -> {out_dir}")

    rows = []
    fields = None
    for i, d in enumerate(data):
        inst = instances_from_dicts([d], device=device)
        alpha = float(np.round(float(inst.c.sum() / inst.r.sum()), 2))
        rec = evaluate_instance(model, env, inst, ec, device, amp, gen)
        best = float(d["best"])
        gap = lambda v: (best - v) / best * 100.0
        row = {"id": d["id"], "M": d["M"], "N": d["N"], "alpha": alpha, "best_known": best, **rec}
        row["gap_dual_greedy"] = gap(rec["dual_greedy_obj"])
        if "greedy_obj" in rec:
            row["gap_greedy"] = gap(rec["greedy_obj"])
        row["gap_policy"] = gap(rec["policy_obj"])
        row["gap_final"] = gap(rec["final_obj"])
        row["gap_lp_bound_to_best"] = (rec["ub_lp"] - best) / best * 100.0
        rows.append(row)
        if fields is None:
            fields = list(row.keys())
        with open(per_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if i == 0:
                w.writeheader()
            w.writerow(row)
        print(f"[{i + 1}/{len(data)}] id {d['id']:3d} M{d['M']} N{d['N']} a{alpha:.2f} | "
              + (f"greedy {row['gap_greedy']:.3f}% | " if "gap_greedy" in row else "")
              + f"policy {row['gap_policy']:.3f}% | final {row['gap_final']:.3f}% | "
              f"dual-greedy {row['gap_dual_greedy']:.3f}% | {rec['time_total_s']:.1f}s", flush=True)

    # ---------------------------------------------------------------- summary
    metric_keys = [k for k in rows[0] if k.startswith("gap_") or k.startswith("time_")
                   or k in ("greedy_rejects", "sample_mean_rejects", "sample_feasible_pick_rate", "kappa",
                            "abs_residual", "abs_log_price") or k.startswith("w_")]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["M"], r["N"], r["alpha"])].append(r)
        groups[(r["M"], r["N"], "all")].append(r)
        groups[("all", "all", "all")].append(r)

    def sort_key(k):
        return tuple((1e9 if v == "all" else float(v)) for v in k)

    summ = []
    for key in sorted(groups, key=sort_key):
        rs = groups[key]
        s = {"M": key[0], "N": key[1], "alpha": key[2], "count": len(rs)}
        for mk in metric_keys:
            vals = [r[mk] for r in rs if mk in r]
            s[f"mean_{mk}"] = float(np.mean(vals)) if vals else ""
        s["max_gap_final"] = float(np.max([r["gap_final"] for r in rs]))
        s["n_optimal_final"] = int(sum(r["gap_final"] <= 1e-9 for r in rs))
        summ.append(s)
    sum_path = os.path.join(out_dir, "summary.csv")
    with open(sum_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader(); w.writerows(summ)

    print("\n=========== Zero-shot summary (gap % vs best known) ===========")
    hdr = f"{'M':>4} {'N':>4} {'cnt':>4} | {'greedy':>8} {'policy':>8} {'final':>8} {'dualGr':>8} | {'time(s)':>8}"
    print(hdr); print("-" * len(hdr))
    for s in summ:
        if s["alpha"] != "all":
            continue
        g = s.get("mean_gap_greedy", "")
        print(f"{str(s['M']):>4} {str(s['N']):>4} {s['count']:>4} | "
              f"{(f'{g:8.4f}' if g != '' else '     n/a')} {s['mean_gap_policy']:8.4f} {s['mean_gap_final']:8.4f} "
              f"{s['mean_gap_dual_greedy']:8.4f} | {s['mean_time_total_s']:8.2f}")
    print(f"\nper-instance -> {per_path}\nsummary      -> {sum_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
