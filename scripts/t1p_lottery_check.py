"""D1: is the B2 per-instance best theta a smooth basin or an isolated lucky point?"""
import json, math, sys, statistics as st
root = sys.argv[1]
sys.path[:0] = [root, root + "/scripts"]
import os; os.chdir(root)
import torch
from t1r_common import config, synthetic, audit
from mkp_env import MKPEnv
from mkp_Model import build_model, theta_from_pre

torch.manual_seed(0); torch.set_num_threads(8)
cfg = config("runs/t1p_static_s1234/config.json")
env = MKPEnv(cfg.env); model = build_model(cfg).eval()
best = {d["id"]: d for d in json.load(open("results/t1r_headroom/best_theta.json"))}
K = 32
SCALES = (0.25, 1.0)

def gaps(inst, pre, u, a, ref):
    out = model(env, inst, num_samples=u.shape[0], decode="greedy", pre=pre,
                theta_override={"u": u, "alpha": a})
    audit(inst, out)
    return 100 * (1 - out["profit"] / ref)

def perturbed(u, a, scale, g):
    v = torch.atanh((u / 3).clamp(-.999, .999)).expand(K, -1)
    z = torch.logit(((a - .5) / 2).clamp(1e-4, 1 - 1e-4)).expand(K)
    v = v + scale * .05 * torch.randn(v.shape, generator=g)
    z = z + scale * .3 * torch.randn(z.shape, generator=g)
    th = theta_from_pre(v, z, 3.0)
    return th["u"], th["alpha"]

rows = []
g = torch.Generator().manual_seed(1)
with torch.no_grad():
    for name, inst, pre, refs in synthetic(cfg, torch.device("cpu")):
        for b in range(inst.B):
            key = f"{name}_{b}"; d = best[key]
            one = type(inst)(inst.p[b:b+1], inst.r[b:b+1], inst.c[b:b+1])
            p1 = (pre[0][b:b+1], pre[1][b:b+1]); ref = float(refs["obj"][b])
            u_star = torch.tensor(d["u"])[None]; a_star = torch.tensor([d["alpha_best"]])
            u0 = torch.zeros_like(u_star); a0 = torch.ones(1)
            r = {"id": key, "tight": round(float(one.c.sum() / one.r.sum()), 2),
                 "fixed": float(gaps(one, p1, u0, a0, ref)[0]),
                 "star": float(gaps(one, p1, u_star, a_star, ref)[0])}
            for s in SCALES:
                for tag, (u, a) in (("star", (u_star, a_star)), ("fixed", (u0, a0))):
                    gp = gaps(one, p1, *perturbed(u, a, s, g), ref)
                    r[f"{tag}_pert{s}_mean"] = float(gp.mean())
                    r[f"{tag}_pert{s}_best"] = float(gp.min())
            rows.append(r)

def m(k): return st.mean(r[k] for r in rows)
print("n", len(rows))
for k in rows[0]:
    if k not in ("id", "tight"): print(f"{k:28s} {m(k):.3f}")
fx, sr = m("fixed"), m("star")
for s in SCALES:
    print(f"scale {s}: gain retained by perturbed theta* = {(fx - m(f'star_pert{s}_mean')) / (fx - sr):.2%}")
for t in sorted({r["tight"] for r in rows}):
    sub = [r for r in rows if r["tight"] == t]
    print(t, len(sub), {k: round(st.mean(r[k] for r in sub), 3) for k in ("fixed", "star", "star_pert1.0_mean", "fixed_pert1.0_mean", "fixed_pert1.0_best")})
