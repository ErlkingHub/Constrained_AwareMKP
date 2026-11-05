"""Shared synthetic validation data and provenance for T1-R tools."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import torch
from mkp_config import config_from_dict, resolve_path
from mkp_env import MKPEnv, generate_instances, instances_from_dicts
from mkp_artifacts import source_provenance, instance_sha, sha256_file, write_json
from mkp_reference import milp_reference
from mkp_evaluation import write_csv
from mkp_Model import build_model
from read_mat import read_mat

def config(path):
    cfg = config_from_dict(json.loads(Path(resolve_path(path)).read_text(encoding="utf-8")))
    cfg.model.score_normalization = True
    cfg.model.temperature_mode = "fixed"
    return cfg

def synthetic(cfg, device, count=None):
    g = torch.Generator().manual_seed(cfg.train.val_seed)
    env = MKPEnv(cfg.env)
    for n in cfg.train.val_sizes:
        for m in cfg.train.val_dims:
            cpu = generate_instances(cfg.train.val_instances, n, m, cfg.env.train_tightness,
                    cfg.env.weight_max, cfg.env.profit_noise, generator=g)
            if count: cpu = type(cpu)(cpu.p[:count], cpu.r[:count], cpu.c[:count])
            refs = milp_reference(cpu, cfg.train.val_milp_time_limit, cfg.train.val_mip_rel_gap,
                                  cfg.train.reference_cache)
            inst = cpu.to(device)
            yield f"N{n}_M{m}", inst, env.precompute(inst), refs

def audit(inst, out):
    x = out["solution"]
    rp = inst.repeat(x.shape[0]//inst.B)
    rem = rp.c-torch.einsum("bmn,bn->bm",rp.r,x)
    assert bool((rem>=-1e-5).all())
    assert not bool(((rp.r<=rem[:,:,None]+1e-6).all(1)&(x==0)).any())

def fresh(path):
    p = Path(resolve_path(path))
    if p.exists(): raise FileExistsError(f"Preserve existing results: {p}")
    p.mkdir(parents=True)
    return p
