"""Two policy-side T3 checks; residual LP precision is tested separately."""
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from mkp_config import Config, config_from_dict, resolve_path
from mkp_env import MKPEnv, generate_instances, instances_from_dicts
from mkp_Model import build_model
from legacy_t0 import legacy_modules
from read_mat import read_mat

class T3PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    @torch.no_grad()
    def test_default_greedy_matches_t1p_snapshot_and_checkpoints(self):
        modules,_,_=legacy_modules("4d3c376")
        data=[d for d in read_mat(resolve_path("MKP_dataset/Benchmarks_Chu.mat")) if d["N"]==100 and d["M"] in (5,10)]
        cases=[next(d for d in data if d["M"]==m) for m in (5,10)]
        for arm in ("fixed","t1p_static_s1234","t1p_dynamic_k8_s1234"):
            if arm=="fixed":
                cfg=Config();cfg.model.temperature_init=.125
                old_cfg=modules["mkp_config"].config_from_dict(cfg.to_dict())
                old=modules["mkp_Model"].build_model(old_cfg).eval();model=build_model(cfg).eval()
                kwargs={"theta_override":"fixed"}
            else:
                path=resolve_path(f"checkpoints/{arm}/best.pt")
                if not Path(path).is_file():
                    self.skipTest("Existing T1-P checkpoint needed for the requested regression")
                ck=torch.load(path,map_location="cpu",weights_only=False)
                cfg=config_from_dict(ck["config"])
                old_cfg=modules["mkp_config"].config_from_dict(ck["config"])
                old=modules["mkp_Model"].build_model(old_cfg).eval();old.load_state_dict(ck["model"])
                model=build_model(cfg).eval();model.load_state_dict(ck["model"]);kwargs={}
            env=MKPEnv(cfg.env)
            for d in cases:
                with self.subTest(arm=arm,M=d["M"]):
                    inst=instances_from_dicts([d]);pre=env.precompute(inst)
                    before=old(env,inst,decode="greedy",pre=pre,**kwargs)
                    after=model(env,inst,decode="greedy",pre=pre,**kwargs)
                    for key in ("solution","profit","log_likelihood"):
                        torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
                    # The extended LP entry point must retain its default root arithmetic.
                    old_mu,old_ub=modules["mkp_env"].lp_dual(inst.p,inst.r,inst.c,cfg.env.dual_iters,cfg.env.dual_lr)
                    torch.testing.assert_close(pre[0],old_mu,rtol=0,atol=0)
                    torch.testing.assert_close(pre[1],old_ub,rtol=0,atol=0)

    @torch.no_grad()
    def test_one_lp_refresh_blocked_basis_and_raw_score_per_scheduled_step(self):
        cfg=Config();cfg.model.embed_dim=16;cfg.model.num_heads=2
        cfg.model.encoder_layers=cfg.model.controller_layers=1;cfg.model.edge_hidden=8
        cfg.model.theta_groups=["price","alpha","block","lp"]
        cfg.model.price_anchor="residual";cfg.model.lp_source="residual";cfg.model.lp_interval=1
        cfg.env.dual_iters=5;cfg.env.residual_lp_iters=3
        env=MKPEnv(cfg.env);model=build_model(cfg).eval()
        inst=generate_instances(1,14,3,[.75],generator=torch.Generator().manual_seed(318));pre=env.precompute(inst)
        from mkp_basis import blocked_mass
        for override in ("fixed",None):
            times=[];refresh=env.refresh_lp
            def wrapped(st):times.append(st.t);return refresh(st)
            with patch.object(env,"refresh_lp",side_effect=wrapped),patch.object(model.decoder,"raw_scores",wraps=model.decoder.raw_scores) as raw,patch("mkp_basis.blocked_mass",wraps=blocked_mass) as blocked:
                out=model(env,inst,decode="greedy",pre=pre,theta_override=override)
            self.assertGreater(out["steps"],1)
            self.assertEqual(times,list(range(1,out["steps"])))
            self.assertEqual(raw.call_count,out["steps"])
            self.assertEqual(blocked.call_count,out["steps"])
            self.assertEqual(out["lp_refreshes"],out["steps"]-1)
            load=torch.einsum("bmn,bn->bm",inst.r,out["solution"])
            self.assertTrue(bool((load<=inst.c+1e-5).all()))
            self.assertFalse(bool(((inst.r<=(inst.c-load)[:,:,None]+1e-5).all(1)&(out["solution"]==0)).any()))

if __name__=="__main__":unittest.main()
