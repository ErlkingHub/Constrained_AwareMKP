"""Only the three T4 checks requested in the research guidance."""
import math
import subprocess
import unittest
from pathlib import Path

import torch
from mkp_config import Config, config_from_dict, resolve_path
from mkp_env import MKPEnv, generate_instances
from mkp_Model import build_model, theta_pre_log_prob
from mkp_Trainer import leave_one_out_max_advantage, MKPTrainer
from mkp_evaluation import parse_stages
from Benchmark_zeroShot import evaluate_stage


def tiny_config():
    c=Config();c.model.embed_dim=16;c.model.num_heads=2
    c.model.encoder_layers=c.model.controller_layers=1;c.model.edge_hidden=8
    c.env.dual_iters=5;c.model.theta_update="static";c.model.grad_checkpoint=False
    return c


class T4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    @torch.no_grad()
    def test_center_prefix_chunk_and_old_stage_specs(self):
        cfg=tiny_config();cfg.eval.max_rollouts_per_chunk=64
        env=MKPEnv(cfg.env);model=build_model(cfg).eval()
        inst=generate_instances(1,11,3,[.5],generator=torch.Generator().manual_seed(841));pre=env.precompute(inst)
        def stage(name):
            return evaluate_stage(model,env,inst,cfg.eval,torch.device("cpu"),False,pre,name,"fixed",1234,"fixture",return_candidates=True)
        one=stage("theta1");plain=model(env,inst,decode="greedy",pre=pre,theta_override="fixed")
        torch.testing.assert_close(one["candidate_solutions"],plain["solution"],rtol=0,atol=0)
        large=stage("theta256");small=stage("theta16")
        torch.testing.assert_close(large["candidate_solutions"][:16],small["candidate_solutions"],rtol=0,atol=0)
        torch.testing.assert_close(large["candidate_objectives"][:16],small["candidate_objectives"],rtol=0,atol=0)
        cfg.eval.max_rollouts_per_chunk=7
        chunked=stage("theta256")
        torch.testing.assert_close(large["candidate_solutions"],chunked["candidate_solutions"],rtol=0,atol=0)
        torch.testing.assert_close(large["candidate_objectives"],chunked["candidate_objectives"],rtol=0,atol=0)
        source=subprocess.run(["git","show","814aaab:mkp_evaluation.py"],capture_output=True,text=True,check=True,encoding="utf-8").stdout
        ns={"__name__":"_old_t3_evaluation"};exec(compile(source,"git:814aaab:mkp_evaluation.py","exec"),ns)
        old_names=["greedy","sample16","greedy+ILS","sampleK+ILS"]
        self.assertEqual(parse_stages(old_names),ns["parse_stages"](old_names))

    def test_leave_one_out_max_advantage_and_antithetic_rejection(self):
        R=torch.tensor([[1.,4.,2.,3.],[4.,4.,1.,2.],[2.,2.,2.,2.],[0.,1.,7.,2.]])
        expected=torch.tensor([[0.,1.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,5.,0.]])
        torch.testing.assert_close(leave_one_out_max_advantage(R),expected,rtol=0,atol=0)
        cfg=tiny_config();cfg.train.signal="theta_maxk";cfg.train.theta_pg_antithetic=True
        cfg.model.theta_sigma_head=True
        with self.assertRaisesRegex(ValueError,"antithetic|Antithetic"):
            MKPTrainer(cfg,torch.device("cpu"))

    def test_tensor_sigma_normal_density_and_legacy_checkpoint(self):
        means={"u_pre":torch.tensor([[.1,-.2,.3]],dtype=torch.float64,requires_grad=True),
               "alpha_pre":torch.tensor([-.8],dtype=torch.float64,requires_grad=True)}
        scales={"u_pre":torch.tensor([[.1,.2,.3]],dtype=torch.float64,requires_grad=True),
                "alpha_pre":torch.tensor([.4],dtype=torch.float64,requires_grad=True)}
        draws={"u_pre":torch.tensor([[.3,.1,-.2]],dtype=torch.float64),"alpha_pre":torch.tensor([-.2],dtype=torch.float64)}
        actual=theta_pre_log_prob(means,draws,scales)
        expected=torch.distributions.Normal(means["u_pre"],scales["u_pre"]).log_prob(draws["u_pre"]).sum(1)
        expected+=torch.distributions.Normal(means["alpha_pre"],scales["alpha_pre"]).log_prob(draws["alpha_pre"])
        torch.testing.assert_close(actual-expected,torch.full_like(actual,.5*4*math.log(2*math.pi)),rtol=1e-10,atol=1e-10)
        actual.sum().backward()
        self.assertTrue(all(bool(torch.isfinite(s.grad).all()) for s in scales.values()))
        from legacy_t0 import legacy_modules
        modules,_,_=legacy_modules("4d3c376")
        for name in ("t1p_static_s1234","t1p_dynamic_k8_s1234"):
            path=Path(resolve_path(f"checkpoints/{name}/best.pt"))
            if not path.exists():self.skipTest("T1-P checkpoint is needed for requested legacy regression")
            ck=torch.load(path,map_location="cpu",weights_only=False);cfg=config_from_dict(ck["config"])
            self.assertFalse(cfg.model.theta_sigma_head)
            model=build_model(cfg).eval();model.load_state_dict(ck["model"])
            self.assertEqual(set(model.state_dict()),set(ck["model"]))
            old=modules["mkp_Model"].build_model(modules["mkp_config"].config_from_dict(ck["config"])).eval();old.load_state_dict(ck["model"])
            env=MKPEnv(cfg.env);inst=generate_instances(1,9,5,generator=torch.Generator().manual_seed(92));pre=env.precompute(inst)
            with torch.no_grad():
                a=model(env,inst,decode="greedy",pre=pre);b=old(env,inst,decode="greedy",pre=pre)
            torch.testing.assert_close(a["solution"],b["solution"],rtol=0,atol=0)
            torch.testing.assert_close(a["log_likelihood"],b["log_likelihood"],rtol=0,atol=0)
        # New zero-initialized scale heads equal the registered isotropic proposal.
        cfg=tiny_config();cfg.model.theta_sigma_head=True;cfg.model.theta_sigma_dims="a"
        env=MKPEnv(cfg.env);inst=generate_instances(1,9,3,generator=torch.Generator().manual_seed(1))
        with torch.no_grad():proposal=build_model(cfg).eval().proposal_parameters(env,inst)
        torch.testing.assert_close(proposal["sigma_u"],torch.zeros_like(proposal["sigma_u"]),rtol=0,atol=0)
        torch.testing.assert_close(proposal["sigma_alpha"],torch.full_like(proposal["sigma_alpha"],.3),rtol=0,atol=0)

if __name__=="__main__":unittest.main()
