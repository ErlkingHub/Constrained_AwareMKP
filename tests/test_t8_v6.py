"""The three T8 checks; full 180-case D/G reproduction remains user-run."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
from mkp_config import Config,config_from_dict
from mkp_Model import build_model
from mkp_env import MKPEnv as NewEnv,generate_instances
from mkp_v6_env import MKPEnv
from mkp_v6_lp import lp_relaxation
from mkp_v6exec import StrategyTheta,WeakDecoder
from mkp_v6_config import ModelConfig as ExecutorConfig
from mkp_v6_ppo import generalized_advantage,shaped_rewards

ROOT=Path(__file__).resolve().parents[1]
LEGACY=Path(os.environ.get("T5_LEGACY_ROOT","D:/PhD_Doctor/Study/Codes/RL_ModelInhance"))
MEANS=ROOT/"results/t5_v6/attribution/theta_means.json"


def tiny():
    c=Config();c.model.executor=c.model.theta_family="v6";c.model.ctrl_residual_lp_features=True
    c.model.embed_dim=16;c.model.num_heads=2;c.model.encoder_layers=c.model.controller_layers=1;c.model.edge_hidden=8
    return c


class T8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_executor_and_lp_match_six_legacy_shapes(self):
        if not (LEGACY/"mkp_Model.py").exists() or not MEANS.exists():self.skipTest("External v6 source/checkpoints/means required")
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"old.npz"
            result=subprocess.run([sys.executable,"-B",str(ROOT/"scripts/t8_legacy_parity.py"),"--legacy_root",str(LEGACY),
                "--checkpoints",str(ROOT/"checkpoints/lp_dual_v6"),"--means",str(MEANS),"--output",str(path),"--device","cpu"],
                capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=600)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            means=json.loads(MEANS.read_text(encoding="utf-8"))
            with np.load(path,allow_pickle=False) as archive:
                cases=json.loads(str(archive["metadata"].item()))
                self.assertEqual(len(cases),12)
                for index,case in enumerate(cases):
                    cfg=ExecutorConfig(**case["config"]);decoder=WeakDecoder(cfg)
                    env=MKPEnv(*(torch.tensor(archive[f"{n}_{index}"]).unsqueeze(0) for n in ("p","w","c")))
                    theta=StrategyTheta(**{k:torch.tensor(v,dtype=torch.float32) for k,v in means[case["arm"]][str(case["M"])].items()})
                    sequence=[]
                    while bool(env.state.active[0]):
                        t=len(sequence);bound=lp_relaxation(env.state,cfg).bound;policy=decoder(env.state,theta)
                        np.testing.assert_allclose(bound.numpy(),archive[f"bound_{index}_{t}"],rtol=1e-6,atol=1e-6)
                        np.testing.assert_array_equal(policy.items.numpy(),archive[f"items_{index}_{t}"])
                        np.testing.assert_allclose(policy.score_terms["score"].numpy(),archive[f"scores_{index}_{t}"],rtol=1e-5,atol=1e-5)
                        action=policy.choose(False);sequence.append(int(action[0]));env.step(action)
                    np.testing.assert_array_equal(sequence,archive[f"sequence_{index}"])
                    np.testing.assert_array_equal(env.state.selected.numpy(),archive[f"selected_{index}"])

    def test_initialization_and_default_checkpoint_compatibility(self):
        from legacy_t0 import legacy_modules
        for seed in (14,915):
            for m in (5,10):
                torch.manual_seed(seed);c=tiny();model=build_model(c).eval();inst=generate_instances(1,9,m)
                state=MKPEnv(inst.p.double(),inst.r.transpose(1,2).double(),inst.c.double()).state
                with torch.no_grad():
                    theta=model.policy(state)[1];default=model.v6_engine.default_theta(state)
                    source=json.loads(MEANS.read_text(encoding="utf-8"))["D"][str(m)] if MEANS.exists() else None
                    for n in theta.__dataclass_fields__:
                        torch.testing.assert_close(getattr(theta,n),getattr(default,n),rtol=1e-6,atol=1e-7)
                        if source is not None:
                            torch.testing.assert_close(getattr(theta,n),torch.tensor(source[n]),rtol=1e-6,atol=1e-7)
                    a=model(NewEnv(c.env),inst,decode="greedy");b=model(NewEnv(c.env),inst,decode="greedy",theta_override="fixed")
                    self.assertTrue(torch.equal(a["solution"],b["solution"]))
        modules,_,_=legacy_modules("4d3c376")
        for run in ("t1p_static_s1234","t1p_dynamic_k8_s1234"):
            path=ROOT/"checkpoints"/run/"best.pt"
            if not path.exists():self.skipTest("T1-P checkpoint required")
            ck=torch.load(path,map_location="cpu",weights_only=False);c=config_from_dict(ck["config"])
            self.assertEqual(c.model.executor,"g0g1")
            new=build_model(c).eval();new.load_state_dict(ck["model"])
            old=modules["mkp_Model"].build_model(modules["mkp_config"].config_from_dict(ck["config"])).eval();old.load_state_dict(ck["model"])
            inst=generate_instances(1,9,5);env=NewEnv(c.env);pre=env.precompute(inst)
            with torch.no_grad():a=new(env,inst,decode="greedy",pre=pre);b=old(env,inst,decode="greedy",pre=pre)
            self.assertTrue(torch.equal(a["solution"],b["solution"]));self.assertTrue(torch.equal(a["log_likelihood"],b["log_likelihood"]))

    def test_gae_shaping_and_unclipped_policy_gradient(self):
        r=torch.tensor([[.2,.4],[.3,0.],[.5,0.]])
        v=torch.tensor([[.1,.2],[.2,0.],[.3,0.]])
        active=torch.tensor([[1,1],[1,0],[1,0]],dtype=torch.bool)
        done=torch.tensor([[0,1],[0,1],[1,1]],dtype=torch.bool)
        adv,ret=generalized_advantage(r,v,done,active,1.,.95)
        expected=torch.tensor([[.3+.95*(.4+.95*.2),.2],[.4+.95*.2,0.],[.2,0.]])
        torch.testing.assert_close(adv,expected);torch.testing.assert_close(ret,expected+v)
        phi=torch.tensor([[.9,.7],[.6,0.],[.2,0.],[0.,0.]])
        shaped=shaped_rewards(r,phi,active)
        torch.testing.assert_close(shaped.sum(0),r.sum(0)-phi[0])
        logits=torch.tensor([[.1,.2,-.4],[.2,-.3,.1]],requires_grad=True)
        actions=torch.tensor([0,2]);a=torch.tensor([.7,-.4])
        logp=logits.log_softmax(-1)[torch.arange(2),actions];old=logp.detach()
        ratio=(logp-old).exp();ppo=-(torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a)).mean()
        gp=torch.autograd.grad(ppo,logits,retain_graph=True)[0]
        plain=-(logp*a).mean();pg=torch.autograd.grad(plain,logits)[0]
        torch.testing.assert_close(gp,pg,rtol=0,atol=0)


if __name__=="__main__":unittest.main()
