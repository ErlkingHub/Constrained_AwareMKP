"""Five requested checks: dual identities, bounds, compatibility, LP certificates."""
import itertools
import unittest
import numpy as np
import torch
from scipy.optimize import linprog
from mkp_config import Config
from mkp_Model import build_model
from mkp_env import MKPInstances,MKPEnv as NewEnv
from mkp_v6_env import MKPEnv
from mkp_v6_config import ModelConfig
from mkp_v6_lp import lp_relaxation
from mkp_cert_basis import cert_basis,dual_identity
from mkp_cert_exec import CertDecoder,constant_theta


def fixture(seed,n=9,m=3):
    g=torch.Generator().manual_seed(seed)
    p=torch.randint(1,40,(1,n),generator=g).double()
    w=torch.randint(1,15,(1,n,m),generator=g).double()
    c=(w.sum(1)*(.2+.45*(seed%4)/3)).floor().clamp_min(14)
    return MKPEnv(p,w,c)


def enumerated(state):
    n=state.profits.shape[1]
    x=torch.tensor(list(itertools.product((False,True),repeat=n)))
    legal=(~x|state.feasible[0]).all(1)&(x.double()@state.weights[0]<=state.remaining[0]).all(1)
    return x[legal],x[legal].double()@state.profits[0]


class CertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_i1_gap_identity(self):
        for seed in range(15):
            state=fixture(seed).state;x,values=enumerated(state)
            # Every chosen completion is feasible; maximum-cardinality tie makes it maximal.
            selected=x[values.argmax()].unsqueeze(0)
            prices=torch.rand(1,3,generator=torch.Generator().manual_seed(seed)).double()*10
            dual,parts=dual_identity(state,selected,prices)
            torch.testing.assert_close(dual-(state.profits*selected).sum(-1),parts.sum(-1),rtol=1e-9,atol=1e-9)

    def test_i2_one_step_identity(self):
        for seed in range(12):
            state=fixture(seed).state;prices=torch.rand(1,3,dtype=torch.float64)*7
            basis=cert_basis(state,ModelConfig(),prices=prices)
            for item in state.feasible[0].nonzero().flatten():
                child=MKPEnv.from_state(state);child.step(item.reshape(1))
                dual,_=dual_identity(child.state,torch.zeros_like(child.state.selected),prices)
                expected=basis.dual_value-(basis.values[:,item,1]+basis.values[:,item,2])*basis.scale
                torch.testing.assert_close(state.profits[:,item]+dual,expected,rtol=1e-9,atol=1e-9)

    def test_p1_bracket_and_p3_terminal_optimality(self):
        terminal_checked=0
        for seed in range(30):
            env=fixture(seed);state=env.state;xs,profits=enumerated(state)
            b=cert_basis(state,ModelConfig())
            for item in state.feasible[0].nonzero().flatten():
                optimum=profits[xs[:,item]].max()
                lower=state.profits[0,item]+b.values[0,item,4]*b.scale[0]
                upper=b.dual_value[0]-(b.values[0,item,1]+b.values[0,item,2])*b.scale[0]
                self.assertLessEqual(float(lower),float(optimum)+1e-6);self.assertLessEqual(float(optimum),float(upper)+1e-6)
            # Reach a terminal region with at most two jointly feasible items.
            while int(xs.sum(1).max())>2:
                env.step(state.feasible[0].nonzero()[0].reshape(1));xs,profits=enumerated(state)
            optimum=float(profits.max());initial=float(env.objective[0])
            decoder=CertDecoder(ModelConfig())
            while bool(state.active[0]):env.step(decoder(state,constant_theta('K-pair')).choose(False))
            self.assertAlmostEqual(float(env.objective[0])-initial,optimum,places=7);terminal_checked+=1
        self.assertEqual(terminal_checked,30)

    def test_initialization_and_compatibility(self):
        from scripts.t9_interface import Probe,legacy_features
        from mkp_v6exec import WeakDecoder,StrategyTheta
        from scripts.t7_interface import global_draws
        state=fixture(17).state;config=ModelConfig()
        theta=global_draws(config,3,4,torch.Generator().manual_seed(71))
        split=legacy_features(state,config).logits(theta)
        repeated=state.index(torch.zeros(4,dtype=torch.long))
        direct=WeakDecoder(config)(repeated,StrategyTheta(**theta))
        torch.testing.assert_close(split,direct.logits.reshape(4,-1),rtol=1e-5,atol=1e-5)
        for family in ('cert+x_lp','cert+scar','cert+comp3','cert+bneck','cert+inter1','cert+rand2','cert+scar+comp3'):
            probe=Probe(state,config,family,[.2]*5,8,'unit',0);scores,_=probe.scores()
            self.assertTrue(bool(torch.isfinite(scores).all()))
        for seed in (51,91):
            torch.manual_seed(seed);cfg=Config();cfg.model.executor=cfg.model.theta_family='cert';cfg.train.signal='item_pg'
            cfg.model.ctrl_residual_lp_features=cfg.model.ctrl_basis_features=True
            cfg.model.embed_dim=16;cfg.model.num_heads=2;cfg.model.encoder_layers=cfg.model.controller_layers=1;cfg.model.edge_hidden=8
            model=build_model(cfg).eval();state=fixture(seed).state
            self.assertFalse(hasattr(model.cert_engine.actor.controller,'price_head'))
            with torch.no_grad():
                theta=model.policy(state)[1];default=model.cert_engine.default_theta(state)
                torch.testing.assert_close(theta.weights,default.weights,rtol=0,atol=1e-7)
                torch.testing.assert_close(theta.temperature,default.temperature,rtol=0,atol=1e-7)
                inst=MKPInstances(state.profits,state.weights.transpose(1,2),state.capacities)
                a=model(NewEnv(cfg.env),inst,decode='greedy');b=model(NewEnv(cfg.env),inst,decode='greedy',theta_override='fixed')
                self.assertTrue(torch.equal(a['solution'],b['solution']))

    def test_lp_certificate(self):
        for seed in range(15):
            env=fixture(seed)
            for step in range(3):
                state=env.state;lp=lp_relaxation(state,ModelConfig());mask=state.feasible[0].numpy()
                if not mask.any():break
                result=linprog(-state.profits[0,mask].numpy(),A_ub=state.weights[0,mask].numpy().T,
                    b_ub=state.remaining[0].numpy(),bounds=(0,1),method='highs')
                self.assertTrue(result.success)
                exact=-result.fun;tol=1e-6*max(1,exact)
                self.assertLessEqual(float(lp.primal_value[0]),exact+tol)
                self.assertLessEqual(exact,float(lp.dual_value[0])+tol)
                self.assertGreaterEqual(float(lp.certificate_gap[0]),0.)
                env.step(state.feasible[0].nonzero()[0].reshape(1))


if __name__=='__main__':unittest.main()
