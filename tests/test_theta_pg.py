"""The three new T1-P checks; legacy-path coverage reuses the existing suite."""
import math
import unittest

import torch
from mkp_config import Config
from mkp_env import MKPEnv, generate_instances
from mkp_Model import build_model, sample_theta_pre, theta_pre_log_prob


def config():
    c=Config();c.model.embed_dim=16;c.model.num_heads=2
    c.model.encoder_layers=c.model.controller_layers=1;c.model.edge_hidden=8
    c.model.grad_checkpoint=False;c.env.dual_iters=5
    return c


class ThetaPGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_score_function_gradient_matches_analytic_and_finite_difference(self):
        means={"u_pre":torch.tensor([[.1,-.2]],dtype=torch.float64,requires_grad=True),
               "alpha_pre":torch.tensor([-.7],dtype=torch.float64,requires_grad=True)}
        sigmas={"u_pre":.05,"alpha_pre":.3}
        draws,logp=sample_theta_pre(means,sigmas,1,1,generator=torch.Generator().manual_seed(13))
        gradients=torch.autograd.grad(logp.sum(),tuple(means.values()))
        for (key,mean),grad in zip(means.items(),gradients):
            expected=(draws[key]-mean.detach())/sigmas[key]**2
            torch.testing.assert_close(grad,expected,rtol=1e-10,atol=1e-10)
            numerical=torch.zeros_like(mean)
            for j in range(mean.numel()):
                plus={k:v.detach().clone() for k,v in means.items()}
                minus={k:v.detach().clone() for k,v in means.items()}
                plus[key].reshape(-1)[j]+=1e-6;minus[key].reshape(-1)[j]-=1e-6
                numerical.reshape(-1)[j]=(theta_pre_log_prob(plus,draws,sigmas).sum()-theta_pre_log_prob(minus,draws,sigmas).sum())/2e-6
            torch.testing.assert_close(grad,numerical,rtol=1e-7,atol=1e-7)
            self.assertFalse(draws[key].requires_grad)

    def test_antithetic_first_decision_and_static_controller_called_on_B_once(self):
        cfg=config();cfg.model.theta_update="static"
        env=MKPEnv(cfg.env);model=build_model(cfg).eval()
        inst=generate_instances(2,15,3,[.5],generator=torch.Generator().manual_seed(7));pre=env.precompute(inst)
        calls=[];hook=model.controller.register_forward_hook(lambda module,args,out:calls.append(args[0].shape[0]))
        try:
            out=model(env,inst,num_samples=4,decode="sample",pre=pre,return_trace=True,
                      theta_noise={"sigma_u":.05,"sigma_alpha":.3,"antithetic":True,"generator":torch.Generator().manual_seed(91)})
        finally:hook.remove()
        self.assertEqual(calls,[inst.B]);self.assertEqual(tuple(out["theta_logp"].shape),(8,))
        first=out["trace"][0]
        for key in ("u_pre","alpha_pre"):
            diff=first["pre_sample"][key]-first["pre_mean"][key]
            diff=diff.reshape(inst.B,2,2,-1)
            torch.testing.assert_close(diff[:,:,0],-diff[:,:,1],atol=1e-7,rtol=1e-5)
        self.assertEqual(sum(t["theta_decision"] for t in out["trace"]),1)
        # Noise chooses theta; item actions remain exact argmax even if decode='sample'.
        st=env.reset(inst.repeat(4),pre[0].repeat_interleave(4,0),pre[1].repeat_interleave(4,0))
        for step in out["trace"]:
            score,_=model.decoder(st,{"u":step["u"],"alpha":step["alpha"]})
            torch.testing.assert_close(score.argmax(-1),step["action"])
            st=env.step(st,step["action"])

    def test_interval_holds_theta_and_k1_reproduces_legacy_step(self):
        cfg=config();cfg.model.untrained_random_heads=True;cfg.model.theta_interval=3
        env=MKPEnv(cfg.env);torch.manual_seed(61);model=build_model(cfg).eval()
        inst=generate_instances(2,17,3,[.75],generator=torch.Generator().manual_seed(9));pre=env.precompute(inst)
        calls=[];hook=model.controller.register_forward_hook(lambda *args:calls.append(1))
        try:
            with torch.no_grad():out=model(env,inst,decode="greedy",pre=pre,return_trace=True)
        finally:hook.remove()
        self.assertEqual(len(calls),math.ceil(out["steps"]/3))
        for j,step in enumerate(out["trace"]):
            self.assertEqual(step["theta_decision"],j%3==0)
            if j%3:
                torch.testing.assert_close(step["u"],out["trace"][j-1]["u"])
                torch.testing.assert_close(step["alpha"],out["trace"][j-1]["alpha"])
        # Same serialized parameters against the immutable pre-T1-P implementation.
        from legacy_t0 import legacy_modules
        modules,_,_=legacy_modules("5b390e4")
        old_cfg=modules["mkp_config"].config_from_dict(cfg.to_dict())
        old=modules["mkp_Model"].build_model(old_cfg).eval();old.load_state_dict(model.state_dict())
        cfg.model.theta_interval=1
        with torch.no_grad():
            before=old(env,inst,decode="greedy",pre=pre,return_trace=True)
            after=model(env,inst,decode="greedy",pre=pre,return_trace=True)
        for key in ("solution","profit","log_likelihood"):
            torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
        for left,right in zip(before["trace"],after["trace"]):
            torch.testing.assert_close(left["u"],right["u"],rtol=0,atol=0)
            torch.testing.assert_close(left["alpha"],right["alpha"],rtol=0,atol=0)

if __name__=="__main__":unittest.main()
