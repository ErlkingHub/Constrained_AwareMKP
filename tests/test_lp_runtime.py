"""Execution acceleration must preserve LP tensors, RNG, actions and gradients."""
from dataclasses import fields
import unittest
import torch
from mkp_env import generate_instances
from mkp_v6_env import MKPEnv
from mkp_v6_config import ModelConfig
from mkp_v6_lp import solve_lp_relaxation
from mkp_t8_train import T8Trainer
from mkp_Trainer import set_seed
from scripts.profile_t9 import cert_config
import mkp_lp_runtime as runtime


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def tearDown(self):runtime.set_backend('eager');runtime.clear()

    def test_cpu_fallback_is_identical(self):
        inst=generate_instances(2,9,3,generator=torch.Generator().manual_seed(4))
        state=MKPEnv(inst.p,inst.r.transpose(1,2),inst.c).state
        runtime.set_backend('eager');reference=solve_lp_relaxation(state,ModelConfig())
        runtime.set_backend('cuda_graph');actual=solve_lp_relaxation(state,ModelConfig())
        for f in fields(reference):self.assertTrue(torch.equal(getattr(reference,f.name),getattr(actual,f.name)),f.name)
        self.assertEqual(runtime.statistics().get('captures',0),0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_lp_bitwise_and_owned_outputs(self):
        cfg=ModelConfig()
        for batch,n,m in ((1,17,5),(16,50,5),(64,100,10),(1,100,30)):
            inst=generate_instances(batch,n,m,generator=torch.Generator().manual_seed(300+m)).to('cuda:0')
            env=MKPEnv(inst.p,inst.r.transpose(1,2),inst.c);state=env.state
            runtime.set_backend('eager');reference=solve_lp_relaxation(state,cfg)
            runtime.set_backend('cuda_graph');before_rng=torch.cuda.get_rng_state();actual=solve_lp_relaxation(state,cfg)
            self.assertTrue(torch.equal(before_rng,torch.cuda.get_rng_state()))
            for f in fields(reference):self.assertTrue(torch.equal(getattr(reference,f.name),getattr(actual,f.name)),(batch,n,m,f.name))
            snapshot={f.name:getattr(actual,f.name).clone() for f in fields(actual)}
            env.step(state.feasible.long().argmax(-1));solve_lp_relaxation(state,cfg)
            for name,value in snapshot.items():self.assertTrue(torch.equal(getattr(actual,name),value),f'graph output overwritten: {name}')
            self.assertGreater(runtime.statistics().get('replays',0),0)
            self.assertEqual(runtime.statistics().get('fallback_calls',0),0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_full_update_actions_rng_gradients_and_parameters(self):
        outputs=[]
        for backend in ('eager','cuda_graph'):
            set_seed(617);cfg=cert_config();cfg.device.lp_backend=backend
            cfg.model.embed_dim=16;cfg.model.num_heads=2;cfg.model.encoder_layers=cfg.model.controller_layers=1;cfg.model.edge_hidden=8
            trainer=T8Trainer(cfg,torch.device('cuda:0'));trainer.step=1
            # Nonzero readout ensures the encoder, not just the zero head, gets gradients.
            with torch.no_grad():trainer.model.cert_engine.actor.controller.cert_head.weight.normal_(std=.03)
            inst=generate_instances(6,13,5,generator=torch.Generator().manual_seed(81)).to('cuda:0')
            sequences=[]
            handle=trainer.model.register_forward_hook(lambda model,args,out:sequences.append(out['sequence'].detach().cpu()))
            stats=trainer.update(inst);handle.remove()
            outputs.append(dict(sequences=sequences,rng=torch.cuda.get_rng_state(),stats=stats,
                weights={n:p.detach().clone() for n,p in trainer.model.named_parameters()},
                gradients={n:p.grad.clone() for n,p in trainer.model.named_parameters() if p.grad is not None}))
            if backend=='cuda_graph':self.assertGreater(runtime.statistics().get('replays',0),0)
        a,b=outputs
        self.assertTrue(torch.equal(a['rng'],b['rng']))
        self.assertEqual(len(a['sequences']),len(b['sequences']))
        for x,y in zip(a['sequences'],b['sequences']):self.assertTrue(torch.equal(x,y))
        for name in a['weights']:torch.testing.assert_close(a['weights'][name],b['weights'][name],rtol=1e-6,atol=1e-7)
        for name in a['gradients']:torch.testing.assert_close(a['gradients'][name],b['gradients'][name],rtol=1e-5,atol=1e-6)
        self.assertAlmostEqual(a['stats']['loss'],b['stats']['loss'],places=6)


if __name__=='__main__':unittest.main()
