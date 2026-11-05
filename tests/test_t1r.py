import copy, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import torch
from mkp_config import Config, config_from_dict
from mkp_Model import build_model, WeakDecoder
from mkp_env import MKPEnv, generate_instances, MKPInstances
from mkp_Trainer import MKPTrainer

class T1RTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def cfg(self):
        c=Config();c.model.embed_dim=16;c.model.num_heads=2
        c.model.encoder_layers=c.model.controller_layers=1;c.model.edge_hidden=8;c.env.dual_iters=5
        return c
    def test_legacy_config_is_explicitly_raw(self):
        d=Config().to_dict();d["model"].pop("score_normalization")
        self.assertFalse(config_from_dict(d).model.score_normalization)
        self.assertTrue(Config().model.score_normalization)
    def test_normalization_formula_and_zero_std_gradients(self):
        cfg=self.cfg();env=MKPEnv(cfg.env)
        inst=MKPInstances(torch.tensor([[4.,7.,1.],[1.,1.,1.]]),torch.tensor([[[1.,2.,9.]],[[8.,8.,8.]]]),torch.tensor([[3.],[0.]]))
        st=env.reset(inst,torch.ones(2,1),torch.tensor([20.,20.]))
        u=torch.zeros(2,1,requires_grad=True);alpha=torch.ones(2,requires_grad=True);dec=WeakDecoder(cfg.model,cfg.env)
        raw,_=dec.raw_scores(st,{"u":u,"alpha":alpha});z,_=dec(st,{"u":u,"alpha":alpha})
        expected=(raw[0,:2]-raw[0,:2].mean())/(raw[0,:2].std(unbiased=False)+1e-6)
        torch.testing.assert_close(z[0,:2],expected);self.assertTrue(bool((z[1]==0).all()))
        z[0,:2].square().sum().backward();self.assertTrue(bool(torch.isfinite(alpha.grad).all()))
        st.status[0,1]=1;env._refresh(st);z,_=dec(st,{"u":u,"alpha":alpha})
        self.assertEqual(float(z[0,0].detach()),0.)
    def test_raw_and_normalized_greedy_and_theta_gradient(self):
        cfg=self.cfg();cfg.model.untrained_random_heads=True
        env=MKPEnv(cfg.env);inst=generate_instances(3,13,3,generator=torch.Generator().manual_seed(31))
        model=build_model(cfg).eval();pre=env.precompute(inst)
        for override in (None,"fixed"):
            model.decoder.normalize=False;a=model(env,inst,decode="greedy",pre=pre,theta_override=override)
            model.decoder.normalize=True;b=model(env,inst,decode="greedy",pre=pre,theta_override=override)
            torch.testing.assert_close(a["solution"],b["solution"])
    def test_prepare_configs_does_not_train_and_requires_full_calibration(self):
        import json,sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
        import run_t1r
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cfg=Config();cfg.train.epochs=40
            cfg.train.runs_dir=str(root/"runs");cfg.save(str(root/"source.json"))
            meta={"beta0":8,"smoke_only":False,"selection":"synthetic fixed-theta best-of-16; ties larger beta","config":cfg.to_dict()}
            (root/"cal.json").write_text(json.dumps(meta));(root/"reg.json").write_text(json.dumps({"smoke_only":False,"mismatch_count":0}))
            argv=["run_t1r.py","--config",str(root/"source.json"),"--calibration",str(root/"cal.json"),"--regression",str(root/"reg.json"),"--output",str(root/"configs")]
            with patch.object(sys,"argv",argv),patch.object(run_t1r,"MKPTrainer") as trainer:
                run_t1r.main();trainer.assert_not_called()
            d=json.loads((root/"configs/step.json").read_text())
            self.assertEqual(d["train"]["epochs"],10);self.assertEqual(d["model"]["temperature_init"],.125)
            self.assertEqual(d["train"]["lr"],cfg.train.lr);self.assertTrue(d["train"]["gate_early_stop"])
            meta["smoke_only"]=True;(root/"cal.json").write_text(json.dumps(meta))
            with patch.object(sys,"argv",argv),self.assertRaises(ValueError):run_t1r.main()

    def test_headroom_search_budget_and_legal_witness(self):
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
        from theta_headroom import search
        cfg=self.cfg();env=MKPEnv(cfg.env);inst=generate_instances(1,8,2,generator=torch.Generator().manual_seed(12))
        model=build_model(cfg).eval()
        result=search(model,env,inst,env.precompute(inst),42,4,4)
        self.assertEqual(result["evaluations"],8)
        self.assertGreaterEqual(result["objective"],result["fixed_objective"])
        self.assertAlmostEqual(sum(result["u"]),0.,places=5)
        self.assertTrue(all(abs(u)<=3.00001 for u in result["u"]))

    def test_epoch_minus_one_selects_initialization_and_dynamic_gate_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg=self.cfg();cfg.train.epochs=10;cfg.train.steps_per_epoch=1
            cfg.train.num_samples=16;cfg.train.val_instances=1;cfg.train.val_sizes=[7];cfg.train.val_dims=[2]
            cfg.train.validation_reference="lp";cfg.train.runs_dir=temp+"/runs";cfg.train.ckpt_dir=temp+"/ckpts"
            cfg.train.run_name="gate";cfg.train.gate_early_stop=True
            trainer=MKPTrainer(cfg,torch.device("cpu"))
            with patch.object(trainer,"validate",side_effect=[{"gap_mean":1.}]+[{"gap_mean":1.31}]*4),patch.object(trainer,"train_step",return_value={}):
                trainer.fit()
            best=torch.load(trainer.best_checkpoint_path,weights_only=False)
            self.assertEqual(best["global_step"],0);self.assertEqual(best["best_val"],1.)
            self.assertEqual(trainer.epoch,4)
            self.assertTrue(Path(trainer.run_dir,"early_stop.json").exists())
            self.assertIn("-1,0",Path(trainer.run_dir,"val_log.csv").read_text())
if __name__=="__main__":unittest.main()
