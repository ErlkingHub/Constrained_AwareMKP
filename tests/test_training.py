"""Validation-reference, checkpoint compatibility and training regression checks."""
import copy
import itertools
import os
import tempfile
import unittest

import numpy as np
import torch

from mkp_artifacts import instance_sha, sha256_file
from mkp_config import Config
from mkp_env import MKPInstances
from mkp_reference import milp_reference
from mkp_Trainer import CHECKPOINT_SCHEMA, MKPTrainer, _CSVLogger, _check_checkpoint, load_policy, set_seed


class ReferenceTests(unittest.TestCase):
    def test_integer_reference_matches_enumeration_and_cache(self):
        inst = MKPInstances(torch.tensor([[7., 6., 4., 3.]]),
                            torch.tensor([[[5., 3., 2., 1.], [1., 4., 2., 3.]]]),
                            torch.tensor([[6., 6.]]))
        feasible = []
        for bits in itertools.product((0., 1.), repeat=inst.N):
            x = torch.tensor(bits)
            if bool((inst.r[0] @ x <= inst.c[0]).all()):
                feasible.append(float(inst.p[0] @ x))
        with tempfile.TemporaryDirectory() as directory:
            result = milp_reference(inst, time_limit=2., cache_dir=directory, log=lambda *_: None)
            cached = milp_reference(inst, time_limit=2., cache_dir=directory, log=lambda *_: None)
            self.assertEqual(float(result["obj"][0]), max(feasible))
            self.assertTrue(result["optimal"][0])
            self.assertGreaterEqual(result["bound"][0] + 1e-6, result["obj"][0])
            self.assertTrue(np.array_equal(result["solution"], cached["solution"]))
            self.assertEqual(result["sha256"], cached["sha256"])
            self.assertEqual(result["settings"]["data_sha256"], instance_sha(inst))

    def test_legacy_checkpoint_fails_before_config_fields_are_discarded(self):
        with self.assertRaisesRegex(ValueError, "Legacy delta"):
            _check_checkpoint({"model": {}, "config": {"model": {"residual_max": 3.}}})
        with self.assertRaisesRegex(ValueError, "Legacy delta"):
            _check_checkpoint({"model": {"controller.res_head.weight": torch.zeros(1)}, "config": {}})
        with self.assertRaisesRegex(ValueError, "schema"):
            _check_checkpoint({"model": {}, "config": {}})

    def test_corrupt_reference_cache_is_rejected(self):
        inst = MKPInstances(torch.tensor([[5., 4., 2.]]), torch.tensor([[[3., 2., 1.]]]),
                            torch.tensor([[3.]]))
        with tempfile.TemporaryDirectory() as directory:
            original = milp_reference(inst, time_limit=2., cache_dir=directory, log=lambda *_: None)
            with np.load(original["path"], allow_pickle=False) as cache:
                valid = {k: cache[k].copy() for k in cache.files}
            cases = {
                "settings": lambda data: data.update(settings="{}"),
                "nonbinary": lambda data: data["solution"].__setitem__((0, 0), 0.5),
                "infeasible": lambda data: data.update(solution=np.ones((1, 3), dtype=np.int8)),
                "objective": lambda data: data.update(obj=data["obj"] + 1.),
                "bound": lambda data: data.update(bound=data["obj"] - 1.),
                "status": lambda data: data.update(status=np.array([1])),
                "shape": lambda data: data.update(optimal=np.ones((1, 1), dtype=bool)),
            }
            for name, corrupt in cases.items():
                with self.subTest(name=name):
                    data = {k: value.copy() for k, value in valid.items()}
                    if name == "nonbinary":
                        data["solution"] = data["solution"].astype(float)
                    corrupt(data)
                    np.savez_compressed(original["path"], **data)
                    with self.assertRaises(RuntimeError):
                        milp_reference(inst, time_limit=2., cache_dir=directory, log=lambda *_: None)

    def test_csv_append_preserves_one_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "rows.csv")
            _CSVLogger(path).log({"epoch": 0, "loss": 1.})
            _CSVLogger(path).log({"epoch": 1, "loss": .5})
            with open(path, encoding="utf-8") as stream:
                self.assertEqual(stream.read().count("epoch,loss"), 1)
            with self.assertRaisesRegex(ValueError, "schema changed"):
                _CSVLogger(path).log({"epoch": 2, "new_loss": .2})

    def test_dynamic_checkpoint_cannot_resume_static_training(self):
        cfg = Config()
        ck = {"schema": CHECKPOINT_SCHEMA, "model": {}, "config": cfg.to_dict()}
        cfg.model.theta_update = "static"
        with self.assertRaisesRegex(ValueError, "theta_update"):
            _check_checkpoint(ck, expected_model=cfg.model)


class TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_update_validation_and_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = Config()
            cfg.device.amp = False
            cfg.env.train_sizes, cfg.env.train_dims = [8], [2]
            cfg.env.dual_iters = 5
            cfg.model.embed_dim, cfg.model.num_heads = 16, 2
            cfg.model.encoder_layers, cfg.model.controller_layers = 1, 1
            cfg.model.edge_hidden = 8
            cfg.train.batch_size, cfg.train.num_samples = 2, 2
            cfg.train.epochs, cfg.train.steps_per_epoch = 1, 1
            cfg.train.val_instances, cfg.train.val_sizes, cfg.train.val_dims = 2, [8], [2]
            cfg.train.val_milp_time_limit = 2.
            cfg.train.runs_dir = os.path.join(directory, "runs")
            cfg.train.ckpt_dir = os.path.join(directory, "checkpoints")
            cfg.train.reference_cache = os.path.join(directory, "cache")
            cfg.train.run_name = "roundtrip"
            set_seed(2026)
            trainer = MKPTrainer(cfg, torch.device("cpu"))
            stats = trainer.train_step()
            for name in ("loss", "reward", "grad_norm", "alpha_mean", "u_abs_mean", "temperature"):
                self.assertTrue(np.isfinite(stats[name]), name)
            self.assertEqual(stats["finite_gradients"], 1)
            val = trainer.validate()
            self.assertTrue(np.isfinite(val["gap_mean"]))
            trainer.best_val = val["gap_mean"]
            trainer.save(is_best=True)
            path = os.path.join(trainer.ckpt_dir, "best.pt")
            loaded, loaded_cfg, ck = load_policy(path, torch.device("cpu"))
            self.assertEqual(loaded_cfg.model.theta_update, cfg.model.theta_update)
            self.assertIn("benchmark data excluded", ck["selection_rule"])
            self.assertEqual(len(ck["validation_manifest"]), 1)
            self.assertTrue(os.path.isfile(path + ".meta.json"))
            for expected, actual in zip(trainer.model.parameters(), loaded.parameters()):
                self.assertTrue(torch.equal(expected, actual))
            bad = copy.deepcopy(cfg)
            bad.train.resume = path
            bad.train.val_seed += 1
            bad.train.run_name = "bad-resume"
            with self.assertRaisesRegex(ValueError, "changing validation"):
                MKPTrainer(bad, torch.device("cpu"))
            for name, value in (("dual_iters", 6), ("feasibility_mode", "perceive"),
                                ("reject_penalty", .5)):
                bad = copy.deepcopy(cfg)
                bad.train.resume = path
                bad.train.run_name = "bad-resume-" + name
                setattr(bad.env, name, value)
                with self.subTest(env_field=name), self.assertRaisesRegex(ValueError, "changing validation"):
                    MKPTrainer(bad, torch.device("cpu"))
            resumed_cfg = copy.deepcopy(cfg)
            resumed_cfg.train.resume = os.path.join(trainer.ckpt_dir, "last.pt")
            resumed_cfg.train.run_name = "resumed"
            resumed = MKPTrainer(resumed_cfg, torch.device("cpu"))
            self.assertTrue(resumed.inherited_best["available"])
            self.assertEqual(resumed.inherited_best["source_path"], path)
            self.assertEqual(sha256_file(resumed.best_checkpoint_path), sha256_file(path))
            isolated = os.path.join(directory, "isolated")
            os.makedirs(isolated)
            isolated_path = os.path.join(isolated, "last.pt")
            with open(resumed_cfg.train.resume, "rb") as stream, open(isolated_path, "wb") as target:
                target.write(stream.read())
            missing_cfg = copy.deepcopy(cfg)
            missing_cfg.train.resume, missing_cfg.train.run_name = isolated_path, "missing-best"
            missing = MKPTrainer(missing_cfg, torch.device("cpu"))
            self.assertFalse(missing.inherited_best["available"])
            self.assertIsNone(missing.best_checkpoint_path)
            if trainer.tb is not None:
                trainer.tb.close()
            for resumed_trainer in (resumed, missing):
                if resumed_trainer.tb is not None:
                    resumed_trainer.tb.close()


if __name__ == "__main__":
    unittest.main()
