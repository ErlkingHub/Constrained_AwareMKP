import copy
import unittest
from unittest.mock import patch

import torch

from mkp_config import Config
from mkp_env import MKPEnv, generate_instances
from mkp_evaluation import paired_report, paired_seed, parse_stages, summarize
from mkp_Model import build_model
from Benchmark_zeroShot import _load_arms, evaluate_stage, protocol_sha256
from legacy_t0 import intervention, legacy_modules


def row(arm="learned_dynamic", seed=1, instance=1, gap=.2, **kw):
    return {"id": instance, "M": 5, "N": 100, "alpha": .25, "arm": arm, "seed": seed,
            "stage": "greedy", "budget": 1, "search_starts": 0, "ils_rounds": 0,
            "ls_max_iters": 0, "ls_drop_k": 0, "ils_perturb_max": 0,
            "sample_temperature": 1.0, "evaluation_seed": 1234, "data_sha256": "data",
            "protocol_sha256": "g0g1",
            "gap_pct": gap, "objective": 100 - gap, "time_lp_s": .1,
            "time_construct_s": .2, "time_refine_s": 0., "time_total_s": .3, **kw}


class EvaluationTests(unittest.TestCase):
    def test_stage_budgets_are_explicit_and_no_hidden_greedy(self):
        parsed = parse_stages(["greedy", "sample64", "sampleK+ILS"], 256)
        self.assertEqual([r["budget"] for r in parsed], [1, 64, 256])
        self.assertEqual(parsed[-1]["name"], "sample256+ILS")
        with self.assertRaises(ValueError):
            parse_stages(["sample0"])
        with self.assertRaises(ValueError):
            parse_stages(["greedy", "greedy"])

    def test_pairing_excludes_mismatched_budget_seed_and_dataset(self):
        pairs, _ = paired_report([row(), row("fixed", gap=.4)])
        self.assertEqual(len(pairs), 1)
        self.assertAlmostEqual(pairs[0]["gap_delta_pp"], -.2)
        for changed in ({"seed": 2}, {"data_sha256": "different"}, {"protocol_sha256": "different"}):
            pairs, _ = paired_report([row(), row("fixed", gap=.4, **changed)])
            self.assertFalse(pairs)
        with self.assertRaisesRegex(ValueError, "Mismatched inference budgets"):
            paired_report([row(), row("fixed", gap=.4, budget=64)])
        with self.assertRaises(ValueError):
            paired_report([row(), row()])

    def test_cross_seed_inference_uses_instances_not_repeated_seed_rows(self):
        rows = []
        for seed in (1, 2, 3):
            for instance in (1, 2):
                rows.extend([row(seed=seed, instance=instance, evaluation_seed=seed),
                             row("fixed", seed, instance, .4, evaluation_seed=seed)])
        _, stats = paired_report(rows)
        combined = [r for r in stats if r["seed"] == "mean_over_seeds" and r["M"] == "all"
                    and r["N"] == "all" and r["alpha"] == "all"]
        self.assertEqual(len(combined), 1)
        self.assertEqual(combined[0]["n_pairs"], 2)
        self.assertEqual(combined[0]["training_seeds"], 3)
        self.assertTrue(combined[0]["meets_three_seed_requirement"])

    def test_summary_has_worst_p90_and_all_three_marginal_strata(self):
        stats = summarize([row(gap=.2), row(instance=2, gap=.6)])
        overall = next(r for r in stats if r["M"] == r["N"] == r["alpha"] == "all")
        self.assertAlmostEqual(overall["worst_gap_pct"], .6)
        self.assertAlmostEqual(overall["p90_gap_pct"], .56)
        self.assertEqual(len(stats), 6)

    def test_seed_independent_of_order_and_repeatable(self):
        self.assertEqual(paired_seed(1, 4, "sample64"), paired_seed(1, 4, "sample64"))
        self.assertNotEqual(paired_seed(1, 4, "sample64"), paired_seed(1, 5, "sample64"))

    def test_dynamic_checkpoint_rejected_for_static_arm(self):
        cfg = Config()
        cfg.model.theta_update = "step"
        ck = {"config": cfg.to_dict()}
        with patch("Benchmark_zeroShot.load_policy", return_value=(object(), cfg, ck)):
            with self.assertRaisesRegex(ValueError, "separately trained"):
                _load_arms(cfg, ["learned_static"], "unused.pt", None, torch.device("cpu"))

    def test_protocol_hash_ignores_static_but_tracks_executor_configuration(self):
        cfg = Config()
        reference = protocol_sha256(cfg)
        cfg.model.theta_update = "static"
        cfg.model.untrained_random_heads = True
        self.assertEqual(protocol_sha256(cfg), reference)
        cfg.model.price_floor += .01
        self.assertNotEqual(protocol_sha256(cfg), reference)

    def test_untrained_update_mode_does_not_depend_on_arm_order(self):
        cfg = Config()
        cfg.model.embed_dim, cfg.model.num_heads = 16, 2
        cfg.model.encoder_layers = cfg.model.controller_layers = 1
        static = copy.deepcopy(cfg)
        static.model.theta_update = "static"
        def loaded(path, *args, **kwargs):
            mc = static if path.endswith("static.pt") else cfg
            return object(), mc, {"config": mc.to_dict()}
        with patch("Benchmark_zeroShot.load_policy", side_effect=loaded), \
                patch("Benchmark_zeroShot.sha256_file", return_value="hash"):
            for arms in (["learned_static", "learned_dynamic", "untrained"],
                         ["learned_dynamic", "learned_static", "untrained"]):
                _, configs, _, _ = _load_arms(cfg, arms, "dynamic.pt", "static.pt", torch.device("cpu"))
                self.assertEqual(configs["untrained"].model.theta_update, "step")

    def test_fixed_stage_reproducibility_and_exact_sample_count(self):
        torch.set_num_threads(2)
        cfg = Config()
        cfg.model.embed_dim, cfg.model.num_heads = 16, 2
        cfg.model.encoder_layers = cfg.model.controller_layers = 1
        cfg.env.dual_iters = 10
        cfg.eval.max_rollouts_per_chunk = 2
        env = MKPEnv(cfg.env)
        inst = generate_instances(1, 10, 3, [.5], device="cpu", generator=torch.Generator().manual_seed(4))
        pre = env.precompute(inst)
        model = build_model(cfg).eval()
        captures = []
        a = evaluate_stage(model, env, inst, cfg.eval, torch.device("cpu"), False, pre,
                           "sample3", "fixed", 42, 9, artifact_sink=captures.append)
        b = evaluate_stage(model, env, inst, cfg.eval, torch.device("cpu"), False, pre,
                           "sample3", "fixed", 42, 9)
        self.assertEqual(a["solution"], b["solution"])
        self.assertEqual(a["objective"], b["objective"])
        self.assertEqual(sum(c["solutions"].shape[0] for c in captures), 3)
        self.assertTrue(a["feasible"] and a["maximal"])

    def test_t0_interventions_keep_c_kappa_and_separate_fully_fixed_arm(self):
        from types import SimpleNamespace
        cfg = SimpleNamespace(model=SimpleNamespace(primitives=["profit", "price_ratio"],
                              init_primitive="price_ratio", init_primitive_logit=2., kappa_init=3.))
        initial = {"residual": torch.ones(2, 5), "log_price": torch.ones(2, 3),
                   "prim_w": torch.full((2, 2), .5), "kappa": torch.full((2,), 86.)}
        a = intervention("a", cfg)(None, None, initial)
        b = intervention("b", cfg)(None, None, initial)
        c = intervention("c", cfg)(None, None, initial)
        full = intervention("c_fullfixed", cfg)(None, None, initial)
        self.assertTrue(torch.equal(a["residual"], initial["residual"]))
        self.assertTrue(torch.equal(b["residual"], torch.zeros(2, 5)))
        self.assertTrue(torch.equal(b["log_price"], initial["log_price"]))
        self.assertTrue(torch.equal(c["log_price"], torch.zeros(2, 3)))
        self.assertTrue(torch.equal(c["kappa"], initial["kappa"]))
        self.assertTrue(torch.equal(full["kappa"], torch.full((2,), 3.)))
        self.assertTrue(torch.equal(initial["residual"], torch.ones(2, 5)))

    def test_legacy_source_import_does_not_replace_current_modules(self):
        import mkp_env
        original = mkp_env.MKPEnv
        modules, commit, sources = legacy_modules("6a4cf82")
        self.assertIs(mkp_env.MKPEnv, original)
        self.assertIsNot(modules["mkp_env"].MKPEnv, original)
        self.assertEqual(len(commit), 40)
        self.assertEqual(len(sources["mkp_Model.py"]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
