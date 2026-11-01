"""Metric/audit invariants: evaluation never turns a reference label into input."""
import unittest

import numpy as np

from Benchmark_zeroShot import _instances, audit_solution, paired_comparisons, paired_seed, signed_gap, summarize
from mkp_config import ExperimentConfig
from mkp_data import MKPInstance


class BenchmarkMetricsTest(unittest.TestCase):
    def test_synthetic_test_instances_ignore_training_parameters_and_other_grid_cells(self):
        config = ExperimentConfig()
        config.eval.synthetic = True
        config.eval.synthetic_n = (8,)
        config.eval.synthetic_m = (2,)
        config.eval.synthetic_distributions = ("weakly_correlated",)
        config.eval.synthetic_ratios = (0.5,)
        config.eval.synthetic_instances_per_group = 2
        trained = ExperimentConfig()
        original, _, original_sha, _ = _instances(config, trained)
        trained.data.coefficient_max = 7
        trained.data.correlation_noise = 0
        changed, _, changed_sha, _ = _instances(config, trained)
        self.assertEqual(original_sha, changed_sha)
        self.assertEqual([x.fingerprint() for x in original], [x.fingerprint() for x in changed])
        config.eval.synthetic_n = (5, 8)
        expanded, _, _, _ = _instances(config, trained)
        expanded_by_id = {x.id: x.fingerprint() for x in expanded}
        for instance in original:
            self.assertEqual(instance.fingerprint(), expanded_by_id[instance.id])

    def test_signed_gap_and_missing_reference(self):
        self.assertAlmostEqual(signed_gap(110, 100), -10.0)
        self.assertIsNone(signed_gap(10, None))
        self.assertIsNone(signed_gap(0, 0))

    def test_raw_solution_audit_rejects_duplicates_and_infeasibility(self):
        instance = MKPInstance("tiny", np.array([5, 4]), np.array([[2, 1], [2, 2]]), np.array([2, 2]))
        result = audit_solution(instance, {"selected": [0], "objective": 5})
        self.assertTrue(result["feasible"])
        self.assertTrue(result["maximal"])
        with self.assertRaises(RuntimeError):
            audit_solution(instance, {"selected": [0, 0], "objective": 10})
        with self.assertRaises(RuntimeError):
            audit_solution(instance, {"selected": [0, 1], "objective": 9})
        with self.assertRaises(RuntimeError):
            audit_solution(instance, {"selected": [], "objective": 0})

    def test_pairing_seed_is_independent_of_arm_iteration_order(self):
        self.assertEqual(paired_seed(1, "a", "sample_best_of_k", 2), paired_seed(1, "a", "sample_best_of_k", 2))
        self.assertNotEqual(paired_seed(1, "a", "sample_best_of_k", 2), paired_seed(1, "b", "sample_best_of_k", 2))
        self.assertNotEqual(paired_seed(1, "a", "sample_best_of_k", 2), paired_seed(1, "a", "sample_best_of_k", 3))

    def test_summary_keeps_stages_and_signed_paired_gap_separate(self):
        common = dict(instance_id="a", n=2, m=1, capacity_ratio_group=0.5,
                      distribution="test", seen_n=False, seen_m=True, seen_distribution=False,
                      feasible=True, maximal=True, lp_gap_pct=5.0, steps=1,
                      utilization_mean=0.9, lp_seconds=0.1, policy_seconds=0.2,
                      total_seconds=0.3, policy_seconds_per_decision=0.2,
                      theta_change_mean=0.0)
        rows = [dict(common, arm="dynamic", stage="greedy", budget=1, objective=101., best_known_gap_pct=-1.),
                dict(common, arm="fixed", stage="greedy", budget=1, objective=99., best_known_gap_pct=1.),
                dict(common, arm="dynamic", stage="sample_best_of_k", budget=4, objective=100., best_known_gap_pct=0.)]
        overall = [row for row in summarize(rows) if row["grouping"] == "overall"]
        self.assertEqual(len(overall), 3)
        self.assertEqual(overall[0]["best_known_gap_pct_mean"], -1.)
        paired = paired_comparisons(rows)
        self.assertEqual(len(paired), 1)
        self.assertEqual(paired[0]["best_known_gap_delta_pp"], -2.)
        self.assertEqual(paired[0]["objective_gain_dynamic_minus_control"], 2.)


if __name__ == "__main__":
    unittest.main()
