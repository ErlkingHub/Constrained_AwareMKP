import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mkp_data import DISTRIBUTIONS, MKPInstance, generate_instance, load_chu, solve_root_lp
from mkp_utils import PROJECT_ROOT, atomic_torch_save, load_checkpoint, write_json


class DataTests(unittest.TestCase):
    def test_canonical_shape_rejects_transpose(self):
        with self.assertRaises(ValueError):
            MKPInstance("bad", np.ones(3), np.ones((2, 3)), np.ones(2))

    def test_nonfinite_and_negative_rejected(self):
        for bad in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                MKPInstance("bad", np.array([bad]), np.ones((1, 1)), np.ones(1))

    def test_synthetic_reproducibility_and_no_labels(self):
        for family in DISTRIBUTIONS:
            first = generate_instance(np.random.default_rng(12), 12, 3, family, .25, "a")
            second = generate_instance(np.random.default_rng(12), 12, 3, family, .25, "a")
            self.assertEqual(first.fingerprint(), second.fingerprint())
            self.assertIsNone(first.best_known)
            self.assertEqual(first.w.shape, (12, 3))
            self.assertTrue(np.array_equal(first.c, np.floor(.25 * first.w.sum(axis=0))))

    def test_lp_dual_bounds_exhaustive_optimum(self):
        instance = MKPInstance("tiny", np.array([8, 7, 6, 2.]),
                               np.array([[4, 3], [2, 4], [3, 2], [0, 0.]]), np.array([5, 6.]))
        info = solve_root_lp(instance)
        feasible = [float(instance.p @ x) for x in map(np.array, itertools.product((0, 1), repeat=4))
                    if np.all(x @ instance.w <= instance.c)]
        self.assertGreaterEqual(info.upper_bound, max(feasible))
        self.assertTrue(np.all(info.prices >= 0))
        self.assertLessEqual(info.upper_bound, instance.p.sum() + 1e-7)

    def test_lp_zero_capacity_and_free_items(self):
        instance = MKPInstance("zero", np.array([4., 9.]), np.array([[0.], [1.]]), np.array([0.]))
        self.assertAlmostEqual(solve_root_lp(instance).upper_bound, 4.0, places=7)
        empty_value = MKPInstance("allzero", np.zeros(2), np.ones((2, 1)), np.zeros(1))
        self.assertEqual(solve_root_lp(empty_value).upper_bound, 0.0)

    def test_chu_reader_full_dataset(self):
        data = load_chu(PROJECT_ROOT / "MKP_dataset" / "Benchmarks_Chu.mat")
        self.assertGreater(len(data), 0)
        self.assertEqual(len(data), len({x.id for x in data}))
        self.assertEqual({x.N for x in data}, {100, 250, 500})
        for instance in data:
            self.assertEqual(instance.w.shape, (instance.N, instance.M))
            self.assertGreater(instance.best_known, 0)

    def test_missing_benchmark_fails(self):
        with self.assertRaises(FileNotFoundError):
            load_chu(PROJECT_ROOT / "missing_benchmark.mat")

    def test_checkpoint_roundtrip_and_schema_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            atomic_torch_save({"schema_version": 1, "update": 2, "actor": {}, "config": {}}, path)
            self.assertEqual(load_checkpoint(path)["update"], 2)
            atomic_torch_save({"schema_version": 999}, path)
            with self.assertRaises(ValueError):
                load_checkpoint(path)
            with self.assertRaises(ValueError):
                write_json(Path(directory) / "invalid.json", {"metric": float("nan")})


if __name__ == "__main__":
    unittest.main()
