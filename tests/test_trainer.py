"""Credit-assignment and exact-resume contracts, using tiny synthetic instances."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from mkp_config import ExperimentConfig, TrainConfig
from mkp_data import MKPInstance, solve_root_lp
from mkp_env import MKPState
from mkp_Trainer import MKPTrainer, compute_gae, normalized_reward
from mkp_utils import load_checkpoint


class CreditAssignmentTests(unittest.TestCase):
    def test_full_return_reaches_first_action_of_long_episode(self):
        rewards = np.zeros(250)
        rewards[-1] = 0.93
        values = np.linspace(-0.2, 0.6, len(rewards))
        advantages, targets = compute_gae(rewards, values, gamma=1.0, gae_lambda=1.0)
        np.testing.assert_allclose(targets, np.full(250, .93), atol=1e-12)
        self.assertAlmostEqual(advantages[0], .93 - values[0], places=12)

    def test_lambda_zero_is_one_step_td_and_terminal_bootstrap_is_zero(self):
        advantages, targets = compute_gae([.1, .2, .5], [.7, .4, .9], gamma=1.0, gae_lambda=0.0)
        np.testing.assert_allclose(advantages, [-.2, .7, -.4], atol=1e-12)
        np.testing.assert_allclose(targets, [.5, 1.1, .5], atol=1e-12)

    def test_incremental_mc_targets_are_remaining_profit(self):
        advantages, targets = compute_gae([.2, .1, .4], [.3, .2, .1], gamma=1.0, gae_lambda=1.0)
        np.testing.assert_allclose(targets, [.7, .5, .4], atol=1e-12)
        np.testing.assert_allclose(advantages, [.4, .3, .3], atol=1e-12)

    def test_reward_modes_equal_objective_and_shaping_telescopes(self):
        instance = MKPInstance("credit", np.array([6., 5., 4.]),
                               np.array([[3.], [2.], [2.]]), np.array([5.]))
        lp = solve_root_lp(instance)
        returns = {}
        for mode in ("terminal", "incremental"):
            for shaping in (0.0, .7):
                state = MKPState(instance, lp)
                initial_potential = state.potential()
                config = replace(TrainConfig(), reward_mode=mode, shaping_coef=shaping)
                rewards = []
                for action in (0, 1):
                    before = state.potential()
                    profit = state.step(action)
                    rewards.append(normalized_reward(profit, state.objective, state.done,
                                                     lp.upper_bound, before, state.potential(), config))
                self.assertTrue(state.done)
                self.assertEqual(state.potential(), 0.0)
                expected = state.objective / lp.upper_bound - shaping * initial_potential
                self.assertAlmostEqual(sum(rewards), expected, places=12)
                returns[mode, shaping] = sum(rewards)
        for shaping in (0.0, .7):
            self.assertAlmostEqual(returns["terminal", shaping], returns["incremental", shaping], places=12)

    def test_mismatched_trajectory_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            compute_gae([1., 2.], [0.])


class ConfigurationTests(unittest.TestCase):
    def test_unknown_config_keys_fail(self):
        for payload in ({"trainer": {}}, {"train": {"learining_rate": 1e-3}}):
            with self.assertRaises(ValueError):
                ExperimentConfig.from_dict(payload)

    def test_reject_invalid_or_nonfinite_research_settings(self):
        invalid = [
            {"model": {"price_log_bound": float("nan")}},
            {"model": {"temperature": float("inf")}},
            {"model": {"lp_input": "false"}},
            {"train": {"seed": 1.5}},
            {"train": {"seed": True}},
            {"train": {"actor_lr": float("inf")}},
            {"train": {"gamma": .99}},
            {"train": {"gae_lambda": 1.01}},
            {"eval": {"best_of_k": .5}},
            {"eval": {"filter_n": [-1]}},
            {"data": {"train_n": [True]}},
        ]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                ExperimentConfig.from_dict(payload)


class ResumeTests(unittest.TestCase):
    @staticmethod
    def tiny_config(directory, updates):
        config = ExperimentConfig()
        config.model.hidden_dim = 8
        config.model.message_rounds = 1
        config.model.pair_features = False
        config.data.train_n = (8,)
        config.data.train_m = (2,)
        config.data.capacity_ratios = (.5,)
        config.data.validation_instances = 2
        config.train.device = "cpu"
        config.train.updates = updates
        config.train.episodes_per_update = 2
        config.train.ppo_epochs = 1
        config.train.minibatch_size = 16
        config.train.validate_every = 1
        config.train.checkpoint_every = 1
        config.train.log_root = str(Path(directory) / "runs")
        config.train.checkpoint_root = str(Path(directory) / "checkpoints")
        return config.validate()

    @staticmethod
    def close_logger(trainer):
        for handler in list(trainer.logger.handlers):
            trainer.logger.removeHandler(handler)
            handler.close()

    def test_exact_resumed_cpu_update_matches_uninterrupted_training(self):
        trainers = []
        with tempfile.TemporaryDirectory() as directory:
            try:
                continuous = MKPTrainer(self.tiny_config(directory, updates=2))
                trainers.append(continuous)
                continuous.train()
                expected = load_checkpoint(continuous.checkpoint_dir / "last.pt")

                interrupted = MKPTrainer(self.tiny_config(directory, updates=1))
                trainers.append(interrupted)
                interrupted.train()
                resumed = MKPTrainer(self.tiny_config(directory, updates=2),
                                     resume=interrupted.checkpoint_dir / "last.pt")
                trainers.append(resumed)
                resumed.train()
                actual = load_checkpoint(resumed.checkpoint_dir / "last.pt")
                for component in ("actor", "critic", "initial_actor"):
                    self.assertEqual(expected[component].keys(), actual[component].keys())
                    for name in expected[component]:
                        self.assertTrue(torch.equal(expected[component][name], actual[component][name]),
                                        f"Resume drift in {component}.{name}")
                for name in ("update", "total_decisions", "optimizer_steps", "best_update", "best_score"):
                    self.assertEqual(expected[name], actual[name], name)
                self.assertTrue(torch.equal(expected["rng"]["torch"], actual["rng"]["torch"]))
                self.assertEqual(expected["rng"]["generator"], actual["rng"]["generator"])
                original_rows = [json.loads(line) for line in (continuous.run_dir / "episodes.jsonl").read_text().splitlines()]
                resumed_rows = [json.loads(line) for line in (resumed.run_dir / "episodes.jsonl").read_text().splitlines()]
                self.assertEqual([row for row in original_rows if row["update"] == 2], resumed_rows)
            finally:
                for trainer in trainers:
                    self.close_logger(trainer)


if __name__ == "__main__":
    unittest.main()
