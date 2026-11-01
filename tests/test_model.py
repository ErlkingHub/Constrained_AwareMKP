"""Contracts of the state/controller/executor, not training-effectiveness tests."""
import unittest
from dataclasses import replace

import numpy as np
import torch

from mkp_config import ModelConfig
from mkp_data import LPInfo, MKPInstance, generate_instance, solve_root_lp
from mkp_env import MKPState
from mkp_Model import (FixedExecutor, ResourceController, ResidualCritic,
                       _fixed_inputs, run_episode)


class ModelContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(3)
        self.config = ModelConfig(hidden_dim=16)
        self.rng = np.random.default_rng(4)
        self.instance = generate_instance(self.rng, 12, 3, "uncorrelated", 0.4, "contract")
        self.lp = solve_root_lp(self.instance)
        self.actor = ResourceController(self.config)

    def state(self, device="cpu"):
        return MKPState(self.instance, self.lp, device)

    def test_gradient_reaches_encoder_through_fixed_executor(self):
        state = self.state()
        theta = self.actor(state)
        executor = FixedExecutor(self.config)
        self.assertEqual(len(list(executor.parameters())), 0)
        self.assertEqual(theta.prices.shape, (self.instance.M,))
        self.assertEqual(theta.weights.shape, (4,))
        action = state.feasible.nonzero(as_tuple=True)[0][0]
        (-executor.distribution(state, theta).log_prob(action)).backward()
        gradients = [p.grad for p in self.actor.encoder.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertTrue(any(g.abs().sum() > 0 for g in gradients))

    def test_item_resource_equivariance(self):
        # Reorder the same LP prices: solving a degenerate LP anew may select a
        # different valid dual and is a separate solver-invariance question.
        item_order = self.rng.permutation(self.instance.N)
        resource_order = self.rng.permutation(self.instance.M)
        inst = self.instance
        permuted = MKPInstance("permuted", inst.p[item_order],
                               inst.w[item_order][:, resource_order], inst.c[resource_order])
        state = self.state()
        other = MKPState(permuted, LPInfo(self.lp.prices[resource_order], self.lp.upper_bound, 0.0))
        theta, other_theta = self.actor(state), self.actor(other)
        torch.testing.assert_close(theta.prices[resource_order], other_theta.prices, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(theta.weights, other_theta.weights, atol=2e-6, rtol=2e-6)
        executor = FixedExecutor(self.config)
        torch.testing.assert_close(executor.scores(state, theta)[item_order],
                                   executor.scores(other, other_theta), atol=2e-6, rtol=2e-6)

    def test_lp_input_is_independently_removable(self):
        config = replace(self.config, lp_input=False)
        model = ResourceController(config)
        changed = MKPState(self.instance, LPInfo(self.lp.prices + 5, self.lp.upper_bound * 2, 0))
        before = model.encoder(self.state())
        after = model.encoder(changed)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(before, after)))

    def test_disabled_criteria_have_no_mass(self):
        config = replace(self.config, lp_score=False, pair_features=False, pair_score=False)
        theta = ResourceController(config)(self.state())
        torch.testing.assert_close(theta.weights[2:], torch.zeros(2))
        self.assertAlmostEqual(float(theta.weights.detach().sum()), 1.0, places=6)

    def test_pair_input_ablation_leaves_executor_unchanged(self):
        config = replace(self.config, pair_features=False)
        state = self.state()
        theta = self.actor(state)
        torch.testing.assert_close(FixedExecutor(config).scores(state, theta),
                                   FixedExecutor(self.config).scores(state, theta))

    def test_all_modes_are_feasible_maximal_and_static_is_frozen(self):
        for mode in ("dynamic", "static", "fixed", "lp_dual"):
            result = run_episode(self.actor, self.state(), mode=mode)
            indices = result["selected"]
            self.assertTrue(result["feasible"])
            self.assertTrue(result["maximal"])
            self.assertTrue(np.all(self.instance.w[indices].sum(0) <= self.instance.c))
            self.assertEqual(result["objective"], float(self.instance.p[indices].sum()))
            if mode != "dynamic":
                self.assertEqual(result["theta_change_mean"], 0.0)

    def test_zero_capacity_zero_profit_and_single_resource(self):
        cases = [
            MKPInstance("zero_capacity", np.array([0., 2., 1.]),
                        np.array([[0., 0.], [0., 1.], [1., 0.]]), np.array([0., 1.])),
            MKPInstance("zero_profit", np.zeros(4), np.ones((4, 1)), np.array([2.])),
            MKPInstance("terminal", np.ones(2), np.ones((2, 1)), np.array([0.])),
        ]
        for instance in cases:
            state = MKPState(instance, solve_root_lp(instance))
            result = run_episode(self.actor, state)
            self.assertTrue(result["feasible"] and result["maximal"])
            self.assertTrue(np.isfinite(result["objective"]))
            self.assertEqual(state.potential(), 0.0)
        self.assertEqual(run_episode(self.actor, MKPState(cases[0], solve_root_lp(cases[0])))["objective"], 2.)

    def test_tied_items_have_equal_probability_greedy_uses_first_index(self):
        # A deterministic argmax cannot choose an equivariant unique item from
        # perfectly identical candidates. Probability equivariance still holds.
        instance = MKPInstance("ties", np.ones(3), np.ones((3, 2)), np.ones(2))
        state = MKPState(instance, solve_root_lp(instance))
        dist = FixedExecutor(self.config).distribution(state, self.actor(state))
        torch.testing.assert_close(dist.probs, torch.full((3,), 1 / 3))
        self.assertEqual(run_episode(self.actor, state)["selected"], [0])

    def test_snapshot_and_potential_telescoping(self):
        state = self.state()
        self.actor(state)  # Populate cached descriptors before snapshot.
        snapshot = state.clone()
        before_remaining = snapshot.remaining.clone()
        initial_potential = state.potential()
        differences = 0.0
        while not state.done:
            before = state.potential()
            state.step(int(state.feasible.nonzero(as_tuple=True)[0][0]))
            differences += state.potential() - before
        self.assertAlmostEqual(differences + initial_potential, 0.0, places=12)
        self.assertEqual(snapshot.selected, [])
        self.assertTrue(torch.equal(snapshot.remaining, before_remaining))
        with self.assertRaises(ValueError):
            state.step(state.selected[0])

    def test_pair_descriptors_match_enumeration(self):
        state = self.state()
        blocked, partner = state.pair_descriptors(chunk_size=2)
        feasible = np.flatnonzero(state.feasible.cpu().numpy())
        total = self.instance.p[feasible].sum()
        for item in feasible:
            incompatible, compatible = [], []
            for other in feasible:
                if item == other:
                    continue
                if np.all(self.instance.w[item] + self.instance.w[other] <= self.instance.c):
                    compatible.append(other)
                else:
                    incompatible.append(other)
            self.assertAlmostEqual(float(blocked[item]), self.instance.p[incompatible].sum() / total)
            expected = max(self.instance.p[compatible], default=0) / self.instance.p.max()
            self.assertAlmostEqual(float(partner[item]), expected)

    def test_independent_critic_and_device_dtype(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            actor = ResourceController(self.config).to(device)
            critic = ResidualCritic(self.config).to(device)
            self.assertFalse(set(map(id, actor.parameters())) & set(map(id, critic.parameters())))
            state = self.state(device)
            self.assertEqual(state.remaining.dtype, torch.float64)
            self.assertTrue(all(x.dtype == torch.float32 for x in _fixed_inputs(state, self.config)))
            theta, value = actor(state), critic(state)
            self.assertEqual(theta.prices.device.type, device)
            self.assertEqual(value.device.type, device)
            (-FixedExecutor(self.config).distribution(state, theta).entropy() + value.square()).backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters()))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in critic.parameters()))


if __name__ == "__main__":
    unittest.main()
