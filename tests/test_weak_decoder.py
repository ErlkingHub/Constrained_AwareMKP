"""T1 interface contracts: transparent scores, resource theta, and feasible rollouts."""
import unittest

import torch

from mkp_config import Config
from mkp_env import MKPEnv, MKPInstances, generate_instances
from mkp_Model import ConstrainedMKPPolicy, WeakDecoder


def small_config():
    cfg = Config()
    cfg.model.embed_dim = 16
    cfg.model.num_heads = 2
    cfg.model.encoder_layers = 1
    cfg.model.controller_layers = 1
    cfg.model.edge_hidden = 8
    cfg.model.grad_checkpoint = False
    cfg.env.dual_iters = 8
    return cfg


def sample_instances(n=13, batch=2, dims=3):
    return generate_instances(batch, n, dims, tightness=(0.5,), weight_max=100,
                              profit_noise=37.7, generator=torch.Generator().manual_seed(512))


def anchor(inst):
    # Deliberately unequal prices exercise resource equivariance without LP solver noise.
    mu = torch.linspace(0.7, 1.3, inst.M).expand(inst.B, -1).clone()
    return mu, inst.p.sum(1)


def assert_feasible_maximal(test, inst, result):
    solution = result["solution"]
    load = torch.einsum("bmn,bn->bm", inst.r, solution)
    test.assertTrue(bool((load <= inst.c + 1e-5).all()))
    remaining = inst.c - load
    remaining_fit = (inst.r <= remaining[:, :, None] + 1e-5).all(1)
    test.assertFalse(bool((remaining_fit & (solution == 0)).any()))
    torch.testing.assert_close(result["profit"], (inst.p * solution).sum(1))


class EnvironmentRewardTests(unittest.TestCase):
    def test_hard_mask_reward_has_no_rejection_penalty(self):
        cfg = small_config()
        cfg.env.reject_penalty = 99.0
        env = MKPEnv(cfg.env)
        inst = sample_instances()
        st = env.reset(inst, *anchor(inst))
        st.profit = torch.tensor([9.0, 13.0])
        st.n_reject = torch.tensor([2.0, 7.0])
        torch.testing.assert_close(env.get_reward(st), st.profit / st.ub)

    def test_rejection_ablation_retains_its_penalty(self):
        cfg = small_config()
        cfg.env.feasibility_mode = "perceive"
        cfg.env.reject_penalty = 0.3
        env = MKPEnv(cfg.env)
        inst = sample_instances()
        st = env.reset(inst, *anchor(inst))
        st.profit = torch.tensor([9.0, 13.0])
        st.n_reject = torch.tensor([2.0, 7.0])
        torch.testing.assert_close(env.get_reward(st),
                                   st.profit / st.ub - 0.3 * st.n_reject / inst.N)


class WeakDecoderContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(103)
        self.cfg = small_config()
        self.env = MKPEnv(self.cfg.env)

    def policy(self, random_heads=False):
        self.cfg.model.untrained_random_heads = random_heads
        return ConstrainedMKPPolicy(self.cfg)

    def test_scores_equal_the_documented_formula(self):
        inst = MKPInstances(torch.tensor([[3.0, 7.0, 11.0, 4.0]]),
                            torch.tensor([[[1.0, 3.0, 9.0, 1.0],
                                           [2.0, 1.0, 1.0, 1.0]]]),
                            torch.tensor([[7.0, 5.0]]))
        st = self.env.reset(inst, torch.tensor([[2.0, 0.0]]), torch.tensor([30.0]))
        st.c_rem = torch.tensor([[4.0, 3.0]])
        st.status[:, 3] = 1
        self.env._refresh(st)
        theta = {"u": torch.tensor([[0.2, -0.2]]), "alpha": torch.tensor([1.4])}
        decoder = WeakDecoder(self.cfg.model, self.cfg.env)
        scores, lam = decoder(st, theta)
        expected_lam = (st.mu + self.cfg.model.price_floor * st.mu.mean(1, keepdim=True)
                        + 1e-6) * theta["u"].exp()
        cost = torch.einsum("bm,bmn->bn", expected_lam,
                            inst.r / (st.c_rem[:, :, None] + 1.0))
        expected = theta["alpha"][:, None] * (inst.p / inst.p.mean(1, keepdim=True)).log() - cost.log()
        torch.testing.assert_close(lam, expected_lam)
        torch.testing.assert_close(scores[st.fits], expected[st.fits])
        self.assertTrue(bool(torch.isneginf(scores[~st.fits]).all()))

    def test_no_item_learning_head_or_decoder_parameters(self):
        policy = self.policy()
        for name in ("res_head", "prim_head", "kappa_head"):
            self.assertFalse(hasattr(policy.controller, name))
        self.assertFalse(isinstance(policy.decoder, torch.nn.Module))
        self.assertFalse(any("decoder" in name for name, _ in policy.named_parameters()))
        self.assertEqual(policy.log_temperature.numel(), 1)

    def test_theta_dimension_independent_of_item_count(self):
        policy = self.policy(random_heads=True).eval()
        for n in (7, 19):
            inst = sample_instances(n=n)
            with torch.no_grad():
                out = policy(self.env, inst, decode="greedy", pre=anchor(inst), return_trace=True)
            for theta in out["trace"]:
                self.assertEqual(tuple(theta["u"].shape), (inst.B, inst.M))
                self.assertEqual(theta["alpha"].numel(), inst.B)
                self.assertEqual(theta["u"].numel() // inst.B + theta["alpha"].numel() // inst.B,
                                 inst.M + 1)
                torch.testing.assert_close(theta["u"].sum(1), torch.zeros(inst.B), atol=2e-6, rtol=0)
                self.assertTrue(bool(((theta["alpha"] >= 0.5) & (theta["alpha"] <= 2.5)).all()))

    def test_fixed_bypasses_network_and_is_repeatable(self):
        policy = self.policy(random_heads=True).eval()
        inst = sample_instances()
        calls = []
        hook = policy.controller.register_forward_hook(lambda *_: calls.append(1))
        try:
            with torch.no_grad():
                a = policy(self.env, inst, decode="greedy", pre=anchor(inst), theta_override="fixed")
                torch.manual_seed(809)
                b = policy(self.env, inst, decode="greedy", pre=anchor(inst), theta_override="fixed")
        finally:
            hook.remove()
        self.assertEqual(calls, [])
        torch.testing.assert_close(a["solution"], b["solution"])
        assert_feasible_maximal(self, inst, a)

    def test_warm_start_heads_match_fixed_and_untrained_heads_differ(self):
        inst = sample_instances()
        policy = self.policy().eval()
        with torch.no_grad():
            learned = policy(self.env, inst, decode="greedy", pre=anchor(inst), return_trace=True)
            fixed = policy(self.env, inst, decode="greedy", pre=anchor(inst), theta_override="fixed")
        torch.testing.assert_close(learned["solution"], fixed["solution"])
        for theta in learned["trace"]:
            torch.testing.assert_close(theta["u"], torch.zeros_like(theta["u"]))
            torch.testing.assert_close(theta["alpha"], torch.ones_like(theta["alpha"]))
        random_policy = self.policy(random_heads=True).eval()
        with torch.no_grad():
            random_out = random_policy(self.env, inst, decode="greedy", pre=anchor(inst), return_trace=True)
        self.assertTrue(any(not torch.allclose(theta["alpha"], torch.ones_like(theta["alpha"]))
                            for theta in random_out["trace"]))

    def test_item_and_resource_permutations_preserve_selected_sets(self):
        policy = self.policy(random_heads=True).eval()
        inst = sample_instances()
        pre = anchor(inst)
        ip = torch.randperm(inst.N)
        rp = torch.randperm(inst.M)
        permuted = MKPInstances(inst.p[:, ip], inst.r[:, rp][:, :, ip], inst.c[:, rp])
        with torch.no_grad():
            original = policy(self.env, inst, decode="greedy", pre=pre)
            changed = policy(self.env, permuted, decode="greedy", pre=(pre[0][:, rp], pre[1]))
        torch.testing.assert_close(original["solution"], changed["solution"][:, torch.argsort(ip)])

    def test_static_theta_uses_controller_once_and_remains_frozen(self):
        self.cfg.model.theta_update = "static"
        policy = self.policy(random_heads=True).eval()
        inst = sample_instances()
        calls = []
        hook = policy.controller.register_forward_hook(lambda *_: calls.append(1))
        try:
            with torch.no_grad():
                out = policy(self.env, inst, decode="greedy", pre=anchor(inst), return_trace=True)
        finally:
            hook.remove()
        self.assertEqual(len(calls), 1)
        self.assertGreater(len(out["trace"]), 1)
        first = out["trace"][0]
        for theta in out["trace"][1:]:
            torch.testing.assert_close(theta["u"], first["u"])
            torch.testing.assert_close(theta["alpha"], first["alpha"])

    def test_dynamic_theta_calls_controller_at_each_step(self):
        policy = self.policy(random_heads=True).eval()
        inst = sample_instances()
        calls = []
        hook = policy.controller.register_forward_hook(lambda *_: calls.append(1))
        try:
            with torch.no_grad():
                out = policy(self.env, inst, decode="greedy", pre=anchor(inst))
        finally:
            hook.remove()
        self.assertEqual(len(calls), out["steps"])
        self.assertGreater(len(calls), 1)

    def test_global_temperature_does_not_change_greedy_solution(self):
        policy = self.policy(random_heads=True).eval()
        inst = sample_instances()
        with torch.no_grad():
            policy.log_temperature.fill_(-5.0)
            cold = policy(self.env, inst, decode="greedy", pre=anchor(inst))
            policy.log_temperature.fill_(2.0)
            warm = policy(self.env, inst, decode="greedy", pre=anchor(inst))
        torch.testing.assert_close(cold["solution"], warm["solution"])

    def test_theta_override_rejects_centered_prices_outside_the_legal_family(self):
        policy = self.policy().eval()
        inst = sample_instances(dims=2)
        invalid = {"u": torch.tensor([[1000.0, -1000.0]]).expand(inst.B, -1),
                   "alpha": torch.ones(inst.B)}
        for decode in ("sample", "greedy"):
            with self.subTest(decode=decode), self.assertRaisesRegex(ValueError, "u_max|span|range"):
                policy(self.env, inst, decode=decode, pre=anchor(inst), theta_override=invalid)

    def test_sampled_rollouts_are_feasible_maximal_with_finite_gradients(self):
        for static in (False, True):
            with self.subTest(static=static):
                self.cfg.model.theta_update = "static" if static else "step"
                self.cfg.model.grad_checkpoint = True
                policy = self.policy().train()
                inst = sample_instances(n=9)
                result = policy(self.env, inst, decode="sample", num_samples=4, pre=anchor(inst))
                assert_feasible_maximal(self, inst.repeat(4), result)
                self.assertTrue(bool((result["n_reject"] == 0).all()))
                reward = result["reward"].view(inst.B, 4)
                advantage = (reward - reward.mean(1, keepdim=True)).detach().reshape(-1)
                loss = -(advantage * result["log_likelihood"]).mean() - 0.01 * result["entropy"].mean()
                self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward()
                gradients = [p.grad for p in policy.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
                self.assertTrue(any(bool((g.abs() > 0).any()) for g in gradients))

    def test_finished_rows_and_single_item_do_not_produce_nan(self):
        for n, dims in ((1, 1), (7, 3)):
            with self.subTest(n=n, dims=dims):
                inst = sample_instances(n=n, dims=dims)
                inst.c[0] = 0
                # Positive resource usage makes the first row finished at reset.
                inst.r[0] = inst.r[0].clamp_min(1)
                policy = self.policy().eval()
                with torch.no_grad():
                    out = policy(self.env, inst, decode="sample", pre=anchor(inst))
                for field in ("reward", "profit", "log_likelihood", "entropy"):
                    self.assertTrue(bool(torch.isfinite(out[field]).all()), field)
                self.assertEqual(float(out["solution"][0].sum()), 0)
                self.assertEqual(float(out["n_reject"].sum()), 0)
                assert_feasible_maximal(self, inst, out)


if __name__ == "__main__":
    unittest.main()
