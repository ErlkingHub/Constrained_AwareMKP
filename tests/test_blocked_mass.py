"""The single T2 preimplementation check requested by the T1-P guidance."""
import unittest

import torch

from mkp_basis import blocked_mass
from mkp_env import MKPEnv, MKPInstances, REJECTED, SELECTED


class BlockedMassTests(unittest.TestCase):
    def test_blocked_mass_matches_naive_double_loop(self):
        profits = torch.tensor([[2., 7., 5., 11., 3., 13., 17.],
                                [5., 2., 9., 4., 8., 7., 3.],
                                [2., 3., 5., 7., 11., 13., 17.]], requires_grad=True)
        resources = torch.tensor([[[2., 3., 6., 1., 4., 0., 7.],
                                   [4., 1., 2., 5., 1., 0., 1.]],
                                  [[1., 2., 3., 4., 5., 6., 1.],
                                   [5., 4., 3., 2., 1., 0., 2.]],
                                  [[1., 1., 2., 3., 4., 5., 6.],
                                   [2., 1., 1., 3., 4., 5., 6.]]])
        inst = MKPInstances(profits, resources, torch.tensor([[6., 5.], [7., 6.], [0., 0.]]))
        env = MKPEnv()
        st = env.reset(inst, torch.ones(3, 2), profits.detach().sum(1))
        st.status[0, 1] = SELECTED
        st.status[1, 3] = REJECTED
        st.c_rem[1] = torch.tensor([5., 4.])
        env._refresh(st)
        expected = torch.zeros_like(profits)
        for b in range(inst.B):
            denominator = float(profits[b, st.fits[b]].detach().sum())
            for i in range(inst.N):
                if not bool(st.fits[b, i]):
                    continue
                value = 0.0
                for k in range(inst.N):
                    if k == i or not bool(st.fits[b, k]):
                        continue
                    if bool((resources[b, :, k] > st.c_rem[b] - resources[b, :, i]).any()):
                        value += float(profits[b, k].detach())
                expected[b, i] = value / denominator if denominator > 0 else 0.0
        for chunk_size in (1, 3, 64):
            with self.subTest(chunk_size=chunk_size):
                actual = blocked_mass(st, chunk_size)
                torch.testing.assert_close(actual, expected)
                self.assertFalse(actual.requires_grad)


if __name__ == "__main__":
    unittest.main()
