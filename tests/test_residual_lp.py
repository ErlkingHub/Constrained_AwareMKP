"""The requested small-state HiGHS residual-LP correctness comparison."""
import unittest

import torch
from scipy.optimize import linprog

from mkp_config import Config
from mkp_env import MKPEnv, generate_instances, lp_dual


class ResidualLPTests(unittest.TestCase):
    def test_residual_upper_bounds_match_highs_and_cold_start(self):
        torch.set_num_threads(1)
        cfg = Config()
        env = MKPEnv(cfg.env)
        observed = []
        for seed in (41, 73, 1234):
            rng = torch.Generator().manual_seed(seed)
            inst = generate_instances(3, 30, 5, (.25, .5, .75), generator=rng)
            st = env.reset(inst, *env.precompute(inst))
            root_mu, root_x = st.mu.clone(), st.x_lp_root.clone()
            for step in range(1, 12):
                weights = st.fits.float()
                weights[st.done, 0] = 1.0
                action = torch.multinomial(weights, 1, generator=rng).squeeze(1)
                st = env.step(st, action)
                if step not in (3, 7, 11):
                    continue
                env.refresh_lp(st)
                _, cold_ub = lp_dual(inst.p, inst.r, st.c_rem.clamp_min(1),
                                     iters=300, lr=cfg.env.dual_lr,
                                     item_mask=st.fits, scale=st.lp_scale)
                torch.testing.assert_close(st.mu, root_mu, rtol=0, atol=0)
                torch.testing.assert_close(st.x_lp_root, root_x, rtol=0, atol=0)
                self.assertFalse(st.x_lp.requires_grad)
                self.assertTrue(bool((st.x_lp[~st.fits] == 0).all()))
                for b in range(inst.B):
                    mask = st.fits[b].cpu().numpy()
                    if mask.any():
                        solved = linprog(-inst.p[b].double().numpy()[mask],
                                         A_ub=inst.r[b].double().numpy()[:, mask],
                                         b_ub=st.c_rem[b].double().numpy(),
                                         bounds=(0, 1), method="highs")
                        self.assertTrue(solved.success, solved.message)
                        reference = -float(solved.fun)
                    else:
                        reference = 0.0
                    ub, cold = float(st.ub_res[b]), float(cold_ub[b])
                    scale = max(reference, 1.0)
                    gap = (ub - reference) / scale
                    warm_cold = abs(ub - cold) / max(cold, 1.0)
                    observed.append((seed, step, b, gap, warm_cold))
                    self.assertGreaterEqual(ub, reference - 1e-6 * scale,
                                            (seed, step, b, ub, reference))
        worst_gap = max(row[3] for row in observed)
        worst_cold_difference = max(row[4] for row in observed)
        print(f"Residual LP: {len(observed)} states; warm iters={env.residual_lp_iters}; "
              f"worst HiGHS gap={worst_gap * 100:.6f}%; "
              f"worst warm/cold-300 difference={worst_cold_difference * 100:.6f}%")
        self.assertLessEqual(worst_gap, .005, max(observed, key=lambda row: row[3]))
        self.assertLess(worst_cold_difference, .005, max(observed, key=lambda row: row[4]))


if __name__ == "__main__":
    unittest.main()
