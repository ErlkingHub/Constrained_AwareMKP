"""Exactly two T5 tests; old modules live only in fresh bytecode-disabled children."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
LEGACY = Path(os.environ.get("T5_LEGACY_ROOT", "D:/PhD_Doctor/Study/Codes/RL_ModelInhance"))
CHECKPOINTS = Path(os.environ.get("T5_CHECKPOINT_DIR", str(ROOT / "checkpoints/lp_dual_v6")))

CHILD = r'''
import importlib.util, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
legacy, checkpoints, helper_path, mode = map(str, sys.argv[1:])
legacy, checkpoints, helper_path = Path(legacy).resolve(), Path(checkpoints).resolve(), Path(helper_path).resolve()
project = helper_path.parents[1]
sys.path[:] = [str(legacy)] + [p for p in sys.path if p and Path(p).resolve() not in (project, helper_path.parent)]
import torch
from mkp_env import MKPEnv
from mkp_utils import load_checkpoint, seed_everything
from mkp_Model import StrategyTheta
spec = importlib.util.spec_from_file_location("t5_theta_helpers", helper_path)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
seed_everything(1234, cpu_threads=2)

def fixture(n, m, seed):
    generator = torch.Generator().manual_seed(seed)
    p = torch.randint(1, 40, (1,n), generator=generator).double()
    w = torch.randint(1, 10, (1,n,m), generator=generator).double()
    return MKPEnv(p, w, torch.ceil(.5*w.sum(1))).state

with torch.no_grad():
    if mode == "injection":
        model, _ = load_checkpoint(checkpoints / "mkp_M5_best.pt", "cpu", expected_m=5)
        model.eval().requires_grad_(False)
        state = fixture(9,5,1701)
        original, theta, _ = model.policy(state)
        model.actor = helpers.injected_actor(helpers.theta_to_dict(theta), StrategyTheta)
        injected, reproduced, feasibility = model.policy(state)
        assert torch.equal(original.owners, injected.owners)
        assert torch.equal(original.items, injected.items)
        assert torch.equal(original.logits, injected.logits), "Injected own-theta logits changed"
        assert torch.equal(feasibility, torch.zeros_like(feasibility))
        assert all(torch.equal(getattr(theta,name),getattr(reproduced,name)) for name in helpers.THETA_FIELDS)
        print("Own learned theta injection: bitwise identical candidate logits, N9 M5 actual v6 checkpoint")
    else:
        for m in (5,10):
            model, _ = load_checkpoint(checkpoints / f"mkp_M{m}_best.pt", "cpu", expected_m=m)
            model.eval().requires_grad_(False)
            records=[]
            for index,n in enumerate((9,9,11)):
                state=fixture(n,m,1701+index)
                records.append(helpers.theta_to_dict(model.actor(state)[0]))
            global_theta=helpers.mean_theta(records,model.config)
            categories=[helpers.mean_theta(records[:2],model.config),helpers.mean_theta(records[2:],model.config)]
            for value in [global_theta,*categories]:
                assert helpers.validate_theta(value,model.config)
                assert torch.allclose(value["resource_prices"].mean(-1),torch.ones(1),atol=1e-6,rtol=0)
                assert torch.equal(value["resource_interactions"],value["resource_interactions"].transpose(-1,-2))
            assert helpers.validate_theta(helpers.get_default(model.config,state),model.config)
        print("G/C means: activated ranges, prices mean one and symmetric gate-bounded interactions, M5/M10")
for name,module in list(sys.modules.items()):
    if name.startswith("mkp_") and getattr(module,"__file__",None):
        assert Path(module.__file__).resolve().is_relative_to(legacy), (name,module.__file__)
assert sys.dont_write_bytecode
print(json.dumps({"mode":mode,"legacy_only":True,"bytecode_disabled":True}))
'''


class V6AttributionTests(unittest.TestCase):
    def setUp(self):
        if not (LEGACY / "mkp_Model.py").is_file() or any(
                not (CHECKPOINTS / f"mkp_M{m}_best.pt").is_file() for m in (5, 10)):
            self.skipTest("T5 requires the external v6 source and both local best checkpoints; configure T5_LEGACY_ROOT/T5_CHECKPOINT_DIR")

    def run_child(self, mode):
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-B", "-c", CHILD, str(LEGACY), str(CHECKPOINTS),
                                 str(ROOT / "scripts/t5_theta.py"), mode], cwd=ROOT, env=environment,
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)

    def test_own_learned_theta_injection_preserves_logits_bitwise(self):
        self.run_child("injection")

    def test_global_and_category_means_are_valid_activated_theta(self):
        self.run_child("means")


if __name__ == "__main__":
    unittest.main()
