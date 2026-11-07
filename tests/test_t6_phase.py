"""Only the two T6 checks; legacy modules are confined to fresh child processes."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
LEGACY = Path(os.environ.get("T5_LEGACY_ROOT", "D:/PhD_Doctor/Study/Codes/RL_ModelInhance"))
CHECKPOINTS = Path(os.environ.get("T5_CHECKPOINT_DIR", str(ROOT/"checkpoints/lp_dual_v6")))
RESULTS = Path(os.environ.get("T5_RESULTS_DIR", str(ROOT/"results/t5_v6/attribution")))

CHILD = r'''
import importlib.util, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
project,legacy,checkpoints,results = [Path(p).resolve() for p in sys.argv[1:5]]
mode=sys.argv[5]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
t5=load("t6_test_t5",project/"scripts/t5_v6_attribution.py")
(np,torch,lp,Theta,Env,Instance,batch,load_benchmark,construct,load_checkpoint,resolve_device,seed,verify,helper)=t5.bootstrap(legacy)
phase=load("t6_test_schedule",project/"scripts/t6_schedule.py")
seed(20260910,cpu_threads=2)
def fixture(n,m,s):
    g=torch.Generator().manual_seed(s)
    p=torch.randint(1,40,(1,n),generator=g).double()
    w=torch.randint(1,10,(1,n,m),generator=g).double()
    return p,w,torch.ceil(.5*w.sum(1))
with torch.no_grad():
    if mode=="constant":
        means=json.loads((results/"theta_means.json").read_text(encoding="utf-8"))
        for m in (5,10):
            model,_=load_checkpoint(checkpoints/f"mkp_M{m}_best.pt","cpu",expected_m=m)
            model.eval().requires_grad_(False)
            G=helper.theta_to_dict(means["G"][str(m)])
            for n in (9,11):
                data=fixture(n,m,1701+n)
                model.actor=helper.injected_actor(G,Theta)
                original=construct(data,model,stochastic=False)
                model.actor=phase.ScheduleActor(phase.constant_table(G),Theta,model.config,helper)
                scheduled=construct(data,model,stochastic=False)
                for field in ("selected","sequence","objective"):
                    assert torch.equal(original[field],scheduled[field]),field
        print("Constant G schedule: bitwise identical selections, sequence and objective on four synthetic fixtures")
    else:
        nodes=torch.tensor([.2,.5,.8],dtype=torch.float64)
        values=torch.tensor([[.5,1.],[1.,1.5],[1.75,2.]],dtype=torch.float32)
        assert torch.equal(phase.interpolate(nodes,values,nodes),values)
        endpoints=phase.interpolate(nodes,values,torch.tensor([-1.,2.]))
        assert torch.equal(endpoints,values[[0,2]])
        midpoint=phase.interpolate(nodes,values,torch.tensor([.35,.65],dtype=torch.float64))
        assert torch.equal(midpoint,torch.tensor([[.75,1.25],[1.375,1.75]]))
        env=Env(*fixture(12,3,4));history=[]
        while True:
            current=phase.clocks(env.state)
            history.append(torch.stack([current[n][0] for n in phase.CLOCKS]))
            if not env.state.active.any():break
            item=env.state.feasible[0].nonzero()[0,0]
            env.step(item.reshape(1).long())
        history=torch.stack(history)
        assert bool(((history>=0)&(history<=1)).all())
        assert bool((history[1:]>=history[:-1]-1e-12).all())
        model,_=load_checkpoint(checkpoints/"mkp_M5_best.pt","cpu",expected_m=5)
        G=helper.theta_to_dict(json.loads((results/"theta_means.json").read_text(encoding="utf-8"))["G"]["5"])
        records=[{"dataset":"synthetic_trace","id":str(i),"M":5,"N":100,"tightness":.5,"tau_cap":.2,"theta":G} for i in range(12)]
        table=phase.fit_tables(records,"category","tau_cap",{5:model.config},helper)["5|100|0.5"]
        assert table["knots"]==[.2] and sum(table["bin_counts"])==12
        assert torch.equal(torch.tensor(table["base"]["resource_prices"]),torch.ones(1,5))
        print("Interpolation: exact knots/endpoints/midpoints; clocks monotone in [0,1]; tied quantiles valid and prices uniform")
for name,module in list(sys.modules.items()):
    if name.startswith("mkp_") and getattr(module,"__file__",None):
        assert Path(module.__file__).resolve().is_relative_to(legacy)
'''


class T6Tests(unittest.TestCase):
    def setUp(self):
        if (not (LEGACY/"mkp_Model.py").is_file() or not (RESULTS/"theta_means.json").is_file() or
                any(not (CHECKPOINTS/f"mkp_M{m}_best.pt").is_file() for m in (5,10))):
            self.skipTest("External v6 source, checkpoints and completed T5 theta means are required")

    def child(self, mode):
        result=subprocess.run([sys.executable,"-B","-c",CHILD,str(ROOT),str(LEGACY),str(CHECKPOINTS),str(RESULTS),mode],
            cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1"),capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=120)
        self.assertEqual(result.returncode,0,result.stdout+"\n"+result.stderr)

    def test_constant_G_schedule_matches_T5_G_bitwise(self):
        self.child("constant")

    def test_interpolation_endpoints_and_monotone_deployable_clocks(self):
        self.child("clocks")


if __name__ == "__main__":
    unittest.main()
