"""Exactly two T7 tests; all legacy imports stay in isolated child processes."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
LEGACY=Path(os.environ.get("T5_LEGACY_ROOT","D:/PhD_Doctor/Study/Codes/RL_ModelInhance"))
CHECKPOINTS=Path(os.environ.get("T5_CHECKPOINT_DIR",str(ROOT/"checkpoints/lp_dual_v6")))
T6=Path(os.environ.get("T6_RESULTS_DIR",str(ROOT/"results/t6_phase")))
CHILD=r'''
import importlib.util,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
project,legacy,checkpoints,t6=[Path(p).resolve() for p in sys.argv[1:5]];mode=sys.argv[5]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
t5=load("t7_test_t5",project/"scripts/t5_v6_attribution.py")
(np,torch,lp,Theta,Env,Instance,batch,load_benchmark,construct,load_checkpoint,resolve,seed,verify,helper)=t5.bootstrap(legacy)
from mkp_Model import CandidatePolicy
from mkp_performance import prepare_state,cached_features
phase=load("t7_test_phase",project/"scripts/t6_schedule.py");core=load("t7_test_core",project/"scripts/t7_interface.py")
seed(20260910,cpu_threads=2)
model,_=load_checkpoint(checkpoints/"mkp_M5_best.pt","cpu",expected_m=5)
model.eval().requires_grad_(False);learned=model.actor;cfg=model.config
table=json.loads((t6/"schedules.json").read_text(encoding="utf-8"))["tables"]["category/tau_cap"]["5|100|0.5"]
def schedule():return phase.ScheduleActor(table,Theta,cfg,helper)
def frame(state):return core.StateScores(state,cfg,model.decoder,CandidatePolicy,prepare_state,cached_features)
with torch.no_grad():
    if mode=="singleton":
        g=torch.Generator().manual_seed(71)
        p=torch.randint(1,40,(1,9),generator=g).double();w=torch.randint(1,9,(1,9,5),generator=g).double()
        data=(p,w,torch.ceil(.5*w.sum(1)))
        state=Env(*data).state;original,theta,_=model.policy(state);f=frame(state)
        values=helper.theta_to_dict(theta)
        assert torch.equal(f.logits(values)[0],original.logits),"Cached score arithmetic changed"
        assert set(f.choices(values).tolist())==set(original.choose(False).tolist())
        for threshold in (0.,1.):
            model.actor=learned if threshold==0 else schedule()
            expected=construct(data,model,False)
            actor=core.ControlActor(learned,schedule(),phase,Theta,threshold=threshold)
            model.actor=actor;actual=construct(data,model,False)
            for name in ("objective","sequence","selected"):assert torch.equal(actual[name],expected[name]),name
        print("Singleton L reachability/logits and SL(0)/SL(1) exact endpoint regressions passed")
    else:
        data=(torch.tensor([[3.,4.,2.,1.]],dtype=torch.float64),torch.ones(1,4,5,dtype=torch.float64),torch.full((1,5),4.,dtype=torch.float64))
        def provider(state,executor):
            f=frame(state);tau=float(phase.clocks(state)["tau_cap"][0]);center=core.schedule_theta(table,tau,phase)
            draws=core.global_draws(cfg,5,8,torch.Generator().manual_seed(91+int(state.selected.sum())))
            reach=torch.unique(torch.cat((f.choices(draws,executor),f.choices(center,executor))))
            return f,reach,f.ranking(center,executor)
        for executor in ("E_full","E_min"):
            env=Env(*data);target=torch.ones(4,dtype=torch.bool)
            result,audit=core.oracle_rollout(env,target,provider,executor)
            inst=Instance("fixture",data[0][0].numpy(),data[1][0].numpy(),data[2][0].numpy())
            verify(inst,result,1e-7)
            assert audit["all_target_intersections_nonempty"] and audit["target_prefix_subset"] and audit["final_subset_target"]
            # A single actual S2 witness exercises both fallback and completion.
            def one(state,which):
                f=frame(state);center=core.schedule_theta(table,float(phase.clocks(state)["tau_cap"][0]),phase)
                return f,f.choices(center,which),f.ranking(center,which)
            first=int(one(Env(*data).state,executor)[1][0])
            target=torch.zeros(4,dtype=torch.bool);target[first]=True
            result,audit=core.oracle_rollout(Env(*data),target,one,executor);verify(inst,result,1e-7)
            assert audit["all_target_intersections_nonempty"] and audit["target_prefix_subset"] and len(audit["completion_items"])==3
            target=torch.zeros(4,dtype=torch.bool);target[(first+1)%4]=True
            result,audit=core.oracle_rollout(Env(*data),target,one,executor);verify(inst,result,1e-7)
            assert audit["first_empty"]["step"]==0 and not audit["all_target_intersections_nonempty"]
        print("Oracle per-step legality, maximality, target-only prefix, S2 completion and fallback passed for both dictionaries")
for name,module in list(sys.modules.items()):
    if name.startswith("mkp_") and getattr(module,"__file__",None):assert Path(module.__file__).resolve().is_relative_to(legacy)
'''


class T7Tests(unittest.TestCase):
    def setUp(self):
        if not (LEGACY/"mkp_Model.py").is_file() or not (CHECKPOINTS/"mkp_M5_best.pt").is_file() or not (T6/"schedules.json").is_file():
            self.skipTest("External v6 source/checkpoint and complete T6 schedule are required")

    def child(self,mode):
        result=subprocess.run([sys.executable,"-B","-c",CHILD,str(ROOT),str(LEGACY),str(CHECKPOINTS),str(T6),mode],
            cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1"),capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=120)
        self.assertEqual(result.returncode,0,result.stdout+"\n"+result.stderr)

    def test_singleton_reachability_and_switch_endpoints(self):self.child("singleton")

    def test_oracle_feasibility_maximality_and_target_prefix(self):self.child("oracle")


if __name__=="__main__":unittest.main()
