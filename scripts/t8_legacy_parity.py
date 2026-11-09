"""Isolated read-only legacy exporter for the T8 six-shape D/G equivalence test."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--legacy_root",required=True);parser.add_argument("--checkpoints",required=True)
    parser.add_argument("--means",required=True);parser.add_argument("--output",required=True);parser.add_argument("--device",default="cpu")
    args=parser.parse_args();root=Path(args.legacy_root).resolve();dest=Path(args.output).resolve()
    if dest.is_relative_to(root):raise ValueError("Legacy project is read-only")
    sys.path[:]=[str(root)]+[p for p in sys.path if p and Path(p).resolve()!=Path(__file__).resolve().parents[1]]
    import numpy as np
    import torch
    from mkp_config import ModelConfig
    from mkp_Model import StrategyTheta,WeakDecoder
    from mkp_env import MKPEnv
    from mkp_data import load_benchmark,instance_batch
    from mkp_lp import lp_relaxation
    from mkp_utils import seed_everything
    seed_everything(1234,cpu_threads=2)
    data=load_benchmark(root/"MKP_dataset/Benchmarks_Chu.mat",resources=(5,10),items=(100,250,500))
    means=json.loads(Path(args.means).read_text(encoding="utf-8"));arrays={};cases=[]
    for m in (5,10):
        ck=torch.load(Path(args.checkpoints)/f"mkp_M{m}_best.pt",map_location="cpu",weights_only=True)
        cfg=ModelConfig(**ck["model_config"]);decoder=WeakDecoder(cfg)
        for n in (100,250,500):
            inst=next(i for i in data if i.m==m and i.n==n)
            for arm in ("D","G"):
                index=len(cases);env=MKPEnv(*instance_batch(inst,1,args.device))
                theta=StrategyTheta(**{k:torch.tensor(v,device=args.device,dtype=torch.float32) for k,v in means[arm][str(m)].items()})
                steps=[]
                while bool(env.state.active[0]):
                    state=env.state;bound=lp_relaxation(state,cfg).bound;policy=decoder(state,theta)
                    t=len(steps);action=policy.choose(False)
                    arrays[f"items_{index}_{t}"]=policy.items.cpu().numpy()
                    arrays[f"scores_{index}_{t}"]=policy.score_terms["score"].cpu().numpy()
                    arrays[f"bound_{index}_{t}"]=bound.cpu().numpy();steps.append(int(action[0]));env.step(action)
                arrays[f"p_{index}"]=inst.profits;arrays[f"w_{index}"]=inst.weights;arrays[f"c_{index}"]=inst.capacities
                arrays[f"selected_{index}"]=env.state.selected.cpu().numpy();arrays[f"sequence_{index}"]=np.asarray(steps)
                cases.append({"id":inst.instance_id,"M":m,"N":n,"arm":arm,"steps":len(steps),"config":ck["model_config"]})
                print(f"legacy parity {arm} M{m} N{n}: {len(steps)}",flush=True)
    for name,module in list(sys.modules.items()):
        if name.startswith("mkp_") and getattr(module,"__file__",None):
            if not Path(module.__file__).resolve().is_relative_to(root):raise AssertionError(name)
    arrays["metadata"]=np.asarray(json.dumps(cases));dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open("xb") as stream:np.savez_compressed(stream,**arrays)


if __name__=="__main__":main()
