"""Synthetic-only fixed-theta beta calibration; no Chu selection."""
import argparse, time
from t1r_common import *
from mkp_Trainer import set_seed

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="runs/dynamic_s1234/config.json")
    p.add_argument("--output",default="results/t1r_beta_calib")
    p.add_argument("--device",default="cpu")
    p.add_argument("--limit",type=int,default=0,help="smoke only: examples per group")
    a=p.parse_args(); torch.set_num_threads(2)
    cfg=config(a.config); device=torch.device(a.device); dest=fresh(a.output)
    env=MKPEnv(cfg.env); rows=[]; manifest=[]
    for name, inst, pre, refs in synthetic(cfg,device,a.limit):
        manifest.append({"name":name,"data_sha256":instance_sha(inst),"reference_sha256":refs["sha256"]})
        for beta in (1,2,4,8,16,32,64):
            cfg.model.temperature_init=1/beta
            model=build_model(cfg).to(device).eval()
            set_seed(cfg.device.seed)
            greedy=model(env,inst,decode="greedy",pre=pre,theta_override="fixed")
            out=model(env,inst,num_samples=16,pre=pre,theta_override="fixed")
            audit(inst,greedy);audit(inst,out)
            profits=out["profit"].reshape(inst.B,16).cpu()
            for b in range(inst.B):
                ref=float(refs["obj"][b]); gap=lambda v:100*(1-float(v)/ref)
                rows.append({"group":name,"id":b,"beta":beta,"greedy_gap":gap(greedy["profit"][b]),
                    "sample16_mean_gap":gap(profits[b].mean()),"best_of_16_gap":gap(profits[b].max()),
                    "mean_step_entropy":float(out["entropy"].reshape(inst.B,16)[b].mean())})
    summary=[]
    for beta in (1,2,4,8,16,32,64):
        rs=[r for r in rows if r["beta"]==beta]
        summary.append({"beta":beta,**{k:sum(r[k] for r in rs)/len(rs) for k in
            ("greedy_gap","sample16_mean_gap","best_of_16_gap","mean_step_entropy")}})
    chosen=min(summary,key=lambda r:(r["best_of_16_gap"],-r["beta"]))["beta"]
    assert max(r["greedy_gap"] for r in summary)-min(r["greedy_gap"] for r in summary)<1e-6
    write_csv(dest/"per_instance.csv",rows);write_csv(dest/"summary.csv",summary)
    write_json(dest/"meta.json",{"beta0":chosen,"selection":"synthetic fixed-theta best-of-16; ties larger beta",
        "smoke_only":bool(a.limit) or cfg.train.val_sizes != [100] or cfg.train.val_dims != [5,10] or cfg.train.val_instances != 64,"config":cfg.to_dict(),"data":manifest,"source":source_provenance()})
    print(f"beta0={chosen}; {dest}")
if __name__=="__main__": main()
