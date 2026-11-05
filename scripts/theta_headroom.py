"""Static-theta reachable headroom diagnostic; never a method score or a selector on Chu."""
import argparse, time, numpy as np
from t1r_common import *

@torch.no_grad()
def search(model, env, inst, pre, seed, random_budget=256, local_budget=256):
    g=torch.Generator().manual_seed(seed); m=inst.M
    def project(u):
        u=u-u.mean(-1,keepdim=True)
        return u*(3/u.abs().amax(-1,keepdim=True).clamp_min(3))
    us=project(6*torch.rand(random_budget,m,generator=g)-3)
    alphas=.5+2*torch.rand(random_budget,generator=g)
    us[0]=0;alphas[0]=1
    def evaluate(u,alpha):
        out=model(env,inst,num_samples=len(alpha),decode="greedy",pre=pre,
                  theta_override={"u":u.to(inst.p.device),"alpha":alpha.to(inst.p.device)})
        audit(inst,out);return out
    out=evaluate(us,alphas);fixed=float(out["profit"][0]);i=int(out["profit"].argmax())
    best=float(out["profit"][i]);u=us[i].clone();alpha=alphas[i].clone();solution=out["solution"][i].cpu().tolist()
    for t in range(local_budget):
        j=(t//2)%(m+1); sign=1 if t%2==0 else -1
        step=.5*(.5**(t//(2*(m+1)*8)))
        candidate=u.clone(); ca=alpha.clone()
        if j==m: ca=(ca+sign*step).clamp(.5,2.5)
        else: candidate[j]+=sign*step;candidate=project(candidate[None])[0]
        cur=evaluate(candidate[None],ca[None]);obj=float(cur["profit"][0])
        if obj>best:
            best,u,alpha,solution=obj,candidate,ca,cur["solution"][0].cpu().tolist()
    return {"fixed_objective":fixed,"objective":best,"u":u.tolist(),"alpha_best":float(alpha),
            "solution":solution,"evaluations":random_budget+local_budget}

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="runs/dynamic_s1234/config.json")
    p.add_argument("--output",default="results/t1r_headroom")
    p.add_argument("--device",default="cpu");p.add_argument("--dataset",choices=["synthetic","chu"],default="synthetic")
    p.add_argument("--limit",type=int,default=0);p.add_argument("--random_budget",type=int,default=256);p.add_argument("--local_budget",type=int,default=256)
    a=p.parse_args();torch.set_num_threads(2)
    if a.random_budget<1 or a.local_budget<0: raise ValueError("invalid diagnostic budget")
    cfg=config(a.config);dev=torch.device(a.device);env=MKPEnv(cfg.env);model=build_model(cfg).to(dev).eval();dest=fresh(a.output)
    cases=[];manifest=[]
    if a.dataset=="synthetic":
        for name, inst, pre, refs in synthetic(cfg,dev,a.limit):
            manifest.append({"data_sha256":instance_sha(inst),"reference_sha256":refs["sha256"]})
            for b in range(inst.B):
                one=type(inst)(inst.p[b:b+1],inst.r[b:b+1],inst.c[b:b+1]);cases.append((f"{name}_{b}",one,(pre[0][b:b+1],pre[1][b:b+1]),float(refs["obj"][b])))
    else:
        data=[d for d in read_mat(resolve_path(cfg.eval.mat_file)) if d["N"]==100 and d["M"] in (5,10)]
        if a.limit:data=data[:a.limit]
        manifest=[{"data_sha256":sha256_file(resolve_path(cfg.eval.mat_file))}]
        for d in data:
            inst=instances_from_dicts([d],dev);cases.append((d["id"],inst,env.precompute(inst),float(d["best"])))
    rows=[];scan=[];witness=[]
    for idx,(name,inst,pre,ref) in enumerate(cases):
        start=time.perf_counter();ratio=round(float(inst.c.sum()/inst.r.sum()),2)
        for alpha in (.5,.75,1,1.25,1.5,2,2.5):
            out=model(env,inst,decode="greedy",pre=pre,theta_override={"u":torch.zeros_like(pre[0]),"alpha":torch.tensor([alpha],device=dev)})
            audit(inst,out);scan.append({"id":name,"M":inst.M,"tightness":ratio,"alpha":alpha,"gap_pct":100*(1-float(out["profit"][0])/ref)})
        result=search(model,env,inst,pre,cfg.device.seed+idx,a.random_budget,a.local_budget)
        delta=100*(result["fixed_objective"]-result["objective"])/ref
        rows.append({"id":name,"M":inst.M,"tightness":ratio,"fixed_gap_pct":100*(1-result["fixed_objective"]/ref),"reachable_gap_pct":100*(1-result["objective"]/ref),"gap_delta_pp":delta,"improves_over_0_1pp":delta<-.1,"alpha_best":result["alpha_best"],"u_abs_mean":float(np.mean(np.abs(result["u"]))),"u_boundary_frac":float(np.mean(np.abs(result["u"])>2.85)),"time_s":time.perf_counter()-start,"evaluations":result["evaluations"]})
        witness.append({"id":name,**result})
    base={r["id"]:r["gap_pct"] for r in scan if r["alpha"]==1}
    for r in scan:r["delta_vs_alpha1_pp"]=r["gap_pct"]-base[r["id"]]
    scan_summary=[]
    for alpha in (.5,.75,1,1.25,1.5,2,2.5):
        for m in ("all",5,10):
            for ratio in ("all",.25,.5,.75):
                rs=[r for r in scan if r["alpha"]==alpha and (m=="all" or r["M"]==m) and (ratio=="all" or r["tightness"]==ratio)]
                if rs:scan_summary.append({"alpha":alpha,"M":m,"tightness":ratio,"count":len(rs),"mean_gap_pct":float(np.mean([r["gap_pct"] for r in rs])),"mean_delta_pp":float(np.mean([r["delta_vs_alpha1_pp"] for r in rs]))})
    write_csv(dest/"alpha_scan_summary.csv",scan_summary)
    write_csv(dest/"alpha_scan.csv",scan);write_csv(dest/"per_instance.csv",rows);write_json(dest/"best_theta.json",witness)
    summary=[]
    for m in ("all",5,10):
        for ratio in ("all",.25,.5,.75):
            rs=[r for r in rows if (m=="all" or r["M"]==m) and (ratio=="all" or r["tightness"]==ratio)]
            if not rs:continue
            ds=[r["gap_delta_pp"] for r in rs]
            summary.append({"M":m,"tightness":ratio,"count":len(rs),"mean_delta_pp":float(np.mean(ds)),"median_delta_pp":float(np.median(ds)),"p90_delta_pp":float(np.percentile(ds,90)),"improves_over_0_1pp_fraction":float(np.mean([r["improves_over_0_1pp"] for r in rs]))})
    write_csv(dest/"summary.csv",summary)
    write_json(dest/"meta.json",{"dataset":a.dataset,"diagnostic_only":True,"benchmark_diagnostic":a.dataset=="chu","interpretation":"lower bound on reachable improvement; failure is not impossibility; no selection on Chu", "arguments":vars(a),"config":cfg.to_dict(),"source":source_provenance(),"data":manifest})
if __name__=="__main__":main()
