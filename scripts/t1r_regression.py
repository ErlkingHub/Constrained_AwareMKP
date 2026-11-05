"""User-run full greedy regression of T1 raw versus normalized scores."""
import argparse, copy
from t1r_common import *
from mkp_Trainer import load_policy

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="runs/dynamic_s1234/config.json")
    p.add_argument("--dynamic",default="checkpoints/dynamic_s1234/best.pt")
    p.add_argument("--static",default="checkpoints/static_s1234/best.pt")
    p.add_argument("--fixed_reference",default="results/t1_fixed180/20261102_140130_251675/per_instance.csv")
    p.add_argument("--output",default="results/t1r_greedy_regression")
    p.add_argument("--device",default="cpu");p.add_argument("--limit",type=int,default=0)
    a=p.parse_args();torch.set_num_threads(2);dev=torch.device(a.device);cfg=config(a.config);dest=fresh(a.output)
    import csv
    with open(resolve_path(a.fixed_reference),encoding="utf-8-sig",newline="") as f:
        baseline={str(r["id"]):float(r["objective"]) for r in csv.DictReader(f) if r["arm"]=="fixed"}
    models={"fixed":(build_model(cfg).to(dev).eval(),cfg)}
    hashes={}
    for arm,path in (("learned_dynamic",a.dynamic),("learned_static",a.static)):
        model,mc,_=load_policy(resolve_path(path),dev);models[arm]=(model,mc);hashes[arm]=sha256_file(resolve_path(path))
    data=[d for d in read_mat(resolve_path(cfg.eval.mat_file)) if d["M"] in (5,10) and d["N"] in (100,250,500)]
    if a.limit:data=data[:a.limit]
    rows=[]
    for arm,(model,mc) in models.items():
        mismatches=0;env=MKPEnv(mc.env)
        for d in data:
            inst=instances_from_dicts([d],dev);pre=env.precompute(inst);kwargs={"theta_override":"fixed"} if arm=="fixed" else {}
            model.decoder.normalize=False;before=model(env,inst,decode="greedy",pre=pre,**kwargs)
            model.decoder.normalize=True;after=model(env,inst,decode="greedy",pre=pre,**kwargs)
            audit(inst,before);audit(inst,after)
            changed=float(before["profit"][0])!=float(after["profit"][0])
            stored_mismatch=arm=="fixed" and float(after["profit"][0])!=baseline[str(d["id"])]
            mismatches+=int(changed or stored_mismatch)
            rows.append({"arm":arm,"id":d["id"],"before":float(before["profit"][0]),"after":float(after["profit"][0]),"changed":changed,"stored_reference_mismatch":stored_mismatch})
            write_csv(dest/"per_instance.csv",rows)
            if mismatches>2:raise RuntimeError(f"More than 2 changed instances in {arm}; stop and report before training")
    write_json(dest/"meta.json",{"source":source_provenance(),"checkpoint_sha256":hashes,"smoke_only":bool(a.limit),"mismatch_count":sum(r["changed"] or r["stored_reference_mismatch"] for r in rows),"note":"Any changed instance needs tie inspection before training", "data_sha256":sha256_file(resolve_path(cfg.eval.mat_file))})
if __name__=="__main__":main()
