"""User-run forward/backward traces for N50/100/150, without optimization changes."""
import argparse
from t1r_common import *
from mkp_Trainer import set_seed

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",required=True)
    p.add_argument("--device",default="cuda:0");p.add_argument("--output",default="results/t1r_profile")
    a=p.parse_args();cfg=config(a.config);dev=torch.device(a.device);dest=fresh(a.output);torch.set_num_threads(2)
    activities=[torch.profiler.ProfilerActivity.CPU]
    if dev.type=="cuda":activities.append(torch.profiler.ProfilerActivity.CUDA)
    for n in (50,100,150):
        set_seed(cfg.device.seed);env=MKPEnv(cfg.env);model=build_model(cfg).to(dev).train()
        inst=env.generate(cfg.train.batch_size,n,5).to(dev)
        with torch.profiler.profile(activities=activities,record_shapes=True) as prof:
            with torch.profiler.record_function("t1r/LP"):pre=env.precompute(inst)
            out=model(env,inst,num_samples=cfg.train.num_samples,amp=cfg.device.amp,pre=pre)
            R=out["reward"].reshape(inst.B,cfg.train.num_samples);adv=R-R.mean(1,keepdim=True)
            if cfg.train.adv_normalize:adv=adv/(adv.std()+1e-8)
            loss=-(adv.detach().reshape(-1)*out["log_likelihood"]).mean()-cfg.train.entropy_coef*out["entropy"].mean()
            with torch.profiler.record_function("t1r/backward_total"):loss.backward()
        prof.export_chrome_trace(str(dest/f"N{n}.json"))
        (dest/f"N{n}.txt").write_text(prof.key_averages().table(sort_by="self_cpu_time_total",row_limit=80),encoding="utf-8")
    write_json(dest/"meta.json",{"config":cfg.to_dict(),"source":source_provenance(),"note":"No optimizer step; inspect autograd events for controller backward attribution; total backward includes encoder and scorer. Recomputed checkpoint controller ranges occur during backward. Does not establish historical stop cause or GPU sharing."})
if __name__=="__main__":main()
