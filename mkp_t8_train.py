"""T8 joint-M training using the new controller and ported v6 recipe. Synthetic only."""
import csv
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
import torch
from mkp_config import resolve_path
from mkp_env import MKPEnv,generate_instances
from mkp_Model import build_model
from mkp_artifacts import instance_sha,source_provenance,write_json
from mkp_reference import milp_reference
from mkp_v6_config import TrainConfig as V6Recipe
from mkp_v6_ppo import collect_rollout,PPOUpdate
import mkp_v6_lp as lp
import mkp_v6_ranking as ranking

SCHEMA="weak_decoder_t8_v6_v1"
CERT_SCHEMA="weak_decoder_t9_cert_v1"


def data_tuple(inst):
    return inst.p.double(),inst.r.transpose(1,2).double(),inst.c.double()


def save_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(rows)


class T8Trainer:
    def __init__(self,cfg,device):
        self.cfg,self.device=cfg,device;tc=cfg.train
        if cfg.model.executor=="cert" and (tc.signal!="item_pg" or cfg.model.v6_critic):
            raise ValueError("T9 cert supports item_pg only; no PPO/critic or v6 retraining")
        if cfg.model.executor not in ("v6","cert") or tc.signal not in ("item_pg","ppo_gae") or cfg.model.v6_critic!=(tc.signal=="ppo_gae"):
            raise ValueError("T8-A=item_pg/no critic; T8-B=ppo_gae/independent critic")
        if cfg.model.executor=="cert" and tc.signal!="item_pg":raise ValueError("T9 cert supports item_pg only; no PPO or v6 retraining")
        if tc.resume or tc.init_from:raise ValueError("T8 starts fresh; implicit resume/warm-start is disabled")
        self.model=build_model(cfg).to(device)
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=tc.lr,weight_decay=0.)
        self.generator=torch.Generator().manual_seed(cfg.device.seed)
        self.recipe=V6Recipe()
        for name in asdict(self.recipe):
            if hasattr(tc,name):setattr(self.recipe,name,getattr(tc,name))
        self.recipe.seed=cfg.device.seed;self.recipe.max_grad_norm=tc.grad_clip
        self.recipe.learning_rate=tc.lr;self.recipe.entropy_coefficient=tc.entropy_coef
        self.recipe.lp_reward_shaping=True;self.recipe.gamma=1.
        self.recipe.updates=tc.epochs*tc.steps_per_epoch
        self.recipe.num_resources=tuple(cfg.env.train_dims);self.recipe.train_num_items=tuple(cfg.env.train_sizes)
        self.updater=PPOUpdate(self.recipe,device)
        self.step=0;self.primary_trajectories=0;self.diagnostic_trajectories=0;self.ranking_diagnostic_trajectories=0
        self.rows=[];self.validation=[];self.best=float("inf")
        self.run=Path(resolve_path(tc.runs_dir))/tc.run_name;self.checkpoints=Path(resolve_path(tc.ckpt_dir))/tc.run_name

    def update(self,inst):
        tc=self.cfg.train;b=inst.B;stats={}
        if tc.signal=="ppo_gae":
            rollout=collect_rollout(self.model,data_tuple(inst))
            stats=self.updater.ppo_update(self.model,self.optimizer,data_tuple(inst),rollout,self.step)
            stats.update(rollout["theta_statistics"])
            stats["ranking_acceptance"]=stats["ranking_pairs"]/max(1,stats["ranking_proposed"])
            stats["ranking_retention"] = stats["ranking_return_retention"]
            stats["ranking_retention_definition"]="every25 updates: post-update paired terminal-return labels still accepted with the same sign; score agreement separately in ranking_post_accuracy"
            self.ranking_diagnostic_trajectories+=stats["ranking_diagnostic_trajectories"]
            self.primary_trajectories+=b
        else:
            k=tc.num_samples;per_chunk=max(1,tc.max_rollouts_per_chunk//k)
            self.model.train();self.optimizer.zero_grad(set_to_none=True)
            aggregate={"loss":0.,"entropy":0.,"sample_reward":0.};center_rewards=[];better=[]
            diagnostic=self.step%tc.log_every==0
            for start in range(0,b,per_chunk):
                end=min(start+per_chunk,b)
                from mkp_env import MKPInstances
                sub=MKPInstances(inst.p[start:end],inst.r[start:end],inst.c[start:end]);fraction=(end-start)/b
                out=self.model(MKPEnv(self.cfg.env),sub,num_samples=k,decode="sample")
                rewards=out["reward"].view(end-start,k);adv=rewards-rewards.mean(1,keepdim=True)
                if tc.adv_normalize:adv=adv/(adv.std()+1e-8)
                loss=-(adv.detach()*out["log_likelihood"].view(end-start,k)).mean()-tc.entropy_coef*out["entropy"].mean()
                if not bool(torch.isfinite(loss)):raise FloatingPointError("Non-finite T8-A loss")
                (loss*fraction).backward()
                aggregate["loss"]+=float(loss.detach())*fraction
                aggregate["entropy"]+=float(out["entropy"].mean().detach())*fraction
                aggregate["sample_reward"]+=float(rewards.mean().detach())*fraction
                for key,value in out["theta_statistics"].items():aggregate[key]=aggregate.get(key,0.)+value*fraction
                if diagnostic:
                    with torch.no_grad():
                        self.model.eval();center=self.model(MKPEnv(self.cfg.env),sub,decode="greedy")["reward"]
                        center_rewards.extend(center.tolist());better.extend((rewards>center[:,None]).float().mean(1).tolist())
                        self.model.train();self.diagnostic_trajectories+=end-start
            norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),tc.grad_clip,error_if_nonfinite=True)
            self.optimizer.step();stats=aggregate
            stats.update(grad_norm=float(norm),R_mean=float(np.mean(center_rewards)) if diagnostic else None,
                         frac_better=float(np.mean(better)) if diagnostic else None)
            self.primary_trajectories+=b*k
        self.step+=1
        return stats

    @torch.no_grad()
    def validate(self):
        self.model.eval();values=[]
        for inst,ref in self.validation:
            # Small instance chunks bound attention memory; one greedy per case.
            from mkp_env import MKPInstances
            for start in range(0,inst.B,8):
                sub=MKPInstances(inst.p[start:start+8],inst.r[start:start+8],inst.c[start:start+8])
                out=self.model(MKPEnv(self.cfg.env),sub,decode="greedy")
                values.extend((100*(1-out["profit"]/ref[start:start+8])).tolist())
        return float(np.mean(values))

    def checkpoint(self,name):
        path=self.checkpoints/name
        torch.save({"schema":CERT_SCHEMA if self.cfg.model.executor=="cert" else SCHEMA,"config":self.cfg.to_dict(),"executor_config":asdict(self.model.executor_engine.exec_cfg),
            "model":self.model.state_dict(),"optimizer":self.optimizer.state_dict(),"update":self.step,
            "best_validation_gap":self.best,"selection":"fixed synthetic N100 MILP-reference single-greedy mean gap; ties keep earlier",
            "training_data_source":"synthetic_only"},path)

    def fit(self):
        tc=self.cfg.train
        if self.cfg.model.executor=="cert" and not getattr(self,'gate_provenance',None):
            raise ValueError('Formal cert training requires scripts/run_t9.py with completed gates and explicit decision')
        if self.run.exists() or self.checkpoints.exists():raise FileExistsError("Preserve existing T8 run/checkpoint")
        self.run.mkdir(parents=True);self.checkpoints.mkdir(parents=True)
        self.cfg.save(str(self.run/"config.json"))
        meta={"schema":"t8_training_v1","status":"running","source":source_provenance(),"config":self.cfg.to_dict(),
            "recipe":asdict(self.recipe) if self.cfg.model.executor=='v6' else {
                'signal':'item_pg','num_samples':tc.num_samples,'entropy_coef':tc.entropy_coef,'learning_rate':tc.lr,
                'grad_clip':tc.grad_clip,'baseline':'same-instance shared mean','reward':'terminal profit/root UB','critic':False,'shaping':False},
            "device":str(self.device),"selection":"fixed synthetic N100 greedy gap only; no Chu checkpoint selection",
            "boundaries":"Ranking continuations are training-only frozen-policy labels, never PPO buffer/critic targets. "
                         "Inference is one greedy episode, with neither critic nor feasibility head.",
            "A_diagnostics":"R_mean/frac_better: pre-update greedy training-batch diagnostic every log_every; never a baseline/target",
            "reduction_rule":"At update250: if projected total>24h, A B64->32; B ranking_states16->8. Record actual budgets.",
            "runtime":"v6 float64 geometry/fp32 scores; new dual-axis controller. No memory24 geometry bank; state LP memo shared.",
            "validation":[],"reductions":[]}
        if self.cfg.model.executor=='cert':
            meta.update(schema='t9_cert_training_v1',gate_provenance=self.gate_provenance,
                runtime='float64 physical state/LP/cert bases, fp32 heads/scores; state-local LP/basis memo; cold solver, no AMP or speed optimization',
                boundaries='New cert-only training; no v6/v6c retraining, critic, shaping or continuation ranking. Inference is one greedy trajectory.')
        # Rebase recipe path fields before JSON serialization, no legacy paths.
        meta["recipe"]={k:str(v) if isinstance(v,Path) else v for k,v in meta["recipe"].items()}
        write_json(self.run/"meta.json",meta)
        gen=torch.Generator().manual_seed(tc.val_seed)
        for m in tc.val_dims:
            inst=generate_instances(tc.val_instances,100,m,self.cfg.env.train_tightness,self.cfg.env.weight_max,self.cfg.env.profit_noise,generator=gen)
            refs=milp_reference(inst,tc.val_milp_time_limit,tc.val_mip_rel_gap,tc.reference_cache)
            meta["validation"].append({"M":m,"data_sha256":instance_sha(inst),"reference_sha256":refs["sha256"],"proven":int(refs["optimal"].sum())})
            self.validation.append((inst.to(self.device),torch.tensor(refs["obj"],device=self.device,dtype=torch.float64)))
        start_lp=(lp.LP_SOLVE_BATCHES,lp.LP_SOLVE_STATES);start_rank=ranking.CONTINUATION_TRAJECTORIES
        began=time.perf_counter();update_times=[]
        initial=self.validate();self.best=initial
        self.rows.append({"update":0,"validation_gap":initial});self.checkpoint("best.pt")
        self.checkpoint("last.pt")
        validation_trajectories=len(tc.val_dims)*tc.val_instances
        try:
            for i in range(tc.epochs*tc.steps_per_epoch):
                n=self.cfg.env.train_sizes[int(torch.randint(len(self.cfg.env.train_sizes),(1,),generator=self.generator))]
                m=self.cfg.env.train_dims[int(torch.randint(len(self.cfg.env.train_dims),(1,),generator=self.generator))]
                inst=generate_instances(tc.batch_size,n,m,self.cfg.env.train_tightness,self.cfg.env.weight_max,self.cfg.env.profit_noise,
                                        generator=self.generator).to(self.device)
                if self.device.type=="cuda":torch.cuda.synchronize(self.device)
                tick=time.perf_counter();stats=self.update(inst)
                if self.device.type=="cuda":torch.cuda.synchronize(self.device)
                seconds=time.perf_counter()-tick;update_times.append(seconds)
                row={"update":self.step,"N":n,"M":m,"batch_size":tc.batch_size,"data_sha256":instance_sha(inst),"update_seconds":seconds,**stats,
                    "primary_trajectories":self.primary_trajectories,"diagnostic_trajectories":self.diagnostic_trajectories,
                    "ranking_trajectories":ranking.CONTINUATION_TRAJECTORIES-start_rank,
                    "LP_batch_solves":lp.LP_SOLVE_BATCHES-start_lp[0],"LP_state_solves_including_padding":lp.LP_SOLVE_STATES-start_lp[1]}
                if self.step%tc.steps_per_epoch==0:
                    row["validation_gap"]=self.validate();validation_trajectories+=len(tc.val_dims)*tc.val_instances
                    if row["validation_gap"]<self.best:self.best=row["validation_gap"];self.checkpoint("best.pt")
                if self.step==250:
                    projected=float(np.mean(update_times))*tc.epochs*tc.steps_per_epoch
                    meta["first_epoch_seconds_per_update"]=float(np.mean(update_times));meta["projected_training_hours"]=projected/3600
                    if projected>24*3600:
                        if tc.signal=="item_pg" and tc.batch_size==64:
                            tc.batch_size=32;meta["reductions"].append({"after_update":250,"batch_size":32})
                        elif tc.signal=="ppo_gae" and self.recipe.ranking_states==16:
                            self.recipe.ranking_states=8;tc.ranking_states=8;meta["reductions"].append({"after_update":250,"ranking_states":8})
                    print(f"T8 {tc.signal}: {np.mean(update_times):.2f}s/update, projected {projected/3600:.2f}h",flush=True)
                row.update(validation_trajectories=validation_trajectories,
                    LP_batch_solves=lp.LP_SOLVE_BATCHES-start_lp[0],LP_state_solves_including_padding=lp.LP_SOLVE_STATES-start_lp[1])
                self.rows.append(row)
                if self.step%100==0 or self.step%tc.steps_per_epoch==0:self.checkpoint("last.pt")
                if self.step%tc.log_every==0 or self.step==1:
                    save_csv(self.run/"train_log.csv",self.rows)
                    meta.update(actual_updates=self.step,actual_config=self.cfg.to_dict(),primary_trajectories=self.primary_trajectories,
                        diagnostic_trajectories=self.diagnostic_trajectories,validation_trajectories=validation_trajectories,
                        ranking_trajectories=ranking.CONTINUATION_TRAJECTORIES-start_rank,
                        LP_batch_solves=lp.LP_SOLVE_BATCHES-start_lp[0],LP_state_solves_including_padding=lp.LP_SOLVE_STATES-start_lp[1],
                        elapsed_hours=(time.perf_counter()-began)/3600,
                        GPU_hours=(time.perf_counter()-began)/3600 if self.device.type=="cuda" else 0.)
                    write_json(self.run/"meta.json",meta)
                    print(f"T8 {tc.signal} update {self.step}: {seconds:.2f}s, best {self.best:.4f}%",flush=True)
            meta["status"]="complete"
        except (Exception,KeyboardInterrupt) as exc:meta.update(status="failed",error=repr(exc));raise
        finally:
            meta.update(actual_updates=self.step,actual_config=self.cfg.to_dict(),primary_trajectories=self.primary_trajectories,
                diagnostic_trajectories=self.diagnostic_trajectories,validation_trajectories=validation_trajectories,
                ranking_trajectories=ranking.CONTINUATION_TRAJECTORIES-start_rank,
                ranking_diagnostic_trajectories=self.ranking_diagnostic_trajectories,
                total_trajectories=self.primary_trajectories+self.diagnostic_trajectories+validation_trajectories+ranking.CONTINUATION_TRAJECTORIES-start_rank,
                LP_batch_solves=lp.LP_SOLVE_BATCHES-start_lp[0],LP_state_solves_including_padding=lp.LP_SOLVE_STATES-start_lp[1],
                elapsed_hours=(time.perf_counter()-began)/3600,
                GPU_hours=(time.perf_counter()-began)/3600 if self.device.type=="cuda" else 0.,
                seconds_per_update=float(np.mean(update_times)) if update_times else None,
                accounting="LP and wall time include initial/periodic validation; exclude MILP-reference setup. GPU_hours is run wall allocation, not kernel-active hours.")
            self.checkpoint("last.pt")
            save_csv(self.run/"train_log.csv",self.rows);write_json(self.run/"meta.json",meta)
