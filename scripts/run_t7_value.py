"""T7 single-episode value-location and static-state capacity diagnostics. No training."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def jsonl(path):
    with path.open(encoding="utf-8") as stream:return [json.loads(line) for line in stream if line.strip()]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy_root",default="D:/PhD_Doctor/Study/Codes/RL_ModelInhance")
    parser.add_argument("--checkpoint_dir",default="checkpoints/lp_dual_v6")
    parser.add_argument("--instances_dir",default="results/t5_v6")
    parser.add_argument("--t5_results",default="results/t5_v6/attribution")
    parser.add_argument("--t6_results",default="results/t6_phase")
    parser.add_argument("--output",default="results/t7_value")
    parser.add_argument("--device",default="auto")
    parser.add_argument("--stages",default="policy,capacity,oracle",help="independent stages; use distinct output directories")
    parser.add_argument("--oracle_targets",choices=("all","proven"),default="all")
    parser.add_argument("--theta_chunk",type=int,default=128)
    parser.add_argument("--reference_cache",default="cache")
    parser.add_argument("--limit",type=int,choices=(0,2),default=0)
    args=parser.parse_args(argv)
    stages=args.stages.split(",")
    if len(set(stages))!=len(stages) or not set(stages)<={"policy","capacity","oracle"} or args.theta_chunk<1:
        parser.error("Invalid stage list or theta chunk")
    t5=load("_t7_t5",PROJECT/"scripts/t5_v6_attribution.py")
    root,checkpoints,source,previous,t6root,dest=map(t5.path_from_project,
        (args.legacy_root,args.checkpoint_dir,args.instances_dir,args.t5_results,args.t6_results,args.output))
    if dest.exists() or dest.is_relative_to(root):raise ValueError("Output must be new and outside the read-only legacy project")
    export=json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    prior=json.loads((previous/"meta.json").read_text(encoding="utf-8"))
    t6meta=json.loads((t6root/"meta.json").read_text(encoding="utf-8"))
    frozen=json.loads((t6root/"report_manifest.json").read_text(encoding="utf-8"))
    for meta in (export,prior,t6meta):
        if meta.get("status")!="complete" or meta.get("smoke_only") is not False:raise ValueError("Formal T5/T6 inputs are required")
    if (prior["export_manifest_sha256"]!=t5.sha256(source/"manifest.json") or
            t6meta["export_manifest_sha256"]!=t5.sha256(source/"manifest.json") or
            t6meta["t5_rows_sha256"]!=t5.sha256(previous/"rows.jsonl")):
        raise ValueError("T5/T6 instance/reference contracts changed")
    for name in ("rows.jsonl","schedules.json","traces.jsonl"):
        expected=(frozen["input_sha256"] if name=="rows.jsonl" else frozen["fit_artifact_sha256"])[name]
        if t5.sha256(t6root/name)!=expected:raise ValueError(f"Frozen T6 artifact changed: {name}")
    before=t5.legacy_snapshot(root)
    if before["source_sha256"]!=prior["legacy_before"]["source_sha256"] or not t6meta["legacy_unchanged"]:
        raise ValueError("Legacy executor differs from T5/T6")
    dest.mkdir(parents=True)
    meta={"schema":"t7_value_v1","status":"running","smoke_only":bool(args.limit),"arguments":vars(args),"stages":stages,
        "legacy_before":before,"global_draws":32 if args.limit else 2048,"local_draws":16 if args.limit else 512,
        "input_sha256":{str(p):t5.sha256(p) for p in (source/"manifest.json",t6root/"rows.jsonl",t6root/"traces.jsonl",t6root/"schedules.json")},
        "source":{"commit":t5.git(PROJECT,"rev-parse","HEAD"),"status":t5.git(PROJECT,"status","--short"),
                  "sha256":{p.name:t5.sha256(p) for p in (PROJECT/"scripts").glob("*.py") if p.name.startswith(("t5","t6","t7")) or p.name.startswith("run_t7")}},
        "sampling":"Scalar box-uniform; price logits U[-4,4]; uniform total combination mass times Dirichlet(1) allocation. "
                   "Local activated Gaussian draws are clipped/radially projected into legal budgets. Same probes for both executors; independent global/local state seeds.",
        "residuals":"Only T6 synthetic fitting traces. Population residual std per category/tau_cap quantile bin; "
                    "exchangeable pooled price/upper-triangle scales, companion per component, scalar per field. Empty bins use category std.",
        "interpretation":"Finite sampled reachable sets are lower bounds on reachability. Empty sampled intersections are not infeasibility certificates. "
                         "Target-aware oracle is a single diagnostic episode, not an inference method or a certified ceiling.",
        "timing":"Synchronized instrumented total/LP/controller-inclusive-LP/controller-minus-LP times; serialization excluded.",
        "oracle_targets":args.oracle_targets,"seed":20260910}
    t5.write_json(dest/"meta.json",meta)
    streams=[]
    try:
        (np,torch,lp,Theta,Env,Instance,batch,load_benchmark,construct,load_checkpoint,resolve_device,seed,verify,helper)=t5.bootstrap(root)
        from mkp_Model import CandidatePolicy
        from mkp_performance import prepare_state,cached_features
        phase=load("_t7_phase",PROJECT/"scripts/t6_schedule.py")
        core=load("_t7_core",PROJECT/"scripts/t7_interface.py")
        seed(20260910,deterministic=True,cpu_threads=4);device=resolve_device(args.device)
        evaluation,_=t5.read_bundle(np,Instance,source,export,"eval","synthetic_N100",False)
        dataset=root/"MKP_dataset/Benchmarks_Chu.mat"
        if t5.sha256(dataset)!=prior["dataset_sha256"]:raise ValueError("Chu dataset changed")
        instances=load_benchmark(dataset,resources=(5,10),items=(100,250,500))
        if Counter((i.m,i.n) for i in instances)!={(m,n):30 for m in (5,10) for n in (100,250,500)}:raise ValueError("Incomplete Chu coverage")
        chu=[]
        for inst in instances:
            ratio=float(inst.capacities.sum()/inst.weights.sum())
            chu.append({"id":inst.instance_id,"dataset":"chu","M":inst.m,"N":inst.n,"inst":inst,
                "tightness":min((.25,.5,.75),key=lambda a:abs(a-ratio)),"reference":float(inst.reference),
                "reference_kind":"Chu best known; not proven optimum","reference_proven":False,
                "data_sha256":t5.data_hash(np,inst.profits,inst.weights,inst.capacities)})
        if len({r["id"] for r in chu})!=180:raise ValueError("Duplicated Chu IDs")
        if args.limit:
            evaluation=[next(r for r in evaluation if r["M"]==m) for m in (5,10)]
            chu=[next(r for r in chu if r["M"]==m and r["N"]==100) for m in (5,10)]
        records=evaluation+chu;by_id={(r["dataset"],r["id"]):r for r in records}
        references={}
        for row in jsonl(t6root/"rows.jsonl"):
            k=(row["dataset"],row["id"])
            if k not in by_id or row["arm"] not in ("L","G","S2"):continue
            for name in ("M","N","tightness","reference","reference_kind","reference_proven","data_sha256"):
                if row[name]!=by_id[k][name]:raise ValueError(f"Reference mismatch: {k}, {name}")
            key=(*k,row["arm"])
            if key in references:raise ValueError("Duplicate reference")
            references[key]={**row,"reused_reference":True}
        if len(references)!=3*len(records):raise ValueError("Missing L/G/S2 references")
        models,actors,configs={},{},{}
        for m in (5,10):
            path=checkpoints/f"mkp_M{m}_best.pt"
            if t5.sha256(path)!=prior["checkpoints"][str(m)]["sha256"]:raise ValueError("Checkpoint changed")
            model,payload=load_checkpoint(path,device,expected_m=m)
            models[m]=model.eval().requires_grad_(False);actors[m]=model.actor;configs[m]=model.config
        targets={}
        if set(stages)&{"capacity","oracle"}:
            cache_root=t5.path_from_project(args.reference_cache)
            for entry in export["evaluation_source_batches"]:
                name=Path(entry["reference_path"].replace("\\","/")).name
                path=cache_root/name
                if t5.sha256(path)!=entry["reference_sha256"]:raise ValueError("MILP cache SHA changed")
                with np.load(path,allow_pickle=False) as archive:
                    settings=json.loads(str(archive["settings"].item()))
                    if settings["data_sha256"]!=entry["source_batch_sha256"]:raise ValueError("MILP cache input changed")
                    for record in evaluation:
                        if record["M"]!=entry["M"]:continue
                        position=int(record["id"].rsplit("_",1)[1])
                        x=archive["solution"][position].copy()
                        if (x.shape!=(record["N"],) or not np.all((x==0)|(x==1)) or
                            bool(archive["optimal"][position])!=record["reference_proven"] or
                            float(archive["obj"][position])!=record["reference"] or
                            np.any(record["inst"].weights.T@x>record["inst"].capacities+1e-7) or
                            float(record["inst"].profits@x)!=record["reference"]):
                            raise ValueError("Invalid or mismatched target solution")
                        targets[record["id"]]=torch.tensor(x,dtype=torch.bool)
            meta["target_counts"]={"proven_optimum":sum(r["reference_proven"] for r in evaluation),
                                   "MILP_incumbent":sum(not r["reference_proven"] for r in evaluation)}
        def load_target(record):return targets[record["id"]]
        tables=json.loads((t6root/"schedules.json").read_text(encoding="utf-8"))["tables"]["category/tau_cap"]
        scales=core.residual_scales(t6root/"traces.jsonl",tables,phase) if "capacity" in stages else {}
        if scales:t5.write_json(dest/"residual_scales.json",scales)
        t5.write_json(dest/"meta.json",meta)
        def sync():
            if device.type=="cuda":torch.cuda.synchronize(device)
        def open_stream(name):
            stream=(dest/name).open("x",encoding="utf-8");streams.append(stream);return stream
        def emit(stream,row):
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n");stream.flush()
        rows_file=open_stream("rows.jsonl")
        for row in references.values():emit(rows_file,row)
        def s2(record):return phase.ScheduleActor(tables[phase.key(record,"category")],Theta,configs[record["M"]],helper)
        def row_result(record,result,elapsed,lp_time):
            objective,_,_,sequence=verify(record["inst"],result,1e-7)
            return {**{k:v for k,v in record.items() if k!="inst"},"objective":objective,"gap_pct":100*(1-objective/record["reference"]),
                "selected_item_indices":sequence,"steps":len(sequence),"feasible":True,"maximal":True,
                "time_total_s":elapsed,"time_lp_s":lp_time,"reused_reference":False}
        # Timers patch only the in-memory LP solver; the original is always restored.
        class Timer:
            def __enter__(self):
                self.total,self.calls=0.,0;self.original=lp.solve_lp_relaxation
                def timed(*a,**kw):
                    sync();start=time.perf_counter()
                    try:return self.original(*a,**kw)
                    finally:sync();self.total+=time.perf_counter()-start;self.calls+=1
                lp.solve_lp_relaxation=timed;return self
            def __exit__(self,*exc):lp.solve_lp_relaxation=self.original
        if "policy" in stages:
            definitions=[("SL05","SL",.5,1),("SL08","SL",.8,1),("SL09","SL",.9,1),("LS08","LS",.8,1),
                         ("K2","interval",0.,2),("K8","interval",0.,8)]
            for record in records:
                for name,mode,threshold,interval in definitions:
                    model=models[record["M"]]
                    with Timer() as timer:
                        actor=core.ControlActor(actors[record["M"]],s2(record),phase,Theta,mode=mode,
                            threshold=threshold,interval=interval,lp_elapsed=lambda:timer.total).to(device).eval()
                        model.actor=actor
                        try:
                            sync();start=time.perf_counter();result=construct(batch(record["inst"],1,device),model,stochastic=False)
                            sync();elapsed=time.perf_counter()-start
                        finally:model.actor=actors[record["M"]]
                    row=row_result(record,result,elapsed,timer.total)
                    if timer.calls!=row["steps"]:raise ValueError("LP must still solve every decision")
                    if mode=="interval" and actor.updates!=(row["steps"]+interval-1)//interval:raise ValueError("Wrong controller interval")
                    row.update(arm=name,controller_calls=actor.updates,controller_inclusive_lp_s=actor.controller_s,
                        controller_excluding_lp_s=actor.controller_non_lp_s,lp_solve_calls=timer.calls,
                        schedule_calls=actor.schedule_updates,held_theta_steps=row["steps"]-actor.updates if mode=="interval" else 0,
                        switch_step=actor.switch_step,switch_feasible=actor.switch_feasible,
                        switch_remaining_steps=None if actor.switch_step is None else row["steps"]-actor.switch_step,
                        switch_remaining_after_action=None if actor.switch_step is None else row["steps"]-actor.switch_step-1,
                        threshold_reached=actor.switch_step is not None if mode!="interval" else None)
                    emit(rows_file,row);print(f"[policy] {name} {record['dataset']}/{record['id']}: {row['gap_pct']:.4f}%",flush=True)
        def features(state,m):return core.StateScores(state,configs[m],models[m].decoder,CandidatePolicy,prepare_state,cached_features)
        @torch.no_grad()
        def probes(record,state,frame,learned,center,local=True):
            draws=core.global_draws(configs[record["M"]],record["M"],meta["global_draws"],
                core.seeded(20260910,record["dataset"],record["id"],state,"global"))
            local_values=None
            if local:
                tau=float(phase.clocks(state)["tau_cap"][0])
                local_values=core.local_draws(center,core.scales_at(scales[phase.key(record,"category")],tau),configs[record["M"]],
                    meta["local_draws"],core.seeded(20260910,record["dataset"],record["id"],state,"local"))
            return draws,local_values
        if "capacity" in stages:
            states_file=open_stream("states.jsonl");replay_file=open_stream("replay.jsonl")
            selected_chu=[];counts=Counter()
            for r in chu:
                category=phase.key(r,"category")
                if counts[category]<2:selected_chu.append(r);counts[category]+=1
            for record in evaluation+selected_chu:
                env=Env(*batch(record["inst"],1,device));sequence=[]
                while bool(env.state.active[0]):
                    state=env.state;frame=features(state,record["M"])
                    with torch.no_grad():learned,_=actors[record["M"]](state)
                    learned=helper.theta_to_dict(learned)
                    tau=float(phase.clocks(state)["tau_cap"][0]);center=core.schedule_theta(tables[phase.key(record,"category")],tau,phase)
                    actual=models[record["M"]].decoder(state,Theta(**{n:v.to(device) for n,v in learned.items()})).choose(False)
                    if not torch.equal(actual,frame.choices(learned)):raise ValueError("Cached score path changed L action")
                    s2_choice=int(frame.choices(center)[0]);full,local=probes(record,state,frame,learned,center)
                    target=load_target(record) if record["dataset"]=="synthetic_N100" else None
                    for executor in ("E_full","E_min"):
                        anchors=torch.cat((frame.choices(learned,executor),frame.choices(center,executor)))
                        reachable=torch.unique(torch.cat((frame.choices(full,executor,args.theta_chunk),anchors)))
                        choices=frame.choices(local,executor,args.theta_chunk);local_reach=torch.unique(choices)
                        row={"dataset":record["dataset"],"id":record["id"],"M":record["M"],"N":record["N"],
                            "tightness":record["tightness"],"step":len(sequence),"tau_cap":tau,"executor":executor,
                            "feasible_count":len(frame.items),"reachable_full":len(reachable),"reachable_local":len(local_reach),
                            "rho_full":len(reachable)/len(frame.items),"rho_local":len(local_reach)/len(frame.items),
                            "L_choice":int(actual[0]),"S2_choice":s2_choice,"L_S2_different":int(actual[0])!=s2_choice,
                            "executor_L_choice":int(anchors[0]),"executor_S2_choice":int(anchors[1]),
                            "stability":float((choices==actual[0]).float().mean()),
                            "stability_projected_L":float((choices==anchors[0]).float().mean()),
                            "global_draws":meta["global_draws"],"local_draws":meta["local_draws"]}
                        if target is not None:
                            allowed=target.to(device)[frame.items]
                            row.update(reference_proven=record["reference_proven"],L_in_target=bool(target[int(actual[0])]),
                                sampled_target_intersection_empty=not bool(torch.isin(reachable,frame.items[allowed]).any()))
                        emit(states_file,row)
                    chosen=int(actual[0]);env.step(actual);sequence.append(chosen)
                expected_sequence=references[(record["dataset"],record["id"],"L")]["selected_item_indices"]
                if sequence!=expected_sequence:raise ValueError("Capacity source trajectory does not reproduce T5 L")
                emit(replay_file,{"dataset":record["dataset"],"id":record["id"],"steps":len(sequence),"sequence_matches_L":True})
                print(f"[capacity] {record['dataset']}/{record['id']}: {len(sequence)} states",flush=True)
        if "oracle" in stages:
            oracle_file=open_stream("oracle.jsonl")
            for record in evaluation:
                if args.oracle_targets=="proven" and not record["reference_proven"]:continue
                target=load_target(record)
                for executor in ("E_full","E_min"):
                    env=Env(*batch(record["inst"],1,device))
                    @torch.no_grad()
                    def provider(state,which):
                        frame=features(state,record["M"])
                        learned=helper.theta_to_dict(actors[record["M"]](state)[0])
                        tau=float(phase.clocks(state)["tau_cap"][0]);center=core.schedule_theta(tables[phase.key(record,"category")],tau,phase)
                        draws,_=probes(record,state,frame,learned,center,local=False)
                        reach=torch.unique(torch.cat((frame.choices(draws,which,args.theta_chunk),frame.choices(learned,which),frame.choices(center,which))))
                        return frame,reach,frame.ranking(center,which)
                    with Timer() as timer:
                        sync();start=time.perf_counter();result,audit=core.oracle_rollout(env,target,provider,executor)
                        sync();elapsed=time.perf_counter()-start
                    row=row_result(record,result,elapsed,timer.total)
                    row.update(executor=executor,arm="target_oracle",target_kind="proven_optimum" if record["reference_proven"] else "MILP_incumbent",**audit)
                    emit(oracle_file,row);print(f"[oracle] {executor} {record['id']}: {row['gap_pct']:.4f}%",flush=True)
        for stream in streams:stream.close()
        subprocess.run([sys.executable,"-B",str(PROJECT/"scripts/t7_report.py"),"--input",str(dest)],check=True)
        meta.update(status="complete",device=str(device),torch=str(torch.__version__),
            evaluation_instances=len(evaluation),chu_policy_instances=len(chu))
    except Exception as exc:
        meta.update(status="failed",error=repr(exc));raise
    finally:
        for stream in streams:stream.close()
        meta["legacy_after"]=t5.legacy_snapshot(root);meta["legacy_unchanged"]=meta["legacy_after"]==before
        if not meta["legacy_unchanged"]:meta["status"]="failed"
        t5.write_json(dest/"meta.json",meta)
        if not meta["legacy_unchanged"]:raise RuntimeError("Legacy project changed")
    return dest


if __name__=="__main__":main()
