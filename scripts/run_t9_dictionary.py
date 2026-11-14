"""T9-2 synthetic-only calibration, fixed Chu270 arms and sampled oracle gates."""
import itertools,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mkp_artifacts import write_json,sha256_file,instance_sha
from mkp_config import resolve_path
from mkp_env import generate_instances,MKPInstances
from mkp_reference import milp_reference
from mkp_Model import build_model
from mkp_v6_config import ModelConfig
from scripts.profile_t9 import cert_config
from scripts.t9_common import parser,fresh,context,tensors,physical,outcome,targets,summarize,sync
from scripts.t9_interface import oracle,EXTENSIONS
from mkp_cert_exec import CONSTANTS
from scripts.t5_report import write_csv


def grid(denominator):
    def recurse(total,length):
        if length==1:yield (total,);return
        for i in range(total+1):
            for tail in recurse(total-i,length-1):yield (i,)+tail
    return [np.asarray(v,dtype=float)/denominator for v in recurse(denominator,5)]


def activated(pi):
    value=torch.as_tensor(pi,dtype=torch.float32).clamp_min(1e-8).log().softmax(-1)
    return value.tolist()


@torch.no_grad()
def calibration(args,dest,device):
    cfg=cert_config();model=build_model(cfg).to(device).eval();env=None
    generation=torch.Generator().manual_seed(2026111401);data=[];reference_meta=[]
    count=2 if args.limit else 32
    for m in (5,10):
        batch=generate_instances(count,100,m,generator=generation)
        refs=milp_reference(batch,.2 if args.limit else 10.,1e-9,'cache')
        reference_meta.append(dict(M=m,sha256=refs['sha256'],proven=int(refs['optimal'].sum()),count=count))
        for i in range(count):
            data.append((i%2,MKPInstances(batch.p[i:i+1],batch.r[i:i+1],batch.c[i:i+1]).to(device),float(refs['obj'][i])))
    coarse=grid(4);assert len(coarse)==70
    if args.limit:coarse=[np.asarray(v)/sum(v) for v in list(CONSTANTS.values())[:2]]+[np.full(5,.2)]
    cache={};rows=[]
    def score(pi,fold=None):
        key=tuple(activated(pi));values=[]
        for index,(which,inst,reference) in enumerate(data):
            if fold is not None and which!=fold:continue
            item=(key,index)
            if item not in cache:
                model.cert_engine.cfg.model.cert_weights=list(key)
                out=model(None,inst,decode='greedy',theta_override='fixed');cache[item]=float(out['profit'][0])/reference
            values.append(cache[item])
        return float(np.mean(values))
    crossfit=[];defaults=[]
    for train_fold in (0,1):
        winner=max(coarse,key=lambda pi:score(pi,train_fold))
        refine=[pi for pi in grid(10) if np.max(np.abs(pi-winner))<=.1000001]
        if args.limit:refine=[winner]
        winner=max(refine+[winner],key=lambda pi:score(pi,train_fold));defaults.append(winner)
        crossfit.append(dict(train_fold=train_fold,default_pi=activated(winner),held_out_reward=score(winner,1-train_fold)))
    # Final deployable constant is fit on all calibration data, never validation/Chu.
    winner=max(coarse,key=lambda pi:score(pi))
    refine=[pi for pi in grid(10) if np.max(np.abs(pi-winner))<=.1000001] if not args.limit else [winner]
    winner=max(refine+[winner],key=lambda pi:score(pi))
    model.cert_engine.cfg.model.cert_weights=activated(winner)
    temperatures=[]
    for beta in (1,2,4,8,16,32,64):
        cfg.model.cert_temperature=1/beta
        # Fixed overrides use configured temperature but preserve greedy scores.
        model.cert_engine.cfg.model.cert_temperature=1/beta
        rewards=[];means=[];entropies=[]
        for _,inst,reference in data:
            torch.manual_seed(1234)
            out=model(None,inst,num_samples=16,decode='sample',theta_override='fixed')
            rewards.append(float(out['profit'].max())/reference);means.append(float(out['profit'].mean())/reference);entropies.append(float(out['entropy'].mean()))
        temperatures.append(dict(beta=beta,temperature=1/beta,best16_reward=float(np.mean(rewards)),sample_mean_reward=float(np.mean(means)),entropy=float(np.mean(entropies))))
    best=max(temperatures,key=lambda r:(r['best16_reward'],r['beta']));tau=best['temperature']
    calibration_meta=dict(schema='t9_calibration_v1',smoke_only=bool(args.limit),seed=2026111401,per_M=count,
        calibration_data_sha256=[instance_sha(inst) for _,inst,_ in data],references=reference_meta,
        coarse_points=70,executed_coarse_points=len(coarse),refinement_step=.1,
        crossfit=crossfit,weights=activated(winner),temperature=tau,temperature_min=tau/4,temperature_max=tau*4,
        selection='calibration-only mean profit/MILP reference; two-fold held-out estimates; final refit all64; tau synthetic best16 gap as calibrate_beta, ties larger beta',
        sampled_calibration_episodes=len(data)*16*len(temperatures),inference='still single greedy; sampling here calibrates training sharpness only')
    write_json(dest/'calibration.json',calibration_meta);write_csv(dest/'temperature.csv',temperatures)
    return calibration_meta


def fixed(records,baselines,args,dest,calibration_,device):
    cfg=cert_config();cfg.model.cert_weights=calibration_['weights'];cfg.model.cert_temperature=calibration_['temperature']
    cfg.model.cert_temperature_min=calibration_['temperature_min'];cfg.model.cert_temperature_max=calibration_['temperature_max']
    model=build_model(cfg).to(device).eval();rows=[]
    v6cfg=__import__('scripts.run_t8',fromlist=['configuration']).configuration('A')
    v6cfg.model.v6_frozen_groups=['resource_prices','resource_interactions','bottleneck_weight','reserve_weight']
    compact=build_model(v6cfg).to(device).eval()
    means=json.loads((Path(args.instances_dir)/'attribution/theta_means.json').read_text())
    for record in records:
        info=record[0];inst=tensors(record,device)
        for arm,pi in list(CONSTANTS.items())+[('K*',calibration_['weights'])]:
            sync(device);tick=time.perf_counter()
            with torch.no_grad():result=model(None,inst,decode='greedy',theta_override=arm if arm in CONSTANTS else 'fixed')
            sync(device);rows.append(outcome(info,arm,result,time.perf_counter()-tick))
        if str(info['M']) in means['G']:
            patch={'resource_prices':1.,'resource_interactions':0.,'bottleneck_weight':0.,'reserve_weight':0.}
            with torch.no_grad():result=compact(None,inst,decode='greedy',theta_override=means['G'][str(info['M'])],theta_patch=patch)
            rows.append(outcome(info,'v6c-G',result))
        write_csv(dest/'fixed_per_instance.csv',rows)
        print(f'T9 fixed {info["dataset"]}/{info["id"]}',flush=True)
    write_csv(dest/'fixed_summary.csv',summarize(rows+baselines))
    return rows


def oracle_gate(records,args,dest,calibration_,device):
    target=targets(args.instances_dir);means=json.loads((Path(args.instances_dir)/'attribution/theta_means.json').read_text());cfg=ModelConfig()
    families=['E_full','E_min','v6c','cert']+[f'{a}+rand{k}' for a in ('cert','E_min') for k in (2,8,32)]
    rows=[];pointer=[];summary=[];lookup={}
    draws=args.draws # reduced only when explicitly requested; --limit is not a score budget
    def evaluate(family):
        for record in records:
            info=record[0]
            if info['dataset']!='synthetic_N100':continue
            center=means['G'][str(info['M'])] if family.startswith(('E_full','E_min','v6c')) else calibration_['weights']
            env=physical(record,device);result,diagnostic,probes=oracle(env,target[str(info['id'])],cfg,family,center,draws,str(info['id']),calibration_['temperature'])
            row=outcome(info,family,result,first_empty=diagnostic['first_empty'],target_kind='proven' if info['reference_proven'] else 'incumbent')
            rows.append(row)
            pointer.extend({**{k:info[k] for k in ('dataset','id','M','N','tightness','reference_proven')},'arm':family,**r} for r in probes)
            write_csv(dest/'oracle_per_instance.csv',rows);write_csv(dest/'pointer.csv',pointer)
            print(f'T9 oracle {family}/{info["id"]}: {row["gap_pct"]:.4f}%',flush=True)
        for proven in (True,False):
            chosen=[r for r in rows if r['arm']==family and r['reference_proven']==proven]
            if chosen:
                kind='proven' if proven else 'incumbent';value=float(np.mean([r['gap_pct'] for r in chosen]))
                summary.append(dict(family=family,target_kind=kind,count=len(chosen),mean_gap_pct=value));lookup[family,kind]=value
        write_csv(dest/'oracle_summary.csv',summary)
    for family in families:evaluate(family)
    kinds=[k for k in ('proven','incumbent') if ('cert',k) in lookup]
    conditions=[lookup['cert',kind]<=lookup['E_full',kind]+.05 for kind in kinds]
    controls=[];retained=[];current='cert'
    # Extensions are conditional, in the specified order. Rejected terms are
    # removed before trying the next, rather than accumulating pointing power.
    if not all(conditions):
        for extension in EXTENSIONS:
            candidate=retained+[extension]
            dimensions=6+sum(2 if name in ('scar','comp3') else 1 for name in candidate)
            family='cert+'+'+'.join(candidate)
            if dimensions>10:
                controls.append(dict(family=family,accepted=False,reason='interface exceeds ten numbers'));continue
            evaluate(family);accepted=True;positive=False
            for kind in kinds:
                random_gain=max((lookup['cert',kind]-lookup[f'cert+rand{k}',kind])/k for k in (2,8,32))
                gain=lookup[current,kind]-lookup[family,kind]
                keep=gain>=2*max(0.,random_gain);positive=positive or gain>0;accepted=accepted and keep
                controls.append(dict(family=family,target_kind=kind,gain_pp=gain,random_gain_per_basis_pp=random_gain,accepted=keep,dimensions=dimensions))
            if accepted and positive:
                retained=candidate;current=family
            for row in controls:
                if row['family']==family:row['kept_overall']=bool(accepted and positive)
            if accepted and positive and all(lookup[current,k]<=lookup['E_full',k]+.05 for k in kinds):break
    write_csv(dest/'oracle_summary.csv',summary)
    write_csv(dest/'random_control.csv',controls)
    write_json(dest/'extension_decision.json',dict(probe_selected=current,retained=retained,
        decision='pending_Claude',training='Only base cert is wired for training; selected extensions require reviewed head/feature integration before training'))
    return bool(conditions) and all(conditions)


def main():
    p=parser(__doc__,'results/t9_dictionary');p.add_argument('--stages',nargs='+',choices=('calibration','fixed','oracle'),default=['calibration','fixed','oracle'])
    p.add_argument('--calibration',help='reuse calibration.json');p.add_argument('--measure',default='results/t9_measure')
    p.add_argument('--draws',type=int,default=2048);a=p.parse_args();torch.set_num_threads(2);device=torch.device(a.device)
    records,baselines,root=context(a)
    if not a.limit:
        gate=json.loads((Path(a.measure)/'gates.json').read_text())
        if gate['smoke_only'] or not gate['quantities'].get('LP_gate_passed') or gate.get('stop_if_price_contradiction'):
            raise ValueError('T9-0 gate not passed; report solver/price contradiction before dictionary runs')
        for arm in ('A','B'):
            if arm+'_P1_retention' not in gate['quantities']:raise ValueError('Complete T9-0 freeze measurement is required')
    dest,meta=fresh(a.output,a);began=time.perf_counter()
    try:
        c=calibration(a,dest,device) if 'calibration' in a.stages else json.loads(Path(a.calibration).read_text())
        if not a.limit and c['smoke_only']:raise ValueError('Cannot use smoke calibration for formal fixed/oracle runs')
        if 'calibration' not in a.stages:
            write_json(dest/'calibration.json',c)
            meta['reused_calibration_sha256']=sha256_file(a.calibration)
        fixed_rows=fixed(records,baselines,a,dest,c,device) if 'fixed' in a.stages else []
        ceiling_ok=oracle_gate(records,a,dest,c,device) if 'oracle' in a.stages else None
        index={(r['dataset'],r['arm'],str(r['id'])):r for r in baselines}
        paired=[(r,index[(r['dataset'],'G_port',str(r['id']))]) for r in fixed_rows if r['dataset']=='chu' and r['arm']=='K*' and (r['dataset'],'G_port',str(r['id'])) in index]
        constant_delta=float(np.mean([a['gap_pct']-b['gap_pct'] for a,b in paired])) if paired else None
        eligible=ceiling_ok is True and constant_delta is not None and constant_delta<=.1 and not a.limit and a.draws>=2048
        write_json(dest/'gate.json',dict(schema='t9_dictionary_gate_v1',decision='pending_Claude',smoke_only=bool(a.limit),
            base_cert_eligible=eligible,ceiling_condition=ceiling_ok,Kstar_minus_G_pp=constant_delta,Kstar_G_pairs=len(paired),
            training='cert only per user instruction; no v6/v6c retraining',calibration=c,
            unresolved='extensions are probe-only until a full gate and an explicit selected extension; no automatic promotion'))
        (dest/'report.md').write_text('# T9 dictionary gate\n\nSee calibration.json (synthetic only), fixed_summary.csv (Chu270), oracle_summary.csv (proven/incumbent separate), random_control.csv, pointer.csv. Oracle scores probe theta, not sampled item trajectories. Research decision pending Claude. Extension controls are diagnostics; extensions are not automatically put in a trainable controller.\n',encoding='utf-8')
        meta.update(status='complete',elapsed_seconds=time.perf_counter()-began)
    except Exception as exc:meta.update(status='failed',error=repr(exc));raise
    finally:write_json(dest/'meta.json',meta)


if __name__=='__main__':main()
