"""T9-0 exact-LP/core measurements and T9-4 prefix diagnostics, never training."""
import json,sys,time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scipy.optimize import linprog,milp,Bounds,LinearConstraint
from mkp_artifacts import write_json,sha256_file
from mkp_v6_config import ModelConfig
from mkp_v6_lp import lp_relaxation
from mkp_cert_basis import dual_identity
from mkp_env import generate_instances,MKPInstances
from scripts.t9_common import parser,fresh,context,physical,tensors,outcome,targets,paired,summarize,sync
from scripts.t5_report import write_csv
from scripts.eval_t8 import load_arm,Timings


def exact_lp(state):
    p=state.profits[0].cpu().numpy();w=state.weights[0].cpu().numpy();c=state.remaining[0].cpu().numpy()
    feasible=state.feasible[0].cpu().numpy();x=np.zeros_like(p)
    if not feasible.any():return dict(value=0.,x=x,prices=np.zeros_like(c),fractional=0)
    r=linprog(-p[feasible],A_ub=w[feasible].T,b_ub=c,bounds=(0,1),method='highs')
    if not r.success:raise RuntimeError(f'Exact LP failed: {r.message}')
    x[feasible]=r.x
    return dict(value=float(-r.fun),x=x,prices=-r.ineqlin.marginals,fractional=int(((x>1e-8)&(x<1-1e-8)).sum()))


def identity_parts(state,selected,prices):
    x=torch.tensor(selected,device=state.profits.device,dtype=torch.bool).reshape(1,-1)
    y=torch.tensor(prices,device=state.profits.device,dtype=torch.float64).reshape(1,-1)
    dual,parts=dual_identity(state,x,y)
    return dict(exact_dual=float(dual[0]),priced_slack=float(parts[0,0]),taken_regret=float(parts[0,1]),skipped_regret=float(parts[0,2]))


def selected(row,n):
    x=np.zeros(n,dtype=bool);x[np.asarray(row['selected_item_indices'],dtype=int)]=True
    return x


def core_measure(info,p,w,c,exact,solutions,target,seconds):
    reduced=p-w@exact['prices'];ratio=np.abs(reduced)/np.maximum(p,1e-12)
    fractional=(exact['x']>1e-8)&(exact['x']<1-1e-8);order=np.argsort(ratio,kind='stable');rows=[]
    for label,k in (('M',info['M']),('2M',2*info['M']),('2M+10',2*info['M']+10),('0.1N',int(np.ceil(.1*len(p)))),('0.2N',int(np.ceil(.2*len(p))))):
        core=fractional.copy();core[order[:min(k,len(p))]]=True
        row={**info,'core_rule':label,'core_count':int(core.sum()),'core_fraction':float(core.mean())}
        for a,x in solutions.items():
            lb=float(p@x);fixed=np.abs(reduced)>exact['value']-lb
            row[a+'_certified_fixed_fraction']=float(fixed.mean())
            if target is not None:
                difference=x!=target;row[a+'_reference_difference_in_core']=float((difference&core).sum()/difference.sum()) if difference.any() else None
        if 'A' in solutions and 'G_port' in solutions:
            difference=solutions['A']!=solutions['G_port'];row['A_G_difference_in_core']=float((difference&core).sum()/difference.sum()) if difference.any() else None
        row['reference_certified_fixed_fraction']=float((np.abs(reduced)>exact['value']-info['reference']).mean())
        if info['dataset']=='synthetic_N100':
            # This restricted MILP is a diagnostic, never an inference solution.
            rounding=exact['x']>=.5;lo=np.where(core,0,rounding).astype(float);hi=np.where(core,1,rounding).astype(float)
            result=milp(-p,integrality=np.ones(len(p)),bounds=Bounds(lo,hi),constraints=LinearConstraint(w.T,-np.inf,c),
                        options={'time_limit':seconds,'mip_rel_gap':1e-9})
            row.update(core_MILP_status=int(result.status),core_MILP_proven=result.status==0)
            if result.x is not None:
                x=np.rint(result.x);value=float(p@x)
                if np.any(w.T@x>c+1e-7):raise ValueError('Infeasible restricted MILP')
                row.update(core_objective=value,core_reference_gap_pct=100*(1-value/info['reference']))
            else:row['core_error']=result.message
        rows.append(row)
    return rows


def numerical(records,rows,args,dest):
    config=ModelConfig();indices={(r['dataset'],r['arm'],str(r['id'])):r for r in rows}
    reference_x=targets(args.instances_dir);accuracy=[];integrality=[];identities=[];core=[];failures=[]
    for record in records:
        info,p,w,c=record;env=physical(record);state=env.state
        try:
            exact=exact_lp(state);mean=p.mean()
            integrality.append({**info,'LP_reference_gap_pct':100*(exact['value']/info['reference']-1),
                'LP_reference_gap_items':(exact['value']-info['reference'])/mean,'reference_is_optimum':info['reference_proven'],
                **{a+'_gap_pct':indices[(info['dataset'],a,str(info['id']))]['gap_pct'] for a in ('D_port','G_port','A','B') if (info['dataset'],a,str(info['id'])) in indices}})
            solutions={a:selected(indices[(info['dataset'],a,str(info['id']))],len(p)) for a in ('D_port','G_port','A','B') if (info['dataset'],a,str(info['id'])) in indices}
            target=reference_x.get(str(info['id'])) if info['dataset']=='synthetic_N100' else None
            if target is not None:
                if abs(float(p@target)-info['reference'])>1e-5 or np.any(w.T@target>c+1e-7):raise ValueError('Cached target differs from reference')
                solutions['reference']=target
            for arm,x in solutions.items():identities.append({**info,'arm':arm,**identity_parts(state,x,exact['prices']),
                'objective':float(p@x),'gap_to_exact_LP':exact['value']-float(p@x)})
            if 'core' in args.tasks:
                core.extend(core_measure(info,p,w,c,exact,solutions,target,.2 if args.limit else args.milp_seconds))
            trajectory=indices[(info['dataset'],'D_port' if info['M']==30 else 'G_port',str(info['id']))]
            seen=set()
            for step,action in enumerate(trajectory['selected_item_indices']):
                tau=float(1-(state.remaining/state.capacities).mean());decile=min(9,int(tau*10))
                if decile not in seen:
                    seen.add(decile);lp=lp_relaxation(state,config);actual=exact_lp(state)
                    primal=lp.primal[0].double().numpy();usage=w.T@primal
                    accuracy.append({**info,'step':step,'decile':decile,'tau_cap':tau,'exact_LP':actual['value'],
                        'bound_minus_exact_items':(float(lp.bound[0])-actual['value'])/float(lp.scale[0]),
                        'bound_minus_exact_relative':(float(lp.bound[0])-actual['value'])/max(1e-12,actual['value']),
                        'smoothed_fractional':int(((primal>.01)&(primal<.99)).sum()),'exact_fractional':actual['fractional'],
                        'primal_max_violation_raw':float(np.maximum(usage-state.remaining[0].numpy(),0).max()),
                        'certificate_gap_items':float(lp.certificate_gap[0]/lp.scale[0])})
                env.step(torch.tensor([action]))
            if not np.array_equal(state.selected[0].numpy(),selected(trajectory,len(p))):raise ValueError('Trajectory replay differs')
        except Exception as exc:failures.append({**info,'error':repr(exc)})
        for name,values in (('accuracy',accuracy),('integrality',integrality),('identity',identities),('core',core),('failures',failures)):
            write_csv(dest/f'{name}.csv',values)
        print(f"T9 numerical {info['dataset']}/{info['id']}",flush=True)
    groups=defaultdict(list)
    for r in accuracy:
        for group in (('all','all','all'),(r['M'],r['N'],r['decile'])):groups[group].append(r)
    summaries=[dict(M=m,N=n,decile=d,count=len(v),p95_bound_error_items=float(np.quantile([r['bound_minus_exact_items'] for r in v],.95)),
        mean_bound_error_items=float(np.mean([r['bound_minus_exact_items'] for r in v]))) for (m,n,d),v in groups.items()]
    write_csv(dest/'accuracy_summary.csv',summaries)
    # Paired three-term gap differences (all are in the same root dual units).
    index={(r['dataset'],r['arm'],str(r['id'])):r for r in identities};diff=[]
    for r in identities:
        for control in ('G_port','reference'):
            base=index.get((r['dataset'],control,str(r['id'])))
            if base is None or r['arm']==control:continue
            diff.append({**{k:r[k] for k in ('dataset','id','M','N','tightness','arm')},'control':control,
                **{k+'_delta':r[k]-base[k] for k in ('priced_slack','taken_regret','skipped_regret','gap_to_exact_LP')}})
    write_csv(dest/'identity_paired.csv',diff)
    return {'lp_p95_error_items':summaries[0]['p95_bound_error_items'] if summaries else None,
        'LP_gate_passed':bool(summaries) and not failures and summaries[0]['p95_bound_error_items']<=.01,'failures':len(failures)}


def fitted_groups(path):
    sums={};counts=defaultdict(int)
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line)
            if r['arm'] not in ('A','B') or r['dataset']!='synthetic_N100':continue
            key=(r['arm'],r['M']);theta=r['theta'];matrix=np.asarray(theta['resource_interactions']);m=r['M']
            diagonal=np.trace(matrix)/m;off=(matrix.sum()-np.trace(matrix))/(m*(m-1))
            values=np.asarray([diagonal,off,theta['bottleneck_weight'],theta['reserve_weight']])
            sums[key]=sums.get(key,np.zeros(4))+values;counts[key]+=1
    return {key:values/counts[key] for key,values in sums.items()}


@torch.no_grad()
def fit_m30(args):
    """Fresh synthetic traces provide M30 constants without fitting on Chu."""
    batch=generate_instances(2 if args.limit else 32,100,30,generator=torch.Generator().manual_seed(2026111402))
    result={};metadata=[]
    from mkp_artifacts import instance_sha
    for arm in ('A','B'):
        model,_=load_arm(arm,args,torch.device(args.device));total=np.zeros(4);count=0
        for i in range(batch.B):
            inst=MKPInstances(batch.p[i:i+1],batch.r[i:i+1],batch.c[i:i+1]).to(args.device)
            out=model(None,inst,decode='greedy',return_trace=True)
            for trace in out['trace']:
                theta=trace['theta'];matrix=np.asarray(theta['resource_interactions'][0]);diagonal=np.trace(matrix)
                total+=np.asarray([diagonal/30,(matrix.sum()-diagonal)/(30*29),theta['bottleneck_weight'][0],theta['reserve_weight'][0]])
                count+=1
        result[arm,30]=total/max(count,1);metadata.append(dict(arm=arm,M=30,seed=2026111402,instances=batch.B,decisions=count,data_sha256=instance_sha(batch)))
        del model
    return result,metadata


def policy_diagnostics(records,baselines,args,dest,root):
    fits={};rows=[]
    means=json.loads((Path(args.instances_dir)/'attribution/theta_means.json').read_text())
    args.a_checkpoint='checkpoints/t8_a_jointM_s1234/best.pt';args.b_checkpoint='checkpoints/t8_b_jointM_s1234/best.pt';args.tag='best'
    variants=(['P1','F-mean','F-zero'] if 'freeze' in args.tasks else [])+([f'prefix{d}' for d in (.05,.1,.2)] if 'prefix' in args.tasks else [])
    for variant in variants:
        if variant=='F-mean':
            fits=fitted_groups(root/'traces.jsonl')
            added,fitting=fit_m30(args);fits.update(added)
            write_json(dest/'fitted_constants.json',{'trace_sha256':sha256_file(root/'traces.jsonl'),
                'scope':'own synthetic traces only; no Chu fitting','M30_fitting':fitting,
                'per_arm_M':{f'{a}_M{m}':values.tolist() for (a,m),values in fits.items()}})
        for arm in ('A','B','G_port'):
            if arm=='G_port' and not variant.startswith('prefix'):continue
            model,_=load_arm(arm,args,torch.device(args.device))
            for record in records:
                info=record[0];m=info['M']
                if arm=='G_port' and str(m) not in means['G']:continue
                delta=float(variant.removeprefix('prefix')) if variant.startswith('prefix') else None;patch=None
                if variant=='P1':patch={'resource_prices':1.}
                elif variant=='F-zero':patch={'resource_prices':1.,'resource_interactions':0.,'bottleneck_weight':0.,'reserve_weight':0.}
                elif variant=='F-mean':
                    diag,off,bneck,reserve=fits[(arm,m)];matrix=np.full((m,m),off);np.fill_diagonal(matrix,diag)
                    patch={'resource_prices':1.,'resource_interactions':matrix,'bottleneck_weight':bneck,'reserve_weight':reserve}
                inst=tensors(record,args.device);sync(args.device);tick=time.perf_counter()
                with torch.no_grad():result=model(None,inst,decode='greedy',theta_override=means['G'][str(m)] if arm=='G_port' else None,theta_patch=patch,prefix_delta=delta)
                sync(args.device)
                rows.append(outcome(info,arm+'_'+variant,result,time.perf_counter()-tick,checkpoint_arm=arm,
                    controller_calls=result.get('controller_calls',result['steps']),lp_solves=result.get('lp_solves',result['steps'])))
                write_csv(dest/'policy_per_instance.csv',rows)
                print(f'T9 policy {arm} {variant} {info["dataset"]}/{info["id"]}',flush=True)
            del model
        if variant=='P1' and not args.limit:
            idx={(r['dataset'],r['arm'],str(r['id'])):r for r in baselines};contradiction=False
            for arm in ('A','B'):
                matched=[(idx[('chu','G_port',str(r['id']))]['gap_pct'],r['gap_pct'],idx[('chu',arm,str(r['id']))]['gap_pct'])
                    for r in rows if r['arm']==arm+'_P1' and r['dataset']=='chu' and ('chu','G_port',str(r['id'])) in idx]
                g,a,b=np.asarray(matched).mean(0)
                if g-b<=0 or (g-a)/(g-b)<.9:contradiction=True
            if contradiction:
                print('STOP: P1 contradicts the price-channel premise; remaining freeze/prefix work is not run.',flush=True)
                break
    comparisons=[(r['arm'],r['checkpoint_arm']) for r in rows];raw,stats=paired(rows+baselines,sorted(set(comparisons)))
    write_csv(dest/'policy_paired.csv',raw);write_csv(dest/'policy_paired_summary.csv',stats);write_csv(dest/'policy_summary.csv',summarize(rows))
    index={(r['dataset'],r['arm'],str(r['id'])):r for r in rows+baselines};groups=defaultdict(list)
    for r in rows:
        g=index.get((r['dataset'],'G_port',str(r['id'])));base=index.get((r['dataset'],r['checkpoint_arm'],str(r['id'])))
        if g and base:groups[(r['dataset'],r['arm'])].append((g['gap_pct'],r['gap_pct'],base['gap_pct']))
    retention=[]
    for (dataset,arm),v in groups.items():
        g,a,b=np.asarray(v).mean(0);retention.append(dict(dataset=dataset,arm=arm,count=len(v),retention=(g-a)/(g-b) if abs(g-b)>1e-12 else None))
    write_csv(dest/'policy_retention.csv',retention)
    gate={}
    for arm in ('A','B'):
        for name,threshold in (('P1',.9),('F-mean',.9)):
            r=next((r for r in retention if r['dataset']=='chu' and r['arm']==arm+'_'+name),None)
            if r:gate[arm+'_'+name+'_retention']=r['retention']
    return gate


def main():
    p=parser(__doc__,'results/t9_measure');p.add_argument('--tasks',nargs='+',choices=('numerics','core','freeze','prefix'),default=['numerics','core'])
    p.add_argument('--milp_seconds',type=float,default=10.);a=p.parse_args();torch.set_num_threads(2)
    records,rows,root=context(a);dest,meta=fresh(a.output,a)
    began=time.perf_counter()
    meta.update(reference_rows_sha256=sha256_file(root/'rows.jsonl'),F_mean_scope='checkpoint own synthetic traces per arm/M; fresh M30 synthetic set, seed2026111402. No Chu fitting.')
    try:
        gates={}
        if set(a.tasks)&{'numerics','core'}:gates.update(numerical(records,rows,a,dest))
        if set(a.tasks)&{'freeze','prefix'}:gates.update(policy_diagnostics(records,rows,a,dest,root))
        write_json(dest/'gates.json',{'smoke_only':bool(a.limit),'decision':'pending_Claude','quantities':gates,
            'stop_if_price_contradiction':any(gates.get(arm+'_P1_retention',1)<.9 for arm in ('A','B'))})
        (dest/'report.md').write_text('# T9 measurements\n\n'+json.dumps(gates,indent=2)+'\n\nTables: accuracy(_summary), integrality, identity(_paired), core, policy_(summary/paired_summary/retention). Missing tasks have no measured table. Failed/time-limited solves remain labeled. Core MILP incumbents are not certified optima. Chu270 headline, matched G/L subset only. No automatic research decision.\n',encoding='utf-8')
        from scripts.t9_report import measurement_report
        measurement_report(dest)
        meta['status']='complete'
    except Exception as exc:meta.update(status='failed',error=repr(exc));raise
    finally:meta['elapsed_seconds']=time.perf_counter()-began;write_json(dest/'meta.json',meta)


if __name__=='__main__':main()
