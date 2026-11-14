"""Reports for measured T9 artifacts; never train or choose a research decision."""
from collections import defaultdict
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.t5_report import write_csv
from scripts.t9_common import summarize,paired,instance_key
from mkp_artifacts import sha256_file


def theta_tables(traces):
    observations=defaultdict(list);changes=defaultdict(list);previous={}
    for r in traces:
        if r['dataset']!='chu':continue
        values={**dict(zip(('pi_sur','pi_reg','pi_blk','pi_fill','pi_pair'),r['theta']['weights'])), 'temperature':r['theta']['temperature']}
        key=(r['arm'],str(r['id']));category=(r['M'],r['N'],r['tightness'])
        for field,value in values.items():
            observations[(r['arm'],field)].append((category,str(r['id']),value))
            if key in previous:
                changes[(r['arm'],field,'late' if r['tau_cap']>=.8 else 'early')].append(abs(value-previous[key][field]))
        previous[key]=values
    variance=[]
    for (arm,field),data in observations.items():
        categories=defaultdict(list);instances=defaultdict(list);grand=np.mean([v for _,_,v in data])
        for cat,iid,value in data:categories[cat].append(value);instances[(cat,iid)].append(value)
        total=sum((v-grand)**2 for _,_,v in data)
        between=sum(len(v)*(np.mean(v)-grand)**2 for v in categories.values())
        middle=sum(len(v)*(np.mean(v)-np.mean(categories[cat]))**2 for (cat,iid),v in instances.items())
        within=sum(sum((x-np.mean(v))**2 for x in v) for v in instances.values())
        if not np.isclose(total,between+middle+within,rtol=1e-8,atol=1e-12):raise ValueError('Theta variance decomposition failed')
        variance.append(dict(arm=arm,field=field,count=len(data),total_variance=total/len(data),between_category_fraction=between/total if total else None,
            between_instance_fraction=middle/total if total else None,within_trajectory_fraction=within/total if total else None))
    return variance,[dict(arm=a,field=f,phase=p,transitions=len(v),mean_absolute_delta=float(np.mean(v))) for (a,f,p),v in changes.items()]


def evaluation_report(dest,rows,traces,baselines,records,history):
    canonical={instance_key(r[0]):r[0] for r in records};arms={r['arm'] for r in rows};seen=set()
    for arm in arms:
        actual={instance_key(r) for r in rows if r['arm']==arm}
        if actual!=set(canonical):raise ValueError(f'Incomplete Chu270/synthetic128 arm: {arm}')
    for row in rows:
        key=(row['arm'],*instance_key(row))
        if key in seen:raise ValueError('Duplicate evaluated row')
        seen.add(key);info=canonical[instance_key(row)]
        if row['reference']!=info['reference'] or row['data_sha256']!=info['data_sha256']:raise ValueError('Evaluation data mismatch')
    quoted=[{**r,'arm':'L_v6'} for r in history if r['arm']=='L' and instance_key(r) in canonical]
    outcomes=rows+baselines+quoted
    write_csv(dest/'per_instance.csv',outcomes);write_csv(dest/'summary.csv',summarize(outcomes))
    comparisons=[('T9-A',c) for c in ('K*','A','L_v6') if 'T9-A' in arms]
    raw,stats=paired(outcomes,comparisons);write_csv(dest/'paired_per_instance.csv',raw);write_csv(dest/'paired_summary.csv',stats)
    expected={(r['arm'],str(r['id'])):r['steps'] for r in rows if r['dataset']=='chu' and r['arm']=='T9-A'}
    counts=defaultdict(int)
    for r in traces:
        if r['dataset']=='chu' and r['arm']=='T9-A':counts[(r['arm'],str(r['id']))]+=1
    if dict(counts)!=expected:raise ValueError('Incomplete learned theta traces')
    variance,changes=theta_tables(traces);write_csv(dest/'theta_variance.csv',variance);write_csv(dest/'theta_step_changes.csv',changes)
    (dest/'report.md').write_text('# T9 single-greedy evaluation\n\nChu270 for each new arm; historical G/L remain their matched subset. See summary.csv (mean/p90/worst), paired_summary.csv (actual paired counts, two-sided unadjusted Wilcoxon), theta_variance.csv, theta_step_changes.csv, gap_identity.csv. Missing M30 historical controls are never imputed. Single seed, no across-seed significance claim.\n',encoding='utf-8')


def measurement_report(root):
    def read(name):
        path=root/(name+'.csv')
        if not path.exists() or path.stat().st_size==0:return []
        with path.open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))
    def table(name,fields):
        rows=read(name)
        return ['## '+name,'','| '+' | '.join(fields)+' |','| '+' | '.join('---' for _ in fields)+' |']+[
            '| '+' | '.join(str(r.get(f,'')) for f in fields)+' |' for r in rows]+[''] if rows else ['## '+name,'','Not measured / not requested.','']
    lines=['# T9 measurement tables','','Scope and limitations are in meta.json and gates.json; these tables do not make a research decision.','']
    lines+=table('accuracy_summary',['M','N','decile','count','p95_bound_error_items'])
    # Class-level integrality and identity summaries, retaining every raw row.
    for source,fields in (('integrality',('LP_reference_gap_pct','LP_reference_gap_items')),('identity',('priced_slack','taken_regret','skipped_regret','gap_to_exact_LP'))):
        groups=defaultdict(list)
        for r in read(source):groups[(r['dataset'],r.get('arm','all'),r['M'],r['N'])].append(r)
        aggregate=[dict(dataset=d,arm=a,M=m,N=n,count=len(v),**{f:float(np.mean([float(r[f]) for r in v])) for f in fields}) for (d,a,m,n),v in groups.items()]
        write_csv(root/(source+'_summary.csv'),aggregate);lines+=table(source+'_summary',['dataset','arm','M','N','count',*fields])
    lines+=table('core',['dataset','id','core_rule','core_count','A_G_difference_in_core','core_MILP_status','core_MILP_proven','core_reference_gap_pct'])
    lines+=table('policy_paired_summary',['dataset','arm','control','M','N','count','mean_gap_delta_pp','wins','ties','losses','wilcoxon_two_sided_p'])
    lines+=table('policy_retention',['dataset','arm','count','retention'])
    (root/'tables.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path)
    p.add_argument('--run',type=Path);p.add_argument('--output',type=Path,default=Path('results/t9_training_curves'));a=p.parse_args()
    if a.run:plot_training(a.run,a.output)
    elif a.input:measurement_report(a.input)
    else:p.error('provide --input measurement directory or --run training directory')


def plot_training(run,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mkp_cert_exec import STATISTICS
    with (run/'train_log.csv').open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
    output.mkdir(parents=True,exist_ok=True);plotted=[]
    for field in ('validation_gap','R_mean','frac_better','entropy',*STATISTICS):
        points=[(float(r['update']),float(r[field])) for r in rows if r.get(field) not in ('',None) and np.isfinite(float(r[field]))]
        if not points:continue
        figure,ax=plt.subplots(figsize=(6,3.5));ax.plot(*np.asarray(points).T);ax.set(xlabel='Update',ylabel=field,title=run.name);ax.grid(alpha=.2)
        figure.tight_layout();figure.savefig(output/(field+'.png'),dpi=160);plt.close(figure);plotted.append(field)
    (output/'manifest.json').write_text(json.dumps(dict(run=str(run),train_log_sha256=sha256_file(run/'train_log.csv'),
        plotted=plotted,missing=[f for f in STATISTICS if f not in plotted]),indent=2),encoding='utf-8')


if __name__=='__main__':main()
