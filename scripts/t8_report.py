"""T8 measured reports and training curves; no inference or training."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.t5_report import paired_statistics, variance_table, write_csv
from mkp_artifacts import sha256_file
from scripts.t8_data import instance_key, validate_rows


def strata(row):
    return [('all', 'all', 'all'), (row['M'], row['N'], 'all'), (row['M'], row['N'], row['tightness'])]


def report(dest, rows, traces, historical, smoke, records, g_dims):
    validate_rows(rows, records, g_dims)
    historical_index = {(r['dataset'], r['arm'], str(r['id'])): r for r in historical}
    arms = set(r['arm'] for r in rows)
    # Add quoted L rows only after exact instance/reference matching.
    ids = {(r['dataset'], str(r['id'])) for r in rows}
    quoted = [{**r, 'arm': 'L_v6', 'quoted': True} for r in historical if r['arm'] == 'L' and (r['dataset'], str(r['id'])) in ids]
    outcomes = rows + quoted
    groups = defaultdict(list)
    for r in outcomes:
        for m, n, tightness in strata(r):
            groups[(r['dataset'], r['arm'], m, n, tightness)].append(r)
    summary = []
    for (dataset, arm, m, n, tightness), values in groups.items():
        gaps = np.asarray([r['gap_pct'] for r in values])
        summary.append(dict(dataset=dataset, arm=arm, M=m, N=n, tightness=tightness, count=len(values),
            M_values=','.join(str(v) for v in sorted({r['M'] for r in values})),
            mean_gap_pct=float(gaps.mean()), p90_gap_pct=float(np.quantile(gaps, .9)), worst_gap_pct=float(gaps.max()),
            quoted=arm == 'L_v6', **{f'mean_{field}': float(np.mean([r[field] for r in values])) if arm != 'L_v6' else ''
                for field in ('time_lp_s', 'time_controller_s', 'time_total_s')}))
    all_index = {(r['dataset'], r['arm'], str(r['id'])): r for r in outcomes}
    pairs, paired, pair_coverage = [], defaultdict(list), []
    comparisons = [(a, c) for a in ('A', 'B') if a in arms for c in ('D_port', 'G_port', 'L_v6')]
    if {'A', 'B'} <= arms:
        comparisons.append(('A', 'B'))
    for arm, control in comparisons:
        for dataset in ('chu', 'synthetic_N100'):
            arm_ids = {instance_key(r) for r in rows if r['arm']==arm and r['dataset']==dataset}
            control_ids = {instance_key(r) for r in outcomes if r['arm']==control and r['dataset']==dataset}
            pair_coverage.append(dict(dataset=dataset, arm=arm, control=control, arm_count=len(arm_ids), control_count=len(control_ids),
                paired_count=len(arm_ids & control_ids), excluded_arm_count=len(arm_ids-control_ids), scope='matched instance intersection'))
        for r in rows:
            if r['arm'] != arm:
                continue
            base = all_index.get((r['dataset'], control, str(r['id'])))
            if base is None:
                continue  # No historical L/G exists for M30; coverage is explicit.
            delta = r['gap_pct'] - base['gap_pct']
            pairs.append(dict(dataset=r['dataset'], arm=arm, control=control, id=r['id'], M=r['M'], N=r['N'], tightness=r['tightness'], gap_delta_pp=delta))
            for m, n, tightness in strata(r):
                paired[(r['dataset'], arm, control, m, n, tightness)].append(delta)
    paired_rows = [dict(dataset=d, arm=a, control=c, M=m, N=n, tightness=t, **paired_statistics(v)) for (d,a,c,m,n,t),v in paired.items()]
    variance, changes = [], []
    for arm in ('A', 'B'):
        if arm not in arms:
            continue
        trace_arm = [r for r in traces if r['arm'] == arm and r['dataset'] == 'chu']
        expected = {str(r['id']): r['steps'] for r in rows if r['arm'] == arm and r['dataset'] == 'chu'}
        actual = defaultdict(int)
        for r in trace_arm:
            actual[str(r['id'])] += 1
        if dict(actual) != expected:
            raise ValueError('Learned theta trace coverage differs from actual decisions')
        variance.extend(dict(arm=arm, **r) for r in variance_table([{**r, 'arm': 'L'} for r in trace_arm]))
        previous, deltas = {}, defaultdict(list)
        for r in trace_arm:
            iid = str(r['id'])
            if iid in previous:
                last = previous[iid]
                for field, value in r['theta'].items():
                    diff = np.abs(np.asarray(value) - np.asarray(last['theta'][field]))
                    deltas[(field, 'tau_cap>=0.8' if r['tau_cap'] >= .8 else 'tau_cap<0.8')].append(float(diff.mean()))
            previous[iid] = r
        changes.extend(dict(arm=arm, field=f, phase=p, transition_count=len(v), mean_absolute_entry_delta=float(np.mean(v))) for (f,p),v in deltas.items())
    reproduction = []
    for arm, old_arm in (('D_port', 'D'), ('G_port', 'G')):
        if arm not in arms:
            continue
        for dataset in ('chu', 'synthetic_N100'):
            evaluated = [r for r in rows if r['arm'] == arm and r['dataset'] == dataset]
            current = [r for r in evaluated if (dataset, old_arm, str(r['id'])) in historical_index]
            if not current:
                continue
            old = [historical_index[(dataset, old_arm, str(r['id']))] for r in current]
            measured, reference = np.mean([r['gap_pct'] for r in current]), np.mean([r['gap_pct'] for r in old])
            delta = float(measured - reference)
            reproduction.append(dict(arm=arm, dataset=dataset, count=len(current), evaluated_count=len(evaluated),
                not_covered_by_T5=len(evaluated)-len(current), scope='matched T5 subset only', measured_gap_pct=float(measured), T5_gap_pct=float(reference), delta_pp=delta,
                solutions_matching=sum(set(r['selected_item_indices']) == set(o['selected_item_indices']) for r,o in zip(current,old)),
                tolerance_pp=.005, formal_verified=not smoke and abs(delta) <= .005))
            if not smoke and abs(delta) > .005:
                write_json_result = {'status': 'failed', 'checks': reproduction}
                (dest/'reproduction.json').write_text(json.dumps(write_json_result, indent=2), encoding='utf-8')
                raise ValueError(f'{arm}/{dataset} differs from T5 by {delta:.6g} pp > .005 pp')
    recovery = []
    for r in summary:
        if r['arm'] not in ('A','B'):
            continue
        # All three means must use the SAME cases. A270 minus G180 is invalid.
        matched = []
        for value in groups[(r['dataset'],r['arm'],r['M'],r['N'],r['tightness'])]:
            g = all_index.get((r['dataset'],'G_port',str(value['id'])))
            learned = all_index.get((r['dataset'],'L_v6',str(value['id'])))
            if g is not None and learned is not None:
                matched.append((value['gap_pct'],g['gap_pct'],learned['gap_pct']))
        means = np.asarray(matched).mean(0) if matched else (None,None,None)
        a,g,learned = means
        recovery.append({k:r[k] for k in ('dataset','arm','M','N','tightness')} | {'count':len(matched), 'arm_count':r['count'],
            'excluded_arm_count':r['count']-len(matched), 'scope':'matched arm/G/L intersection',
            'arm_mean_gap_pct':a,'G_mean_gap_pct':g,'L_mean_gap_pct':learned,
            'recovery_fraction':(g-a)/(g-learned) if matched and abs(g-learned)>1e-12 else None})
    for name, values in [('per_instance',outcomes),('summary',summary),('paired_per_instance',pairs),('paired_summary',paired_rows),('paired_coverage',pair_coverage),('theta_variance',variance),('theta_step_changes',changes),('recovery',recovery)]:
        write_csv(dest/f'{name}.csv',values)
    (dest/'reproduction.json').write_text(json.dumps({'smoke_only':smoke,'checks':reproduction},indent=2)+'\n',encoding='utf-8')
    columns = ('dataset','arm','M_values','count','mean_gap_pct','p90_gap_pct','worst_gap_pct','quoted')
    table = ['| '+' | '.join(columns)+' |','| '+' | '.join('---' for _ in columns)+' |']
    table += ['| '+' | '.join(f'{r[c]:.6g}' if isinstance(r[c],float) else str(r[c]) for c in columns)+' |' for r in summary if r['M']=='all']
    (dest/'report.md').write_text('\n'.join(['# T8 single-greedy report','',f'smoke_only: {smoke}. best/last remain in separate output folders.',
        'Engineering smoke is not gate evidence. Single training seed; no seed generalization claimed.',
        'Chu uses best-known values; synthetic uses proven optima or incumbents, retaining per-instance flags.',
        'Formal D/A/B coverage is Chu270 (M5/M10/M30); smoke uses only the reduced selected set. G only covers dimensions present in T5 theta_means.json; no M30 theta is invented.',
        'L_v6 is quoted from byte/reference-matched T5 rows (Chu180 M5/M10). F0 is a guidance-only reference: Chu180 0.611%, synthetic 1.079%; no Chu270 or paired claim.',
        '',*table,'','Paired differences = arm minus control (pp); negative favors arm. Two-sided Wilcoxon p values are unadjusted.',
        'M/N/tightness strata and per-instance timing: summary.csv, paired_summary.csv, per_instance.csv.',
        'paired_coverage.csv reports every comparison denominator. In formal runs A/B and D comparisons cover Chu270; comparisons to historical G/L cover only their matched subset.',
        'Recovery: (G_port-arm)/(G_port-L_v6), all three means on the exact same intersection. No clipping; missing controls or zero denominator remain blank. M30 has no historical recovery reference.',
        'reproduction.json checks only the T5 overlap (Chu180), not a historical Chu270 reproduction.',
        'Theta variance follows T5 decision-weighted nested population sums of squares. Step changes average absolute entries across consecutive decisions, assigned to current tau_cap; the threshold-crossing step belongs to the late phase.',
        'Timing includes trace collection and explicit synchronization; LP is excluded from controller time. Historical timing is not compared.',
        'The research gate is decided by Claude after full runs.','']),encoding='utf-8')


def plot_training(runs, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    fields = ['validation_gap','R_mean','frac_better','entropy','explained_variance','approx_kl','clip_fraction','ranking_acceptance','ranking_retention','ranking_return_sign_retention','ranking_post_greedy_aligned_return_gap','ranking_post_accuracy',
        'profit_exponent','scarcity_exponent','bottleneck_weight','temperature','price_std','opportunity_weight','reserve_weight','lp_primal_weight','lp_dual_weight','interaction_weight_sum','companion_weight_sum']
    for run in runs:
        with (run/'train_log.csv').open(encoding='utf-8',newline='') as stream:
            rows=list(csv.DictReader(stream))
        plotted=[]
        for field in fields:
            points=[(float(r['update']),float(r[field])) for r in rows if r.get(field) not in ('',None) and np.isfinite(float(r[field]))]
            if not points:
                continue
            fig, ax=plt.subplots(figsize=(6,3.5));ax.plot(*np.asarray(points).T);ax.set(xlabel='Update',ylabel=field,title=run.name);ax.grid(alpha=.2)
            fig.tight_layout();fig.savefig(output/f'{run.name}_{field}.png',dpi=160);plt.close(fig);plotted.append(field)
        (output/f'{run.name}_manifest.json').write_text(json.dumps({'run':str(run),'train_log_sha256':sha256_file(run/'train_log.csv'),
            'meta_sha256':sha256_file(run/'meta.json'),'fields':plotted,'missing_fields':[f for f in fields if f not in plotted]},indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--runs',nargs='+',required=True,type=Path);parser.add_argument('--output',type=Path,default=Path('results/t8_training_curves'))
    args=parser.parse_args();plot_training(args.runs,args.output)


if __name__=='__main__':
    # Works as both python -m scripts.t8_report and python scripts/t8_report.py.
    main()
