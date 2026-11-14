"""Shared T9 artifacts, canonical Chu270 data and strictly matched comparisons."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from mkp_artifacts import sha256_file,source_provenance,write_json
from mkp_config import resolve_path
from mkp_env import MKPInstances
from mkp_v6_env import MKPEnv
from scripts.t8_data import datasets,instance_key
from scripts.t5_report import read_jsonl,write_csv,paired_statistics


def parser(description,output):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--source',default='results/t8_best_270');p.add_argument('--instances_dir',default='results/t5_v6')
    p.add_argument('--output',default=output);p.add_argument('--device',default='cuda:0')
    p.add_argument('--limit',type=int,choices=(0,2),default=0,help='smoke: one per M per set, never gate evidence')
    return p


def fresh(output,args):
    dest=Path(resolve_path(output))
    if dest.exists():raise FileExistsError(f'Preserve existing output: {dest}')
    dest.mkdir(parents=True)
    meta={'schema':'t9_artifact_v1','status':'running','smoke_only':bool(args.limit),'arguments':vars(args),
          'source':source_provenance(),'boundaries':'one greedy trajectory; core MILP/oracles are explicitly offline diagnostics; no search inference'}
    write_json(dest/'meta.json',meta)
    return dest,meta


def context(args):
    root=Path(resolve_path(args.source));meta=json.loads((root/'meta.json').read_text(encoding='utf-8'))
    if meta['status']!='complete' or meta['smoke_only']:raise ValueError('Require completed formal T8 Chu270 reference run')
    if sha256_file(root/'rows.jsonl')!=meta['rows_sha256']:raise ValueError('T8 rows SHA mismatch')
    rows=read_jsonl(root/'rows.jsonl')
    for arm in ('D_port','A','B'):
        if len({str(r['id']) for r in rows if r['arm']==arm and r['dataset']=='chu'})!=270:
            raise ValueError(f'T9 requires complete Chu270 for {arm}')
        if len({str(r['id']) for r in rows if r['arm']==arm and r['dataset']=='synthetic_N100'})!=128:
            raise ValueError(f'T9 requires complete synthetic128 for {arm}')
    history=read_jsonl(Path(resolve_path(args.instances_dir))/'attribution/rows.jsonl')
    records=datasets(Path(resolve_path(args.instances_dir)),history,bool(args.limit))
    canonical={instance_key(r[0]):r[0] for r in records}
    rows=[r for r in rows if instance_key(r) in canonical]
    for r in rows:
        info=canonical[instance_key(r)]
        if any(r[n]!=info[n] for n in ('data_sha256','reference','M','N','tightness','reference_proven')):raise ValueError('T8 data/reference mismatch')
    return records,rows,root


def tensors(record,device):
    _,p,w,c=record
    return MKPInstances(torch.tensor(p,device=device)[None],torch.tensor(w.T.copy(),device=device)[None],torch.tensor(c,device=device)[None])


def physical(record,device='cpu'):
    inst=tensors(record,device)
    return MKPEnv(inst.p,inst.r.transpose(1,2),inst.c)


def outcome(info,arm,result,seconds=0.,**extras):
    value=float(result['profit'][0])
    return {**info,'arm':arm,'objective':value,'gap_pct':100*(1-value/info['reference']),
        'selected_item_indices':result['sequence'][0].cpu().tolist(),'steps':result['steps'],'time_total_s':seconds,**extras}


def strata(row):return [('all','all','all'),(row['M'],row['N'],'all'),(row['M'],row['N'],row['tightness'])]


def summarize(rows):
    groups=defaultdict(list)
    for r in rows:
        for m,n,t in strata(r):groups[(r['dataset'],r['arm'],m,n,t)].append(r)
    return [dict(dataset=d,arm=a,M=m,N=n,tightness=t,count=len(v),mean_gap_pct=float(np.mean([r['gap_pct'] for r in v])),
        p90_gap_pct=float(np.quantile([r['gap_pct'] for r in v],.9)),worst_gap_pct=max(r['gap_pct'] for r in v),
        mean_time_total_s=float(np.mean([r.get('time_total_s',0.) for r in v])),
        **{f'mean_{field}':float(np.mean([r[field] for r in v])) if all(field in r for r in v) else None
           for field in ('time_lp_s','time_controller_s')}) for (d,a,m,n,t),v in groups.items()]


def paired(rows,comparisons):
    index={(r['dataset'],r['arm'],str(r['id'])):r for r in rows};groups=defaultdict(list);raw=[]
    for a,c in comparisons:
        for r in rows:
            base=index.get((r['dataset'],c,str(r['id'])))
            if r['arm']!=a or base is None:continue
            if r['data_sha256']!=base['data_sha256'] or r['reference']!=base['reference']:raise ValueError('Unmatched pair')
            delta=r['gap_pct']-base['gap_pct']
            raw.append(dict(dataset=r['dataset'],arm=a,control=c,id=r['id'],M=r['M'],N=r['N'],tightness=r['tightness'],gap_delta_pp=delta))
            for m,n,t in strata(r):groups[(r['dataset'],a,c,m,n,t)].append(delta)
    return raw,[dict(dataset=d,arm=a,control=c,M=m,N=n,tightness=t,**paired_statistics(v)) for (d,a,c,m,n,t),v in groups.items()]


def targets(instances_dir):
    """Read original cached binary targets; do not re-solve or change incumbents."""
    directory=Path(resolve_path(instances_dir));manifest=json.loads((directory/'manifest.json').read_text())
    output={}
    for entry in manifest['evaluation_source_batches']:
        path=Path(resolve_path(entry['reference_file'])) if 'reference_file' in entry else Path(resolve_path('cache'))/Path(entry['reference_path'].replace('\\','/')).name
        if sha256_file(path)!=entry['reference_sha256']:raise ValueError('MILP target cache changed')
        with np.load(path,allow_pickle=False) as archive:
            for i,x in enumerate(archive['solution']):output[f"syn100_M{entry['M']}_{i:03d}"]=x.astype(bool)
    return output


def sync(device):
    if torch.device(device).type=='cuda':torch.cuda.synchronize(device)
