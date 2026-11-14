"""T9 evaluation: single greedy, complete Chu270, original synthetic128."""
import json,sys,time
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mkp_artifacts import write_json,sha256_file
from mkp_config import config_from_dict,resolve_path
from mkp_Model import build_model
from mkp_t8_train import CERT_SCHEMA
from scripts.profile_t9 import cert_config
from scripts.t9_common import parser,fresh,context,tensors,physical,outcome
from scripts.t5_report import read_jsonl,write_csv
from scripts.eval_t8 import Timings
from scripts.run_t9_measure import exact_lp,identity_parts,selected
from scripts.t9_report import evaluation_report


def main():
    p=parser(__doc__,'results/t9_best_270');p.add_argument('--dictionary',default='results/t9_dictionary')
    p.add_argument('--arms',nargs='+',choices=('D_cert','K*','T9-A'),default=['D_cert','K*','T9-A'])
    p.add_argument('--checkpoint',default='checkpoints/t9_a_cert_s1234/best.pt');p.add_argument('--tag',choices=('best','last'),default='best')
    a=p.parse_args();torch.set_num_threads(2);device=torch.device(a.device)
    if len(a.arms)!=len(set(a.arms)):raise ValueError('Duplicate arm')
    records,baselines,root=context(a);dictionary=Path(resolve_path(a.dictionary));c=json.loads((dictionary/'calibration.json').read_text())
    if c['smoke_only'] and not a.limit:raise ValueError('Formal evaluation requires formal calibration')
    dest,meta=fresh(a.output,a);rows=[];traces=[]
    meta['reference_rows_sha256']=sha256_file(root/'rows.jsonl')
    meta['calibration_sha256']=sha256_file(dictionary/'calibration.json');meta['models']={}
    try:
        with (dest/'rows.jsonl').open('w',encoding='utf-8') as out,(dest/'traces.jsonl').open('w',encoding='utf-8') as trace_file:
            for arm in a.arms:
                cfg=cert_config();cfg.model.cert_weights=c['weights'];cfg.model.cert_temperature=c['temperature']
                cfg.model.cert_temperature_min=c['temperature_min'];cfg.model.cert_temperature_max=c['temperature_max']
                if arm=='T9-A':
                    path=Path(resolve_path(a.checkpoint))
                    if path.name!=a.tag+'.pt':raise ValueError('Keep best/last separate')
                    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
                    if checkpoint['schema']!=CERT_SCHEMA:raise ValueError('Expected new cert checkpoint, not v6')
                    cfg=config_from_dict(checkpoint['config']);model=build_model(cfg);model.load_state_dict(checkpoint['model'],strict=True)
                    meta['checkpoint_sha256']=sha256_file(path)
                    meta['checkpoint_update']=checkpoint['update']
                else:model=build_model(cfg)
                meta['models'][arm]=cfg.to_dict()
                model=model.to(device).eval()
                for record in records:
                    info=record[0];inst=tensors(record,device)
                    with torch.no_grad(),Timings(SimpleNamespace(v6_engine=model.executor_engine),device) as timing:
                        result=model(None,inst,decode='greedy',theta_override='fixed' if arm=='K*' else None,return_trace=arm=='T9-A')
                    solution=result['solution'][0].cpu().numpy();remaining=record[3]-record[2][solution].sum(0)
                    if np.any(remaining < -1e-8) or ((record[2]<=remaining).all(1)&~solution).any():raise ValueError('Invalid greedy construction')
                    row=outcome(info,arm,result,timing.total,time_lp_s=timing.lp_seconds,
                        time_controller_s=max(0.,timing.controller_seconds-timing.controller_lp),lp_solves=timing.calls)
                    rows.append(row);out.write(json.dumps(row)+'\n');out.flush()
                    for trace in result['trace']:
                        r={**{k:info[k] for k in ('dataset','id','M','N','tightness')},'arm':arm,'step':trace['step'],'tau_cap':trace['tau_cap'][0],
                           'theta':{n:v[0] for n,v in trace['theta'].items()},
                           **{n:trace[n][0] for n in ('own_regret','blocked_regret','dual_reoptimization_gain')}}
                        traces.append(r);trace_file.write(json.dumps(r)+'\n')
                    trace_file.flush();print(f'T9 {arm} {info["dataset"]}/{info["id"]}: {row["gap_pct"]:.4f}%',flush=True)
                del model
        history=read_jsonl(Path(a.instances_dir)/'attribution/rows.jsonl')
        evaluation_report(dest,rows,traces,baselines,records,history)
        identities=[];all_rows=rows+baselines
        for record in records:
            exact=exact_lp(physical(record).state)
            for row in all_rows:
                if row['dataset']!=record[0]['dataset'] or str(row['id'])!=str(record[0]['id']):continue
                identities.append({**record[0],'arm':row['arm'],**identity_parts(physical(record).state,selected(row,record[0]['N']),exact['prices'])})
        write_csv(dest/'gap_identity.csv',identities)
        meta.update(status='complete',rows_sha256=sha256_file(dest/'rows.jsonl'))
    except Exception as exc:meta.update(status='failed',error=repr(exc));raise
    finally:write_json(dest/'meta.json',meta)


if __name__=='__main__':main()
