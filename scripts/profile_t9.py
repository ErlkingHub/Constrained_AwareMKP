"""Measure a few complete cert updates, then project the unchanged T8-A budget.
No formal run/checkpoint is created. No optimizer/LP speed optimization is applied.
"""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mkp_artifacts import write_json,sha256_file
from mkp_env import generate_instances
from mkp_t8_train import T8Trainer
from mkp_Trainer import set_seed
from scripts.run_t8 import configuration
from scripts.t9_common import fresh,sync
from scripts.t5_report import write_csv


def cert_config(calibration=None):
    c=configuration('A');c.model.executor=c.model.theta_family='cert';c.model.ctrl_basis_features=True
    c.train.run_name='t9_a_cert_s1234'
    if calibration:
        values=json.loads(Path(calibration).read_text())
        for field,key in (('cert_weights','weights'),('cert_temperature','temperature'),('cert_temperature_min','temperature_min'),('cert_temperature_max','temperature_max')):
            setattr(c.model,field,values[key])
    return c


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='results/t9_profile');p.add_argument('--device',default='cuda:0')
    p.add_argument('--batches',nargs='+',type=int,default=[64,32]);p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--sizes',nargs='+',type=int,default=[50,100,150]);p.add_argument('--dims',nargs='+',type=int,default=[5,10]);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--calibration',help='optional final calibration.json; re-profile after formal gate selection')
    a=p.parse_args();dest,meta=fresh(a.output,a);device=torch.device(a.device);torch.set_num_threads(2);rows=[]
    meta.update(device=str(device),GPU=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        scope='isolated profiling updates only; no fit, validation selection or formal checkpoint',config=cert_config(a.calibration).to_dict(),
        calibration_sha256=sha256_file(a.calibration) if a.calibration else None)
    write_json(dest/'meta.json',meta)
    try:
        for batch in a.batches:
            for m in a.dims:
                for n in a.sizes:
                    set_seed(1234);cfg=cert_config(a.calibration);cfg.train.batch_size=batch
                    trainer=T8Trainer(cfg,device);trainer.step=1 # measure primary update; greedy diagnostics separately estimated below
                    inst=generate_instances(batch,n,m,generator=torch.Generator().manual_seed(910000+n+m)).to(device)
                    for repeat in range(a.repeats):
                        sync(device);tick=time.perf_counter();stats=trainer.update(inst);sync(device);elapsed=time.perf_counter()-tick
                        row=dict(batch=batch,M=m,N=n,repeat=repeat,seconds=elapsed,trajectories=batch*16)
                        rows.append(row);write_csv(dest/'updates.csv',rows)
                        print(json.dumps(row),flush=True)
                    del trainer,inst
        estimates={str(b):float(np.mean([r['seconds'] for r in rows if r['batch']==b])) for b in a.batches}
        if {'64','32'}<=estimates.keys():
            reduced=estimates['64']*2500>24*3600
            seconds=250*estimates['64']+2250*estimates['32'] if reduced else 2500*estimates['64']
            meta.update(primary_update_hours=seconds/3600,unchanged_rule_reduces_B_after250=reduced,
                engineering_range_hours=[seconds/3600,seconds/3600*1.25],
                limitations='one generated batch per shape, weights near initialization; 0-25% allowance for diagnostics/validation/I/O/drift, not a confidence interval. No full-run measurement.')
        meta.update(status='complete',mean_seconds_per_update=estimates)
    except Exception as exc:meta.update(status='failed',error=repr(exc));raise
    finally:write_json(dest/'meta.json',meta)


if __name__=='__main__':main()
