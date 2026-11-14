"""Prepare cert-only training; formal training requires both gates and a decision."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mkp_artifacts import write_json,source_provenance,sha256_file
from mkp_config import resolve_path,get_device,config_from_dict
from scripts.profile_t9 import cert_config


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dictionary',default='results/t9_dictionary')
    p.add_argument('--measure',default='results/t9_measure');p.add_argument('--output',default='results/t9_configs')
    p.add_argument('--train',action='store_true');p.add_argument('--decision',choices=('cert',),help='Explicit user/Claude selection after reviewing formal gates')
    p.add_argument('--cuda_device',type=int,default=0)
    p.add_argument('--lp_backend',choices=('eager','cuda_graph'),default='eager');a=p.parse_args()
    cfg=cert_config();cfg.device.cuda_device=a.cuda_device;cfg.device.lp_backend=a.lp_backend
    root=Path(resolve_path(a.dictionary));calibration=root/'calibration.json'
    if calibration.exists():
        c=json.loads(calibration.read_text());cfg.model.cert_weights=c['weights'];cfg.model.cert_temperature=c['temperature']
        cfg.model.cert_temperature_min=c['temperature_min'];cfg.model.cert_temperature_max=c['temperature_max']
    dest=Path(resolve_path(a.output));dest.mkdir(parents=True,exist_ok=True);path=dest/'t9_a_cert_s1234.json'
    if path.exists() and config_from_dict(json.loads(path.read_text())).to_dict()!=cfg.to_dict():raise FileExistsError('Preserve different prepared config; use a new output directory')
    cfg.save(str(path));write_json(dest/'meta.json',dict(source=source_provenance(),training_requested=a.train,
        calibration_present=calibration.exists(),configuration='provisional uniform weights until formal calibration; no v6/v6c training',config=cfg.to_dict()))
    print(f'Prepared cert-only config: {path}',flush=True)
    if not a.train:return
    if not calibration.exists():raise ValueError('Training blocked: formal calibration is absent')
    gate=json.loads((root/'gate.json').read_text());measurement=json.loads((Path(resolve_path(a.measure))/'gates.json').read_text())
    if a.decision!='cert' or gate['smoke_only'] or not gate['base_cert_eligible']:
        raise ValueError('Training blocked: require reviewed formal cert gate and explicit --decision cert')
    if measurement['smoke_only'] or not measurement['quantities'].get('LP_gate_passed') or measurement.get('stop_if_price_contradiction'):
        raise ValueError('T9-0 measurement gate not passed')
    for arm in ('A','B'):
        if arm+'_P1_retention' not in measurement['quantities']:raise ValueError('Complete T9-0 freeze gate first')
    if gate['calibration']!=c:raise ValueError('Calibration changed after dictionary gate')
    for directory in (root,Path(resolve_path(a.measure))):
        status=json.loads((directory/'meta.json').read_text(encoding='utf-8'))
        if status['status']!='complete' or status['smoke_only']:raise ValueError('Formal jobs must finish completely before training')
    from mkp_Trainer import set_seed
    from mkp_t8_train import T8Trainer
    set_seed(1234);trainer=T8Trainer(cfg,get_device(cfg))
    trainer.gate_provenance={'decision':a.decision,'dictionary_gate_sha256':sha256_file(root/'gate.json'),
        'measurement_gate_sha256':sha256_file(Path(resolve_path(a.measure))/'gates.json'),'calibration_sha256':sha256_file(calibration)}
    trainer.fit()


if __name__=='__main__':main()
