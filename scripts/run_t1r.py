"""Prepare matched T1-R configs from the original run and synthetic beta calibration.
By default only writes configs; --train explicitly starts the user's full training.
"""
import argparse
from t1r_common import *
from mkp_config import get_device
from mkp_Trainer import MKPTrainer, set_seed

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="runs/dynamic_s1234/config.json")
    p.add_argument("--calibration",default="results/t1r_beta_calib/meta.json")
    p.add_argument("--output",default="results/t1r_configs")
    p.add_argument("--regression",default="results/t1r_greedy_regression/meta.json")
    p.add_argument("--mode",choices=["step","static","both"],default="both")
    p.add_argument("--train",action="store_true")
    a=p.parse_args();meta=json.loads(Path(resolve_path(a.calibration)).read_text(encoding="utf-8"))
    if meta.get("smoke_only") or meta.get("selection")!="synthetic fixed-theta best-of-16; ties larger beta":
        raise ValueError("Requires full synthetic-only calibration")
    beta=float(meta["beta0"])
    if beta not in (1,2,4,8,16,32,64):raise ValueError("Invalid calibrated beta")
    regression=json.loads(Path(resolve_path(a.regression)).read_text(encoding="utf-8"))
    if regression.get("smoke_only") or regression.get("mismatch_count") != 0:
        raise ValueError("Full greedy regression required; inspect any mismatches before training")
    original=config(a.config)
    for section in ("env","model"):
        lhs=original.to_dict()[section];rhs=dict(meta["config"][section])
        for key in ("temperature_init",):lhs.pop(key,None);rhs.pop(key,None)
        if lhs!=rhs:raise ValueError("Calibration configuration differs from training source")
    if original.train.val_seed!=meta["config"]["train"]["val_seed"] or any(original.to_dict()["train"][k]!=meta["config"]["train"][k] for k in ("val_sizes","val_dims","val_instances")):
        raise ValueError("Calibration validation set mismatch")
    dest=Path(resolve_path(a.output))
    dest.mkdir(parents=True,exist_ok=True)
    write_json(dest/"meta.json",{"beta0":beta,"regression_sha256":sha256_file(resolve_path(a.regression)),"calibration_sha256":sha256_file(resolve_path(a.calibration)),"original_config_sha256":sha256_file(resolve_path(a.config)),"source":source_provenance()})
    for mode in (("step","static") if a.mode=="both" else (a.mode,)):
        cfg=config(a.config);cfg.model.theta_update=mode;cfg.model.temperature_init=1/beta
        cfg.train.epochs=10;cfg.train.initial_validation=True;cfg.train.gate_early_stop=mode=="step"
        cfg.train.resume=cfg.train.init_from=""
        cfg.train.run_name="t1r_"+("dynamic" if mode=="step" else "static")+"_s1234"
        if cfg.device.seed!=1234:raise ValueError("T1-R gate specifies seed1234")
        if Path(resolve_path(cfg.train.runs_dir),cfg.train.run_name).exists():raise FileExistsError(cfg.train.run_name)
        cfg_path=dest/(mode+".json")
        if cfg_path.exists() and json.loads(cfg_path.read_text(encoding="utf-8"))!=cfg.to_dict():
            raise FileExistsError(f"Different prepared configuration: {cfg_path}")
        cfg.save(str(cfg_path))
        if a.train:
            set_seed(cfg.device.seed);MKPTrainer(cfg,get_device(cfg)).fit()
if __name__=="__main__":main()
