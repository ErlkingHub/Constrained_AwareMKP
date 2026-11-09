"""Prepare both T8 configurations; training requires explicit --train A or B."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mkp_config import Config,resolve_path,get_device
from mkp_artifacts import write_json,source_provenance


def configuration(arm):
    cfg=Config();cfg.model.executor=cfg.model.theta_family="v6"
    cfg.model.ctrl_residual_lp_features=True;cfg.model.v6_critic=arm=="B"
    cfg.model.theta_update="step";cfg.model.theta_interval=1
    cfg.model.score_normalization=False;cfg.device.amp=False;cfg.device.seed=1234
    cfg.train.signal="item_pg" if arm=="A" else "ppo_gae"
    cfg.train.epochs=10;cfg.train.steps_per_epoch=250;cfg.train.batch_size=64
    cfg.train.num_samples=16 if arm=="A" else 1
    cfg.train.max_rollouts_per_chunk=64 # bounded memory, complete K groups per microbatch
    cfg.train.lr=1e-4;cfg.train.grad_clip=1.;cfg.train.entropy_coef=.01
    cfg.train.run_name=f"t8_{arm.lower()}_jointM_s1234"
    cfg.eval.ls_enable=False
    return cfg


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train",choices=("A","B"))
    parser.add_argument("--cuda_device",type=int,default=0)
    parser.add_argument("--output",default="results/t8_configs")
    args=parser.parse_args(argv)
    dest=Path(resolve_path(args.output));dest.mkdir(parents=True,exist_ok=True)
    configs={}
    for arm in ("A","B"):
        cfg=configuration(arm);cfg.device.cuda_device=args.cuda_device
        path=dest/f"t8_{arm.lower()}_jointM_s1234.json"
        if path.exists():
            import json
            if json.loads(path.read_text(encoding="utf-8"))!=cfg.to_dict():raise FileExistsError(f"Preserve different config: {path}")
        cfg.save(str(path));configs[arm]=cfg
    write_json(dest/"meta.json",{"source":source_provenance(),"training_requested":args.train,
        "inference":"one greedy; ranking continuations are training-only; critic/feasibility head never used for inference",
        "arms":{arm:cfg.to_dict() for arm,cfg in configs.items()}})
    print(f"Prepared T8-A/B: {dest}")
    if args.train:
        from mkp_Trainer import set_seed
        from mkp_t8_train import T8Trainer
        cfg=configs[args.train];set_seed(1234);T8Trainer(cfg,get_device(cfg)).fit()


if __name__=="__main__":main()
