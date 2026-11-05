"""Plot user-produced gate training and validation logs, including epoch -1."""
import argparse,csv
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--runs",nargs="+",default=["runs/t1r_dynamic_s1234","runs/t1r_static_s1234"])
    p.add_argument("--output",default="results/t1r_curves");a=p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dest=Path(a.output);dest.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(3,1,figsize=(9,10),constrained_layout=True)
    for run in a.runs:
        folder=Path(run)
        with (folder/"train_log.csv").open(newline="",encoding="utf-8-sig") as f:train=list(csv.DictReader(f))
        with (folder/"val_log.csv").open(newline="",encoding="utf-8-sig") as f:val=list(csv.DictReader(f))
        name=folder.name
        axes[0].plot([float(r["epoch"]) for r in val],[float(r["gap_mean"]) for r in val],marker="o",label=name)
        for metric in ("alpha_upper_frac","alpha_lower_frac","u_boundary_frac"):
            axes[1].plot([float(r["step"]) for r in train],[float(r[metric]) for r in train],label=name+" "+metric)
        axes[2].plot([float(r["step"]) for r in train],[float(r["raw_score_std"]) for r in train],label=name)
    for ax,title in zip(axes,["Synthetic greedy validation gap (%) including initialization", "Theta boundary fractions", "Raw feasible-score standard deviation"]):
        ax.set_title(title);ax.legend(fontsize=7);ax.grid(alpha=.25)
    axes[0].set_xlabel("epoch (-1 = initialization)");axes[1].set_xlabel("training step");axes[2].set_xlabel("training step")
    fig.savefig(dest/"gate_curves.png",dpi=180);fig.savefig(dest/"gate_curves.pdf");plt.close(fig)
if __name__=="__main__":main()
