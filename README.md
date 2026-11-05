# Constrained_AwareMKP

> 2026-11-04: T1-R is implemented. Follow [the T1-R run instructions](docs/WEAK_DECODER_T1_EXECUTION.md#t1-r). Older sections describe the T1 baseline. Old checkpoints retain raw scores; new configs use normalized scores and fixed temperature. Calibrate beta before experiments.

> 2026-11-04: T1-R implementation is now available. Follow [the T1-R run instructions](docs/WEAK_DECODER_T1_EXECUTION.md#t1-r) for full regression, synthetic beta calibration, diagnostics and gated training. Older sections below describe the T1 baseline; old checkpoints retain raw scores. New configs default to normalized scores and fixed temperature; the experimental beta must be calibrated first.

State-conditioned resource control for multidimensional 0-1 knapsack problems.
The current implementation is **T1 (G0 + G1)** of
[`docs/CODE_GUIDANCE_WEAK_DECODER_MIGRATION.md`](docs/CODE_GUIDANCE_WEAK_DECODER_MIGRATION.md).
T0 diagnoses the historical model separately. G2/G3 and representation changes remain later stages.
An average greedy gap below 0.1% is a research target; this migration does not establish that result.

## Policy

```
item tokens + resource tokens + dense consumption relations
  -> static encoding once, dynamic state fusion at each construction step
  -> controller: theta = (u in R^M, alpha in [0.5, 2.5])
  -> fixed transparent executor on the feasible set
  -> select one item, update capacities, repeat until no item fits
```

The original item-resource relation encoder is retained. Its item embeddings never enter an
item scoring head. The controller outputs only M centered resource log-price corrections and,
when enabled, one profit exponent. There are no learned item residuals, primitive mixtures,
state-dependent temperatures, critic, PPO, repair or search in the executor.

```
u = u_max * tanh(price_head(resource, context)); u -= mean(u)
lambda = (root_mu + price_floor * mean(root_mu) + 1e-6) * exp(u)
score_i = alpha * log(p_i / mean(p))
          - log(sum_j lambda_j * r_ji / (c_rem_j + 1))
pi(i|s) = softmax over currently feasible items of score_i / T
```

`T` is one global learned scalar (initially 1/3), or a configurable constant; it does not
change greedy rankings and is excluded from theta. The default `hard_mask` guarantees
feasible selections; trajectories terminate at maximal solutions. Main-method reward is
`profit / LP_upper_bound`. `perceive` and `soft_mask` are explicitly named ablations with
rejection penalties, never main-method evidence.

`theta_groups=price,alpha` gives M+1 outputs independently of N; `theta_groups=price` gives M.
`price_anchor=root` is supported in T1. Requests for block/LP groups or a residual anchor
fail explicitly until their separate stages are implemented.

## Training and checkpoints

```powershell
pip install -r requirements.txt
python train_ConstrainedMKP.py --print_config

# Separately trained arms with matched budgets and seed; repeat for 3 independent seeds
python train_ConstrainedMKP.py --model.theta_update step --device.seed 1234 --train.epochs 40 --train.run_name dynamic_s1234
python train_ConstrainedMKP.py --model.theta_update static --device.seed 1234 --train.epochs 40 --train.run_name static_s1234
```

Training remains REINFORCE with the mean of K=16 sampled trajectories of the same instance
as baseline. Gradient checkpointing and bf16 autocast remain available. Checkpoints are
selected only by single-trajectory greedy on fixed **generated** validation instances
(default N=100, M=5/10, 64 per group, fixed validation seed), using cached HiGHS MILP
references. Time-limited incumbents are marked separately from proven optima. The LP
reference can be requested explicitly as a diagnostic. Chu data never select checkpoints.

Logs include loss, gradient norm/finite flags, |u|, alpha and global T. Each run preserves
configuration, source commit and diff, seed, synthetic-data/reference hashes and selection
rule; checkpoint sidecars preserve checkpoint SHA256. Legacy delta checkpoints are rejected
by the T1 loader. Dynamic checkpoints cannot stand in for separately trained static policies.

## Evaluation

`greedy` is the primary stage: exactly one argmax trajectory. Fixed theta bypasses the
network. Untrained controls use PyTorch random head initialization with the same seed,
which differs from the zero-initialized warm start used for training. Static and dynamic
learned arms require their respective independently trained checkpoints.

Sampling and ILS are separately named stages with matched trajectory and search budgets.
The optional `mkp_refine.py` implementation is unchanged. Report their gaps separately;
search outcomes are not weak-decoder greedy outcomes. Per-instance paired differences
use percentage points; stratified reports include M, N, tightness, worst and p90 gaps.

See [the execution report](docs/WEAK_DECODER_T1_EXECUTION.md) for verified commands,
smoke configuration, outputs and remaining experimental work.

## Files

| File | Purpose |
|---|---|
| `mkp_config.py` | Dataclass defaults and `--section.field value` overrides |
| `mkp_env.py` | Synthetic generator, approximate root LP dual, feasible construction environment |
| `mkp_Model.py` | Relation encoder, resource theta controller, parameter-free executor |
| `mkp_Trainer.py` | REINFORCE, synthetic validation, checkpoint compatibility and logging |
| `mkp_reference.py` | Cached HiGHS MILP references for synthetic validation |
| `mkp_artifacts.py` | Source/data/checkpoint provenance helpers |
| `Benchmark_zeroShot.py` | Named arms and inference stages, paired and stratified results |
| `mkp_refine.py` | Optional external local search/ILS stage, unchanged |
| `tests/` | Interface, feasibility, permutation, training and evaluation regressions |

```powershell
python -m unittest discover -s tests -v
```
