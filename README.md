# Constrained_AwareMKP

Structure-aware strategy learning for the **Multidimensional Knapsack Problem (MKP)** with deep
reinforcement learning. This is subtopic 1 of *Structure-Aware and Strategy Learning for
Combinatorial Optimization Problems*.

Target: an average gap of ≤ 0.1 % to the best-known values on `MKP_dataset/Benchmarks_Chu.mat`,
evaluated zero-shot. The benchmark instances are never used for training.

## Pipeline

```
MKP dataset / Chu-Beasley generator                     (mkp_env.generate_instances, read_mat.py)
   ↓
Dynamic MKP environment                                 (mkp_env.MKPEnv; RL4CO-style reset/step/reward)
   ↓
State encoder: item↔resource bipartite graph Transformer (mkp_Model.StateEncoder)
   ↓        static pass once per instance + dynamic state injected at every step
High-level strategy controller: dual-axis Transformer   (mkp_Model.StrategyController)
   ↓        outputs a readable strategy (λ_t, w_t, κ_t, δ)
Low-level weak decoder: parameter-free scorer           (mkp_Model.WeakDecoder)
   ↓
select item → env updates remaining capacities → repeat until no unprocessed item fits
   ↓
item sequence / MKP solution  (+ optional post-hoc refinement, mkp_refine.py)
```

### State awareness and constraint awareness
* **Bipartite tokens.** There are N item tokens and M resource tokens. Following Gasse et al.
  (2019), these play the roles of the variable nodes and constraint nodes of the MILP.
* **Edge-biased attention.** The consumption matrix `r_ij` sets the attention scores between
  items and resources, using the MatNet idea (Kwon et al., 2021). The static values are
  `r_ij / mean_j r_ij` and the LP-dual price share. At every step the encoder also gets the
  **current** load `r_ij / c_rem_i` and a per-dimension overflow indicator.
* **Dual-axis multi-head attention** in each layer:
  * item self-attention models **profit competition** among the remaining items. Processed
    items are hidden as keys.
  * item ← resource attention shows each item the resource dimensions it stresses.
  * resource ← item attention gathers the demand that is **competing for each resource**.
  * resource self-attention models coupling and competition between resource dimensions.
* **Per-step dynamic features.** These cover remaining capacity fractions, how many average
  items still fit per dimension, demand pressure, per-item max load, bottleneck slack after
  inclusion, fit status, item status and progress. All features are scale-free, so one policy
  transfers across N, M and the tightness α.

### Constraint awareness without feasibility masks
The default `env.feasibility_mode = "perceive"` uses **no feasibility mask**. The policy can
pick any unprocessed item. If the item does not fit, the environment *rejects* it: the item
is discarded, capacities stay the same, and the reward is penalised by
`reject_penalty * #rejected / N`. Capacities only ever decrease, so a rejected item could
never fit later. A rejection therefore costs no solution quality, and every solution is
feasible by construction. The agent still has to *perceive* feasibility from the state to
avoid wasted steps and penalties. The benchmark reports this as `sample_feasible_pick_rate` and
`greedy_rejects`.

The only mask is the structural "already processed" mask, which prevents picking an item
twice. For ablations, `soft_mask` (a logit penalty on overflow) and `hard_mask` (the classic
−∞ mask) are also available.

### High-level strategy and low-level weak decoder
At each step t the controller outputs:

| symbol | meaning |
|---|---|
| `λ_t ∈ R^M_+` | resource prices: LP-dual prices × exp(±3), adapted to the state |
| `w_t ∈ Δ^K` | mixture over K interpretable primitives |
| `κ_t > 0` | decisiveness (inverse temperature) |
| `δ_j` | bounded per-item residual, \|δ\| ≤ 3 |

The weak decoder has no parameters and is fast and readable:

```
score_j = κ_t · Σ_k w_tk · z(φ_k(j)) + δ_j          (z = z-score over remaining items)
φ_profit      = log p_j
φ_efficiency  = log p_j − log Σ_i r_ij / c_i
φ_price_ratio = log p_j − log Σ_i λ_ti r_ij / c_rem_i      (dynamic surrogate ratio)
φ_slack       = min_i (c_rem_i − r_ij) / c_i               (< 0 ⇔ infeasible: perceived, not masked)
π(j) = softmax(score)
```

The heads are zero-initialised. As a result, the untrained policy is the stochastic classic
LP-dual-ratio heuristic, which gives RL a strong, interpretable starting point. The benchmark
CSV logs the learned mean `w_*`, `κ`, `|δ|` and `|log λ-adjustment|` for every instance.

### Training
The trainer uses REINFORCE with a shared multi-sample baseline (POMO-style). It draws K
rollouts per instance, and the baseline is their mean reward.

* **Reward:** `profit / LP_upper_bound − reject_penalty · rejects / N`. The bound comes from a
  batched Lagrangian-dual solver on the GPU (`mkp_env.lp_dual`), which lands within about
  0.05 % of scipy's LP optimum on the Chu instances.
* **Training data:** generated online with the Chu & Beasley (1998) generator, with
  N ∈ {50, 100, 150}, M ∈ {5, 10} and α ∈ {0.25, 0.5, 0.75}.
* **Memory:** gradient checkpointing of each decision step keeps memory low, and bf16
  autocast is on by default.

### Zero-shot benchmark
For each instance, `Benchmark_zeroShot.py` reports:
* `gap_greedy`: one greedy rollout of the policy.
* `gap_policy`: the best of greedy plus `num_samples` sampled rollouts. This is the pure
  learned policy.
* `gap_final`: the best `ls_pool` distinct policy solutions after the optional refinement in
  `mkp_refine.py`. The refinement uses interpretable moves: ADD fill, best-improvement 1-1
  swap, drop-and-refill, and iterated local search. It runs batched on the GPU.
* `gap_dual_greedy`: the classic LP-dual-ratio greedy, as a reference point.

Calibration without any learning: dual-greedy starts with noise, plus refinement, give
about 0.11 % average gap on a subset of Chu 5/10-dim instances. The learned policy's job is
to provide better and more diverse starting points, mainly on the hard 10-dim instances.
That is how the target falls below 0.1 %. The CSV keeps policy-only and refined numbers
separate.

## Files

| file | content |
|---|---|
| `mkp_config.py` | every hyper-parameter with its default; CLI overrides `--section.field value` |
| `mkp_env.py` | instance generator, LP-dual bound and prices, dynamic batched env, observations |
| `mkp_Model.py` | state encoder, strategy controller, weak decoder, policy |
| `mkp_refine.py` | optional batched post-hoc local search / ILS |
| `mkp_Trainer.py` | REINFORCE trainer; logs → `runs/<timestamp>/`, ckpts → `checkpoints/` |
| `train_ConstrainedMKP.py` | minimal training entry |
| `Benchmark_zeroShot.py` | zero-shot Chu benchmark; CSVs → `results/<timestamp>_<run>/` |
| `read_mat.py` | `.mat` benchmark parser (given) |

## Usage

```bash
pip install -r requirements.txt

# 1) train (uses cuda:1 by default; set --device.cuda_device to change)
python train_ConstrainedMKP.py
python train_ConstrainedMKP.py --train.epochs 200 --train.num_samples 32 --env.train_sizes 100,150,200
python train_ConstrainedMKP.py --train.resume checkpoints/last.pt

# 2) zero-shot benchmark (loads checkpoints/best.pt)
python Benchmark_zeroShot.py
python Benchmark_zeroShot.py --eval.num_samples 512 --eval.ils_rounds 200
python Benchmark_zeroShot.py --eval.ls_enable false          # pure policy numbers only
python Benchmark_zeroShot.py --eval.dims 5,10,30             # include the 30-dim instances
python Benchmark_zeroShot.py --baseline_only                 # heuristic reference, no model
```

Training writes `runs/<timestamp>/{config.json, log.txt, train_log.csv, val_log.csv, TensorBoard}`
and `checkpoints/<timestamp>/{best,last}.pt`. It also copies these to
`checkpoints/{best,last}.pt`. Validation runs greedy decoding on a fixed set of generated
instances and measures the gap to the LP bound. The best checkpoint is chosen on this set,
not on the benchmark.

Tuning tips:
* Use larger `train.num_samples` and `train.batch_size` if memory allows. You can also lower
  `train.max_rollouts_per_chunk` to split rollouts.
* To fine-tune on larger N, set `train.init_from=checkpoints/best.pt` and
  `env.train_sizes=250`.
* At test time, `eval.num_samples`, `eval.ls_pool` and `eval.ils_rounds` trade time for gap.

## References (ideas adopted, all re-implemented)
1. P.C. Chu, J.E. Beasley (1998). *A genetic algorithm for the multidimensional knapsack
   problem.* J. Heuristics 4:63–86. Used for the instance generator (the training
   distribution), the LP-dual pseudo-utility ratio, and the ADD/DROP repair ideas.
2. H. Pirkul (1987). *A heuristic solution procedure for the multiconstraint zero-one knapsack
   problem.* Naval Res. Logistics 34:161–172. Surrogate-dual multipliers are the basis of the
   resource prices that the controller adapts.
3. S. Senju, Y. Toyoda (1968). *An approach to linear programming with 0-1 variables.*
   Management Science 15:B196–B207. Their effective-gradient idea weights resources by
   remaining capacity, which is why `c_rem` appears in the `price_ratio` primitive.
4. M. Gasse et al. (2019). *Exact combinatorial optimization with graph convolutional neural
   networks.* NeurIPS. Source of the variable/constraint bipartite state representation.
5. Y.-D. Kwon et al. (2021). *Matrix encoding networks for neural combinatorial optimization
   (MatNet).* NeurIPS. Source of the idea of injecting matrix values into row/column
   cross-attention scores, used here for the edge-biased item↔resource attention.
6. W. Kool, H. van Hoof, M. Welling (2019). *Attention, learn to solve routing problems!*
   ICLR. Transformer encoder with REINFORCE.
7. Y.-D. Kwon et al. (2020). *POMO: Policy optimization with multiple optima.* NeurIPS. Shared
   multi-rollout baseline.
8. F. Berto et al. (2023). *RL4CO: an extensive RL for CO benchmark.* The environment API
   follows the same structure, so it can be ported to an RL4CO `MKPEnv`.
9. H.R. Lourenço, O. Martin, T. Stützle (2003). *Iterated local search.* Basis of the
   optional ILS refinement.

These parts are new in this project: price-and-primitive hierarchical control with a
parameter-free weak decoder, loss-free rejection in place of feasibility masking, and
per-step edge-biased dual-axis attention over the residual problem.
