# Research handoff to Codex — Stage T9: a compact certified θ and an LP-derived core

> **Date:** 2026-11-14 · **Guide:** Claude · **Executor:** Codex · **Decision owner:** the user
>
> **Upstream:** [T7_RESULTS_AND_T8_GUIDANCE.md](T7_RESULTS_AND_T8_GUIDANCE.md), [WEAK_DECODER_T8_EXECUTION.md](WEAK_DECODER_T8_EXECUTION.md), [GROUP_PRESENTATION_T8_PROGRESS.md](GROUP_PRESENTATION_T8_PROGRESS.md), research boundaries in [RESEARCH_HANDOFF_FROM_OLD_PROJECT.md](RESEARCH_HANDOFF_FROM_OLD_PROJECT.md) §2 and the collaboration rules in [RESEARCH_HANDOFF_TO_CLAUDE.md](RESEARCH_HANDOFF_TO_CLAUDE.md) §6–7.
>
> **Standing constraints (unchanged):** single training seed 1234; checkpoints selected on synthetic validation only; `results/`, `runs/`, `checkpoints/`, `cache/` stay out of git; the old project `D:\PhD_Doctor\Study\Codes\RL_ModelInhance` is read-only; Python is `D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe`; docs and code go in separate commits. The working tree has uncommitted T8 evaluation changes (`scripts/eval_t8.py`, `scripts/t8_data.py`, `scripts/t8_report.py`, tests, README). Keep them; do not revert or overwrite them.

---

## 0. What this stage is for

The project's first claim was "the network influences the choice only by valuing constraints, i.e. by pricing resources". Our own measurements contradict it: the learned gain does not come from the price channel. The user has decided to redesign θ around that fact and to re-examine what we take from the LP relaxation.

Decisions in this document:

1. **The learned M-dimensional price vector is removed from θ**, and so are the M(M+1)/2 resource-pair coefficients. Resource prices are computed (residual LP duals), not learned.
2. **The new θ has 5 ranking weights on a simplex plus a sampling temperature**, independent of both N and M. Each weight multiplies a basis function that is a term of an exact LP-duality identity, so the score is a mixture of a certified upper bound and a certified lower bound on an item's action value.
3. **The LP relaxation stays.** It is not replaced by a Lagrangian or surrogate relaxation. What changes is what we extract from it: reduced-cost regret, a certificate of solver accuracy, and a core set of undecided items.
4. **Nothing is trained until two cheap gates pass** (measurements in T9-0, dictionary acceptance in T9-2). A compaction-only fallback dictionary is defined in case the new bases fail.
5. **Single-trajectory greedy remains the headline metric for now.** Whether to add a search stage on the core is the user's open question; T9-5 defines the measurement that will inform it and must not start without the user's go-ahead.

Evidence grades follow the project convention (A = confirmed from source or raw results, B = implemented without a controlled comparison, C = hypothesis, E = plan).

---

## 1. Facts this guidance rests on

### 1.1 The T8 evaluation is already on disk, and the T8 gate is decided

`GROUP_PRESENTATION_T8_PROGRESS.md` §3.4 lists the evaluation as pending. It has since been run: `results/t8_best_270/` (2026-11-13). One greedy trajectory, best checkpoints, single seed.

| Set | D_port | G_port | A (item-PG) | B (PPO recipe) | L_v6 (quoted) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chu 270 (M = 5, 10, 30) | 0.680 | n/a | 0.437 | 0.426 | n/a |
| Chu 180 (M = 5, 10), matched | 0.571 | 0.514 | 0.359 | 0.365 | 0.310 |
| Synthetic N = 100 (128) | 0.965 | 0.774 | 0.524 | 0.505 | 0.479 |

- A − G_port on Chu 180: −0.155 pp (126 wins, 13 ties, 41 losses, p = 4.8e-13). B − G_port: −0.149 pp (p = 8.6e-10).
- A − B on Chu 270: +0.011 pp (p = 0.87). The two training signals are indistinguishable.
- Both are worse than the prototype: A − L_v6 = +0.049 pp (p = 7.8e-5), B − L_v6 = +0.055 pp (p = 1.2e-3). Recovery of the prototype's gain over G on Chu: A 76%, B 73%.
- Within-trajectory share of θ variance (`theta_variance.csv`): 0.94–0.99 for A's eight scalars and companion weights, 0.71–0.81 for B's.

**Gate verdict (T7 guidance §3, row 1):** A passes. Plain item-level policy gradient learns a dynamic θ on the rich dictionary, so the earlier failure on the minimal interface was about the dictionary, not the training signal (grade A for the T8 result; the comparison with T1 is unmatched, as the presentation doc already notes). B does not reach 0.33%, so the prototype is not reproduced. The final ablation matrix that the T7 guidance scheduled next is **postponed**; this stage comes first.

### 1.2 Evidence that the price channel carries no gain

| Source | Measurement | Result |
| --- | --- | --- |
| T1-R, T1-P (minimal interface) | Learned u and α against fixed | −0.04 pp or less, not significant; θ collapsed to a constant |
| T3 | Residual-LP price anchor instead of root anchor | +0.010 pp, not significant |
| T5 (prototype, constant θ) | Remove prices from G | 0.021 pp |
| T6 arm S5 (prototype) | Schedule scalars, take live prices from the controller | Within 0.022 pp of the schedule alone |
| T8 arm A (current code, `results/t8_best_270`) | Learned prices | Mean `price_std` 0.059 (prices nearly uniform); only 23% of its variance is within-trajectory |

T8 also shows which coefficients the controllers actually move in the endgame (`theta_step_changes.csv`, mean absolute step change for tau_cap ≥ 0.8):

| Field | A | B |
| --- | ---: | ---: |
| opportunity (blocking) weight | 0.123 | 0.128 |
| scarcity exponent | 0.095 | 0.067 |
| LP reduced-cost weight | 0.065 | 0.129 |
| profit exponent | 0.020 | 0.029 |
| companion weights | 0.019 | 0.049 |
| LP primal weight | 0.012 | 0.022 |
| resource prices (per entry) | 0.009 | 0.047 |
| reserve weight | 0.007 | 0.010 |
| pair interactions (per entry) | 0.003 | 0.001 |
| bottleneck weight | 0.0001 | 0.002 |

Caveat: these are raw coefficient changes, not weighted by their effect on the score, and the per-entry figures for prices and interactions are diluted by the number of entries. They are a pointer, not proof. T9-0 item (e) measures the effect directly.

### 1.3 Cost

D_port on Chu 270 takes 8.45 s per instance, of which 6.38 s is the residual LP and 0.98 s the controller (`summary.csv`). The LP solver cold-starts from y = 0 at every state (`mkp_v6_lp.py`, `solve_lp_relaxation`).

---

## 2. Answers to the two design questions

### 2.1 What does the M-dimensional part of θ do?

The θ the user describes, $(u_1,\dots,u_M,\alpha,\gamma,\eta)$, is the `g0g1` executor with `theta_groups = price, alpha, block, lp` (`mkp_Model.py`, `WeakDecoder.raw_scores`). The T8 models use the `v6` family, where the M-part is a free mean-1 price vector and a further M(M+1)/2 pair coefficients exist. The analysis covers both.

In the `g0g1` score, with cost $\kappa_i=\sum_j \lambda_j a_{ij}$, $a_{ij}=r_{ji}/(c_j^{\mathrm{rem}}+1)$ and $\lambda_j=(\mu_j+0.05\bar\mu)e^{u_j}$:

$$
\frac{\partial s_i}{\partial u_j}=-\pi_{ij},\qquad \pi_{ij}=\frac{\lambda_j a_{ij}}{\sum_l \lambda_l a_{il}},\qquad \sum_j \pi_{ij}=1 .
$$

Because $\sum_j u_j=0$, a common shift of all prices changes nothing, and the effect on item $i$ is $-\langle \pi_i-\tfrac{1}{M}\mathbf 1,\,u\rangle$. So **u can re-rank items only according to which resource dominates their cost**. It is a statement about resource identity. It could serve three purposes:

1. *Track the residual problem*, since the root duals μ go stale. A residual LP re-solve does this exactly, and T3 found no gain from it.
2. *Move from LP-optimal multipliers to integer-optimal (surrogate-dual) multipliers.* The LP dual is the optimal resource valuation for a divisible problem. What is left is indivisibility, which is a property of item combinations, not of a resource index. A resource-indexed correction is the wrong shape for it.
3. *Repair degenerate duals* (a resource with μ_j = 0 that later binds). The fixed floor and the division by remaining capacity already do this.

All three are covered by terms the score already has, which is why every measurement in §1.2 comes out at zero. In the `v6` family the same role is played more cheaply by the scarcity exponent β: with uniform prices the effective price is $(1/C_j)(C_j/c_j)^{\beta}$, a one-parameter price family driven by slack. Arm A learned exactly this: uniform prices, an active β.

**Conclusion:** the M-part is a gauge that the LP already fixes. It is removed. Resource-level valuation remains in the method, but as a computed quantity with a certificate.

### 2.2 Adjust the LP relaxation, or replace it?

Keep it, and change what we take from it.

**Why not replace it.**

- Dualising all M constraints gives a Lagrangian subproblem over $\lbrace 0,1\rbrace^N$ with the integrality property, so the Lagrangian dual bound equals the LP bound (Geoffrion 1974). Nothing is gained.
- Surrogate relaxation, or a Lagrangian that keeps one knapsack constraint, is tighter in theory, but each bound evaluation needs an exact 0–1 knapsack solve. That is expensive, hard to batch on a GPU, and brings a solver into every step.
- The looseness that matters is not removed by a tighter relaxation of the same kind. See below.

**Two kinds of looseness, which must be separated.**

1. *Solver inexactness.* The smoothed dual is minimised by 8 × 5 Newton steps and the primal is $\sigma(r/\varepsilon)$, not a vertex. Any $y\ge 0$ still gives a valid bound, so this can only weaken a certificate, never invalidate it. Its size has not been measured for the v6 solver. T9-0 item (a) measures it; T9-1 adds a per-solve certificate.
2. *Integrality gap.* $U^{\mathrm{LP}}-\mathrm{OPT}$ is worth a fraction of one item up to about one item, roughly independent of N and growing with M (C: my expectation from LP theory and from published Chu–Beasley gaps; T9-0 item (b) measures it). That is the same order as the whole learned gain ("about two items"). So the **bound value** cannot rank the decisions that matter, and a shaping potential built on it carries an unavoidable component as large as the signal.

**What the LP does give exactly.** For a state $s$ with remaining capacity $c$, feasible set $\mathcal F$, any $y\ge 0$, reduced costs $r_k=p_k-\sum_j y_j w_{kj}$ and dual value $D_y(s)=\sum_j y_j c_j+\sum_{k\in\mathcal F} r_k^{+}$:

**(I1) Gap identity.** For every feasible completion $x$ of $s$ with final slack $\sigma_j(x)$,

$$
D_y(s)-V(x)=\underbrace{\sum_j y_j\,\sigma_j(x)}_{\text{priced slack}}+\underbrace{\sum_{k:\,x_k=1} r_k^{-}}_{\text{taken against the LP}}+\underbrace{\sum_{k:\,x_k=0} r_k^{+}}_{\text{skipped against the LP}},
$$

with $r^{+}=\max(0,r)$ and $r^{-}=\max(0,-r)$. This is an equality, valid for any $y\ge 0$. Maximising profit is the same as minimising priced slack plus reduced-cost regret.

**(I2) One-step bound.** Let $s'_i$ be the state after taking $i$. With the same $y$,

$$
p_i+D_y(s'_i)=D_y(s)-r_i^{-}-\sum_{k\in\mathcal F\setminus\lbrace i\rbrace,\ k\ \text{blocked by}\ i} r_k^{+},
$$

where "blocked" means $k$ no longer fits after $i$. By weak duality the left side bounds the value of every completion through $i$, so

$$
Q^{*}(s,i)\ \le\ D_y(s)-r_i^{-}-B_i,\qquad B_i=\sum_{k\ \text{blocked by}\ i} r_k^{+}.
$$

Two consequences. The right weight for a blocked item is its positive reduced cost, not its profit (v6's `blocked` uses profit). And the per-step loss of upper bound splits exactly into own regret, blocked regret, and the gain from re-optimising the duals at $s'_i$ ("stranded priced capacity").

**(I3) Reduced-cost fixing.** From (I1), any completion that deviates from the LP rounding on item $k$ has value at most $D_y(s)-\lvert r_k\rvert$. With an incumbent value $\mathrm{LB}$, every completion at least as good as the incumbent agrees with the LP on all items with $\lvert r_k\rvert> D_y(s)-\mathrm{LB}$. This is a certificate about individual binary decisions, and it survives an inexact $y$.

**The core.** The items with small $\lvert r_k\rvert$ are the ones the LP cannot decide. This is the core of the multidimensional knapsack problem (Puchinger, Raidl and Pferschy 2006, 2010), the direct generalisation of the Balas–Zemel core of the 0–1 knapsack through the efficiency $e_k=p_k/\sum_j y_j w_{kj}$ (note $r_k>0 \iff e_k>1$). It is the right construct for "critical binary decisions", and it is derived from the LP rather than replacing it. Whether (I3) certifies a useful share of items on our instances depends on how tight the incumbent is, and I expect it to be weak for small N and large M (C). So the core enters first as a measurement, then as features and an episode factorisation, and never as an unverified hard rule.

**Changes per LP pathway.**

| Pathway | Now | T9 |
| --- | --- | --- |
| Decoder scoring | `log(x_LP + 0.01)` and clipped reduced cost with two free weights; heuristic blocking, reserve, companion, pair terms | The terms of (I1) and (I2) plus an achievable two-item lower bound (§3.2) |
| Encoder input | Residual LP features behind `ctrl_residual_lp_features` | Add the basis values and core-position features behind a new switch (§3.6) |
| Reward | Arm A: profit over root bound. Arm B: potential shaping with the residual bound | Unchanged. Log the three-way split of each step's bound loss for analysis only |
| Solver | Cold start, no accuracy measure | Per-solve certificate; optional warm start behind a flag (§3.5) |
| Baselines and metrics | Dual-ratio greedy; MILP and best-known references | Unchanged. The LP bound is still not a gap reference |

---

## 3. Design specification

### 3.1 Units and notation

Use the normalisation of `mkp_v6_lp.py` exactly. At state $s$: `scale` = mean feasible profit; $\tilde p_i=p_i/\texttt{scale}$; $a_{ij}=w_{ij}/c_j$ with $c_j$ the remaining capacity; $\tilde y_j$ = `lp.dual`; $\tilde r_i$ = `lp.reduced` $=\tilde p_i-\sum_j a_{ij}\tilde y_j$. In these units $D=\sum_j\tilde y_j+\sum_i\tilde r_i^{+}$.

### 3.2 Basis functions (family `cert`)

For every feasible item $i$, all in units of mean feasible profit:

| Name | Definition | Meaning | Origin |
| --- | --- | --- | --- |
| `sur` | $\tilde r_i^{+}$ | LP surplus: how strongly the LP wants $i$ | (I1) |
| `reg` | $\tilde r_i^{-}$ | Certified loss of bound if $i$ is taken against the LP | (I2) |
| `blk` | $\sum_{k\in\mathcal F\setminus\lbrace i\rbrace,\ \neg\mathrm{fits}(k\mid i)}\tilde r_k^{+}$ | Certified loss of bound from the items $i$ locks out | (I2) |
| `fill` | $\sum_j \tilde y_j a_{ij}=\tilde p_i-\tilde r_i$ | Priced capacity that $i$ uses, i.e. priced slack it removes | (I1) |
| `pair` | $\max_{k\in\mathcal F\setminus\lbrace i\rbrace,\ \mathrm{fits}(k\mid i)}\tilde p_k$, 0 if none | Profit of the best single partner that still fits | achievable lower bound |

$\mathrm{fits}(k\mid i)$ means $w_{kj}\le c_j-w_{ij}$ for all $j$, compared in float64 with the same convention as `WeakDecoder.opportunity_features`. `blk` and `pair` come from one chunked pairwise pass. There is no separate profit basis because $\tilde p_i=\texttt{sur}-\texttt{reg}+\texttt{fill}$; adding one would be collinear.

### 3.3 θ and the score

$$
S_i(s;\theta)=\pi_{\mathrm{sur}}\,\texttt{sur}_i-\pi_{\mathrm{reg}}\,\texttt{reg}_i-\pi_{\mathrm{blk}}\,\texttt{blk}_i+\pi_{\mathrm{fill}}\,\texttt{fill}_i+\pi_{\mathrm{pair}}\,\texttt{pair}_i,\qquad
\Pr(i\mid s)=\underset{i\in\mathcal F(s)}{\operatorname{softmax}}\bigl(S_i/\tau\bigr).
$$

| Component | Constraint | Meaning |
| --- | --- | --- |
| $\pi=(\pi_{\mathrm{sur}},\pi_{\mathrm{reg}},\pi_{\mathrm{blk}},\pi_{\mathrm{fill}},\pi_{\mathrm{pair}})$ | non-negative, sum 1 (softmax of 5 logits) | How much each certified criterion counts in this state |
| $\tau$ | bounded interval, calibrated in T9-2 | Sampling sharpness; cannot change the greedy ranking |

Six numbers for any N and any M (v6: 31 for M = 5, 76 for M = 10, 506 for M = 30). The simplex makes the ranking weights scale-free, so sharpness has no ranking parameter to leak into (the T1 failure) and lives only in τ.

Named points of the family, to be implemented as constants for tests and fixed arms:

| Label | π ∝ (sur, reg, blk, fill, pair) | Rule |
| --- | --- | --- |
| K-rc | (1, 1, 0, 0, 0) | Greedy by reduced cost |
| K-bound | (ε, 1, 1, 0, 0), ε = 0.05 | Minimise certified bound loss; surplus breaks ties |
| K-profit | (1, 1, 0, 1, 0) | Greedy by profit |
| K-pair | (1, 1, 0, 1, 1) | Greedy by best two-item value |

What is guaranteed, and what is not:

- **P1 (bracket).** For every state and feasible item, $\tilde p_i+\texttt{pair}_i\ \le\ Q^{*}(s,i)\ \le\ D-\texttt{reg}_i-\texttt{blk}_i$. K-pair is greedy on the lower bound and K-bound is greedy on the upper bound. θ chooses a point between two certified bounds.
- **P2 (LP-tight states).** Suppose $y$ is an optimal dual and the residual LP optimum is integral and strictly complementary (every item it takes has $r>0$, every other feasible item has $r<0$). Then any θ with $\pi_{\mathrm{sur}},\pi_{\mathrm{reg}}>0$ and $\pi_{\mathrm{fill}}=\pi_{\mathrm{pair}}=0$ picks only items of that optimum, and the property is preserved after the pick.
- **P3 (terminal states).** If no two feasible items fit together, K-profit is optimal. If no three do, K-pair is optimal.
- These were checked numerically by Claude on random instances with N ≤ 11 before this handoff: (I1) and (I2) hold to 1e-15 relative for arbitrary $y\ge 0$; P1 on 2,782 state-item pairs against brute-force $Q^{*}$; P3 on 97 states; P2 on 8 states. The tests in T9-1 should reproduce this inside the repository.
- **No guarantee in between.** No compact per-item rule can guarantee an optimal construction on the core; that would solve an NP-hard problem greedily. "Enables the decoder to recover optimal solutions" is therefore a measured property: at each step the set of items reachable through θ must intersect a target optimal solution (the T7 oracle). T9-2 measures it and is the acceptance test for the dictionary.

The claim this design supports, replacing "the network prices resources": **constraint valuation is computed by LP duality with certificates; the network learns how far to trust each certificate in the current residual state.** The interface contracts I-1 to I-6 hold (size independent of N, no per-item learned output, stated meaning, recomputed per state, feasibility by the decoder, gradient through the item softmax).

### 3.4 Extension slots and the stopping rule

T7 showed that a small dictionary can lose all controllability (E_min: reachable set of one item, ceiling 0.381%). If `cert` fails the gate in T9-2, add terms in this fixed order, one at a time, each with one non-negative weight joining the simplex unless stated:

1. `x_lp`: $\log(x_i^{\mathrm{LP}}+0.01)$ (a step at $r=0$).
2. `scar`: v6's $-\log\mathrm{cost}_i(\beta)$ with uniform prices; adds one weight and the exponent β.
3. `comp3`: v6's three-scale companion basis in place of `pair`.
4. `bneck`: $\max_j a_{ij}$.
5. `inter1`: $\sum_{j\le l} z_{ij}z_{il}$ with a single coefficient.

**Stopping rule (this is new and mandatory).** Every dictionary is compared with a **random-basis control**: the same dictionary plus $k$ extra per-item features that are fixed noise (i.i.d. standard normal, seeded by instance and item index, rescaled to the spread of `reg` over feasible items) with free signed weights. A real term is kept only if it lowers the oracle ceiling at least twice as much as one random basis does. If random bases reproduce the ceiling of E_full, then E_full's controllability is generic pointing power, not valuation, and that is itself a result to report.

### 3.5 LP solver changes

- **Certificate (required).** Extend `LPRelaxation` with `dual_value` (the unclamped dual value in raw units, float64), `primal_value` (value of the smoothed primal after scaling it down to feasibility), `certificate_gap = dual_value − primal_value ≥ 0`, and `scale`. Existing fields and arithmetic stay bit-identical.
- **Warm start (optional, behind `lp_warm_start`, default off).** Carry raw-unit prices $y_j\cdot\texttt{scale}/c_j$ from the previous state of the same batch, convert to the new normalisation, and start from a shortened temperature schedule. The cold path must remain the default so that D_port and G_port still reproduce exactly.

### 3.6 Controller and features

- Backbone unchanged: `StateEncoder` plus `StrategyController.representation`, same widths and depths as T8.
- New head for `theta_family="cert"`: one `nn.Linear(embed_dim, 6)` on the context vector. Weights zero-initialised, biases set from the default θ, so an untrained controller emits the default θ for any encoder weights (same guarantee as T8, same test).
- No price head and no `CombinationControlHead` in this family.
- New switch `ModelConfig.ctrl_basis_features` (default `False`; `True` in T9 runs). When on, append to the dynamic features: per item the five basis values (clipped with `feature_clip`, zero for infeasible items), the normalised rank of $\lvert\tilde r_i\rvert/\tilde p_i$ among feasible items, and a core flag for the $\min(\lvert\mathcal F\rvert, 2M+10)$ items with the smallest $\lvert\tilde r_i\rvert/\tilde p_i$; globally $\sum_j\tilde y_j/D$, the normalised certificate gap, and core size over $\lvert\mathcal F\rvert$. Principle: the controller must be able to see every criterion it weights.
- `ctrl_residual_lp_features` stays `True`, as in T8.

### 3.7 Fallback dictionary `v6c` (compaction only)

The v6 executor with `resource_prices = 1`, `resource_interactions = 0`, `bottleneck_weight = 0`, `reserve_weight = 0`. Remaining θ: profit exponent, scarcity exponent, opportunity weight, LP primal weight, LP dual weight, three companion weights, temperature (9 numbers). Implement as `ModelConfig.v6_frozen_groups` (default empty, so T8 behaviour is unchanged). It isolates "remove degrees of freedom" from "change the bases".

---

## 4. Tasks

Follow the existing convention of `WEAK_DECODER_T8_EXECUTION.md` §4: run unit tests and `--limit 2` smoke checks yourself; for long runs provide the exact commands and expected duration for the user.

### T9-0. Measurements on existing artefacts (no model change, no training)

New script `scripts/run_t9_measure.py`, output `results/t9_measure/` with `meta.json` provenance as in T8. Inputs: `results/t5_v6/instances_eval.npz`, `results/t8_best_270/rows.jsonl`, the MILP cache, the T8 checkpoints.

| Item | What to compute | Why |
| --- | --- | --- |
| (a) LP accuracy | Along the G_port trajectory (D_port for M = 30), at one state per tau_cap decile per instance, solve the residual LP exactly with `scipy.optimize.linprog` (HiGHS). Report (v6 bound − exact)/`scale`, the same relative to the exact value, the count of items with 0.01 < x < 0.99 against the exact number of fractional variables, and the primal infeasibility of the smoothed x. By decile and by (M, N). | Separates solver inexactness from integrality gap |
| (b) Integrality gap | At the root: (exact LP − reference)/reference in %, and (exact LP − reference)/mean profit in items, by class, next to the D, G, A, B gaps. | Shows how much of the bound-based signal is unavoidable |
| (c) Gap identity (I1) | With exact root duals, split the gap to the root bound into priced slack, taken-against-LP and skipped-against-LP for the D, G, A, B solutions (all sets) and for the MILP reference solution (synthetic only). Paired differences A − G, B − G, and arm − reference. | Says where learning gains and where policies still lose |
| (d) Core statistics | Root core $K_k$ = the $k$ items with the smallest $\lvert r_i\rvert/p_i$ plus the exact fractional items, for $k\in\lbrace M,\,2M,\,2M+10,\,0.1N,\,0.2N\rbrace$. Report: the share of the symmetric difference between A and G solutions, and between each arm and the reference, that lies in $K_k$; the gap of the core-restricted optimum (HiGHS MILP with all non-core variables fixed to the LP rounding, synthetic only; on Chu compare with best-known); and the share of items fixed by (I3) with incumbent = reference and with incumbent = each arm's value. | Tests whether the learned gain and the remaining error are core items, and what is lost if the LP decides everything outside the core |
| (e) Group freeze on T8 checkpoints | Add a `theta_patch` argument to `V6Policy.episode` that overwrites chosen θ fields at every step. For A and B (best checkpoints), on Chu 180 and synthetic: P1 = prices ← 1; F-mean = prices ← 1 and interactions, bottleneck, reserve ← permutation-invariant constants fitted on the checkpoint's own traces per M (for interactions: one diagonal and one off-diagonal mean); F-zero = prices ← 1 and those three groups ← 0. Report paired differences against the unpatched checkpoint and retention (G − arm)/(G − checkpoint). | Confirms on current-code models that the price channel, and which other groups, can go |

Acceptance: `report.md` with the five tables, per-instance CSVs, and a note on anything that failed to solve. Item (e) is about 6 to 8 GPU hours; give the user the commands.

### T9-1. Code

| File | Change |
| --- | --- |
| `mkp_cert_basis.py` (new) | `cert_basis(state, config)`: the five bases, rank and core flag, memoised per state like the LP memo. One chunked pairwise pass for `blk` and `pair`. |
| `mkp_cert_exec.py` (new) | `CertTheta` (weights [B, 5], temperature [B], `statistics()`), named constants of §3.3, `CertDecoder` returning the existing `CandidatePolicy` so tie-breaking (smallest index) and sampling are identical to v6. |
| `mkp_cert_model.py` (new) | `CertPolicy` with the same interface as `V6Policy` (`policy`, `episode`, `default_theta`, `exec_cfg`), reusing `TokenEncoder` and the controller backbone. PPO support is not required in this stage: raise a clear error for `signal="ppo_gae"`. |
| `mkp_v6_lp.py` | Certificate fields and optional warm start (§3.5). No change to existing outputs. |
| `mkp_v6_model.py`, `mkp_v6exec.py` | `theta_patch` for T9-0 (e); `v6_frozen_groups` for §3.7. Defaults leave T8 behaviour bit-identical. |
| `mkp_Model.py`, `mkp_config.py` | `executor` and `theta_family` accept `"cert"`; add `ctrl_basis_features`, `v6_frozen_groups`, `lp_warm_start`, and the default π and τ range. Old checkpoints must load with defaults filled in, as was done for T8. |
| `mkp_t8_train.py` | Accept `executor in ("v6", "cert")` for `signal="item_pg"`; log `CertTheta.statistics()`. |
| `scripts/run_t9.py`, `scripts/eval_t9.py`, `scripts/t9_report.py` (new) | Configuration, evaluation and report for T9 arms. Reuse `scripts/t8_data.py`. Leave `eval_t8.py` untouched. |

Tests, new file `tests/test_t9_cert.py` (add only these):

1. **(I1)** on random small instances, random feasible maximal $x$ and random $y\ge 0$: both sides agree to 1e-9 relative in float64.
2. **(I2)** for every feasible item of random states: $p_i+D_y(s'_i)$ computed by brute force on the new state equals $D_y(s)-r_i^{-}-B_i$ computed from `cert_basis`, in raw units.
3. **P1 and P3** against brute-force $Q^{*}$ on instances with N ≤ 12.
4. **Initialisation and compatibility:** the untrained `cert` controller emits the default θ for random encoder weights; default `g0g1` and `v6` paths are bit-identical (existing tests pass; the `--limit 2` D_port and G_port smoke matches `results/t8_port`).
5. **Certificate:** on sampled states, `primal_value ≤ exact LP ≤ dual_value` within 1e-6 relative.

### T9-2. Dictionary acceptance (fixed arms and oracle ceilings, no training)

Script `scripts/run_t9_dictionary.py`, output `results/t9_dictionary/`.

1. **Default θ and τ.** On a calibration set of 32 synthetic N = 100 instances per M with a new seed (not the validation set), scan the simplex with step 0.25 (70 points), refine around the best with step 0.1, cross-fitted in two folds. The winner is the default π and the constant arm K\*. Calibrate τ and its range by the method of `scripts/calibrate_beta.py`. Record everything in `meta.json`.
2. **Fixed arms, one greedy trajectory, Chu 270 and synthetic 128:** K-rc, K-bound, K-profit, K-pair, K\*, and `v6c` with G's constants. Reference rows D_port and G_port are reused from `results/t8_best_270`.
3. **Oracle ceilings, synthetic 128 with MILP targets,** using the protocol of `scripts/t7_interface.py` (`oracle_rollout`, about 2,048 sampled θ per state plus the named constants; ties among allowed items broken by the dictionary's own constant arm). Write a thin adapter so the probes run on the ported executors through a basis/score split; do not import the old project. Dictionaries: E_full and E_min (re-measured as anchors; T7 gave 0.118 / 0.177 and 0.381 / 0.369 for proven / incumbent targets), `v6c`, `cert`, each extension of §3.4 in order as needed, and the random-basis controls `cert + rand(k)` and `E_min + rand(k)` for $k\in\lbrace 2, 8, 32\rbrace$.
4. **Pointer check.** On the oracle trajectories report $\rho$ with two denominators: all feasible items, and the core set $K_{2M+10}$.

### T9-3. Training (only after the gate in §5)

One arm, `T9-A`: the chosen dictionary with the **T8-A recipe unchanged** (item-level policy gradient, K = 16, shared mean baseline, terminal reward over the root bound, entropy 0.01, 2,500 updates, same data, same validation set, same selection rule, same budget rule at update 250, seed 1234). The only differences from T8-A are `executor`, `theta_family` and `ctrl_basis_features`. Run name `t9_a_cert_s1234` (or `t9_a_v6c_s1234` if the fallback is chosen).

Evaluation with `scripts/eval_t9.py`, one greedy trajectory, Chu 270 and synthetic 128:

- Arms: D_cert (untrained), K\* (best constant on the same decoder), T9-A, plus reused rows D_port, G_port, A, B and the quoted L_v6.
- Report as in T8: mean, p90, worst; paired differences with wins, ties, losses and Wilcoxon p for T9-A against K\*, against A, against L_v6; strata by M, N, tightness; time split into LP, controller, total; θ variance decomposition and late-phase step changes; the (I1) split of every arm's gap.

### T9-4. Core factorisation at inference (existing checkpoints, no training)

Hypothesis (C): the LP can decide everything outside the core, and the controller is needed only inside it. T7 supports this for the prototype (schedule until tau_cap 0.9, then the learned controller: 95% recovery with 14.6 controller calls instead of 147.8).

Rule `prefix(δ)`: at a state with a fresh LP, let $T=\lbrace i\in\mathcal F:\ x_i^{\mathrm{LP}}\ge 0.99,\ \tilde r_i/\tilde p_i\ge\delta\rbrace$. If $T$ is non-empty, take its items in decreasing $\tilde r_i$, re-checking feasibility, without re-solving the LP and without calling the controller; then re-solve. The first time $T$ is empty the core phase starts and lasts to the end: one LP solve and one controller call per step, as now. It is still one trajectory, one item at a time.

Run for the A and B checkpoints and for G_port with $\delta\in\lbrace 0.05, 0.1, 0.2\rbrace$ on Chu 180 and synthetic. Report gap, retention (G − arm)/(G − checkpoint), controller calls, LP solves and time. Implement as an inference option in `V6Policy.episode` and `CertPolicy.episode`, default off.

### T9-5. Reference rows for the open question on single-trajectory decoding (on hold)

**Do not run without the user's go-ahead.** It adds search in the core phase, which is outside boundaries 1 and 2 of the presentation doc. Prepare nothing beyond this description unless told to.

Measurement: on the same decoder, for the constant arm K\* and the learned arm, compare (i) one greedy trajectory, (ii) a beam of width $W\in\lbrace 4, 16\rbrace$ in the core phase only, children ranked by the decoder score and beams ranked by collected profit plus the residual dual value, (iii) best of 16 and 64 samples; all with wall-clock time. The question is whether learned − constant survives at $W>1$. T0 found that it vanishes under local search.

---

## 5. Gates

Claude decides after Codex reports; the thresholds are fixed now.

| After | Outcome | Next |
| --- | --- | --- |
| T9-0 (a) | p95 of (v6 bound − exact)/`scale` ≤ 0.01 | Solver inexactness is immaterial. Keep the solver; warm start is an efficiency option only |
| T9-0 (a) | Larger than 0.01 | Warm start or a final polish becomes required before T9-2; report the size first |
| T9-0 (e) | F-mean keeps ≥ 90% of the checkpoint's gain over G for both A and B | The frozen groups are not needed for closed-loop control; compaction is supported on current-code models |
| T9-0 (e) | P1 alone loses more than 10% for either arm | Contradicts §1.2. Stop and report before any further work |
| T9-2 | ceiling(`cert`, possibly with extensions, total size ≤ 10) ≤ ceiling(E_full) + 0.05 pp, every kept term passes the random-basis rule, and K\* on Chu 180 is not worse than G_port by more than 0.10 pp | Train `cert` in T9-3 |
| T9-2 | `cert` fails but `v6c` meets the ceiling condition | Train `v6c` in T9-3; the exact bases become later single-factor swaps |
| T9-2 | Both fail | The compact-θ hypothesis is rejected at the oracle level. Do not train. Report |
| T9-2 | Random bases reproduce E_full's ceiling | Report as a finding against the "control handles" reading of T7; still proceed by the rows above |
| T9-3 | T9-A beats K\* by ≥ 0.1 pp on Chu 180 (p < 0.05), within-trajectory variance dominates, and T9-A is within 0.03 pp of A | The compact θ replaces v6 as the main dictionary; the postponed ablation matrix is run on it |
| T9-3 | T9-A is within 0.03 pp of A, but K\* is as good as T9-A | Exact bases make a constant sufficient. Learning on this decoder adds nothing measurable. Report plainly; the user decides the direction |
| T9-3 | T9-A is worse than A by more than 0.05 pp | Compactness costs performance. Report the size of the trade and return to §3.4 |
| T9-4 | Some δ keeps ≥ 95% of the gain for A and B | Core-only control is allowed as a later training arm (shorter horizon, fewer LP solves) |

---

## 6. Boundaries

- Do not change the default behaviour of the `g0g1` or `v6` paths. Every new behaviour sits behind a config field whose default reproduces T8.
- Do not edit the research-programme documents (`RESEARCH_HANDOFF_FROM_OLD_PROJECT.md`, `RESEARCH_HANDOFF_TO_CLAUDE.md`). The change of narrative in §3.3 (prices computed, trust learned) is the user's to adopt there.
- No sampling, multi-start, beam, repair or solver-chosen integers in any T9-0 to T9-4 arm. HiGHS is used only for exact LP values in measurements, for MILP references, and for the core-restricted optimum in T9-0 (d), all of which are diagnostics.
- A claim that learning helps is made only against K\* on the same decoder with the same inference budget.
- Chu is not used to choose defaults, constants, extensions or checkpoints. The calibration set in T9-2 is new synthetic data, separate from the validation set.
- If a measurement contradicts a premise of this document, stop that line and report; do not adapt the design on your own.

## 7. What to report back

1. Files changed and commit hashes (docs and code separately); proof that the old project is untouched.
2. Test results for `tests/test_t9_cert.py` and the existing suites; the D_port / G_port smoke reproduction.
3. T9-0: the five tables, with the gate quantities of §5 stated explicitly.
4. T9-2: default π and τ, fixed-arm table, oracle ceilings for every dictionary and random control, the pointer check.
5. T9-4: retention, calls, LP solves and time per δ.
6. Commands and expected durations for the user-run jobs (T9-0 (e), T9-3).
7. Deviations from this plan, anything not verified, and anything that surprised you.

## 8. References

Bibliographic details are from memory; verify before citing in a paper.

- Balas and Zemel (1980), *An algorithm for large zero-one knapsack problems*, Operations Research 28.
- Chu and Beasley (1998), *A genetic algorithm for the multidimensional knapsack problem*, Journal of Heuristics 4.
- Fréville (2004), *The multidimensional 0–1 knapsack problem: an overview*, European Journal of Operational Research 155.
- Fréville and Plateau (1994), *An efficient preprocessing procedure for the multidimensional 0–1 knapsack problem*, Discrete Applied Mathematics 49.
- Geoffrion (1974), *Lagrangean relaxation for integer programming*, Mathematical Programming Study 2.
- Ng, Harada and Russell (1999), *Policy invariance under reward transformations*, ICML.
- Pirkul (1987), *A heuristic solution procedure for the multiconstraint zero-one knapsack problem*, Naval Research Logistics 34.
- Pisinger (1997), *A minimal algorithm for the 0-1 knapsack problem*, Operations Research 45.
- Puchinger, Raidl and Pferschy (2006), *The core concept for the multidimensional knapsack problem*, EvoCOP, LNCS 3906; and (2010), *The multidimensional knapsack problem: structure and algorithms*, INFORMS Journal on Computing 22.
