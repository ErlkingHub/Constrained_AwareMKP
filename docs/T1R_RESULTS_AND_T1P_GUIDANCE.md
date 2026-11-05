# T1-R 结论与下一步任务 T1-P：θ 空间策略梯度

> **后续（2026-11-05，Claude）：** T1-P 结果与下一步 T3-a 见 [T1P_RESULTS_AND_T3_GUIDANCE.md](T1P_RESULTS_AND_T3_GUIDANCE.md)。D1 诊断表明，§1.3 中 B2 的余量属于搜索（抽签）收益，不能被单个 θ 摊销；§4 中 `frac_better` 判据的局限也在该文 §1.5 说明。

> 日期：2026-11-05。指导方：Claude。执行方：Codex。
> 上游：[T0_T1_ANALYSIS_AND_T1R_GUIDANCE.md](T0_T1_ANALYSIS_AND_T1R_GUIDANCE.md)（T1-R 任务）、[WEAK_DECODER_T1_EXECUTION.md](WEAK_DECODER_T1_EXECUTION.md)（执行报告）。
> 研究边界仍以 [RESEARCH_HANDOFF_FROM_OLD_PROJECT.md](RESEARCH_HANDOFF_FROM_OLD_PROJECT.md) 第 2 节为准。用户决定：**后续训练实验只做单种子（seed 1234）**；减少非核心验证，优先推进研究。

**一句话决策：** 修复锐度问题后，学习策略不再崩坏，但相对 fixed 的增益在统计上为零。与此同时，(u, α) 接口在 greedy 下还有约 0.63 pp 的静态可达余量。所以**瓶颈在训练信号，不在接口的表达能力**。

- 不做 T1 正式实验（单种子也不做），因为目前没有值得固化的学习增益。
- 不进入 T2，因为现有维度的余量都还没用上，再加维度没有意义。
- 下一步是 **T1-P：把强化学习的动作从“选物品”改为“选 θ”**，执行器在训练和推理时都用确定性 greedy。同时**预先写好 T2 的代码**，门控通过后直接训练。

---

## 1. T1-R 结果与结论

所用材料：`results/t1r_greedy_regression/`、`results/t1r_beta_calib/`、`results/t1r_headroom{,_chu}/`、`runs/t1r_{dynamic,static}_s1234/`、`results/20261104_151032_501872/`（Chu 评测）、`results/t1r_profile/`。

### 1.1 工程验收：全部通过

- **A1 回归：** fixed、旧 dynamic、旧 static 三臂在 Chu 180 个实例上，标准化前后 greedy 结果逐实例一致（540/540 未改变）。z-score 不改变 greedy，已得到证实。
- **A3 β 校准（合成验证集 128 例）：**

| β | greedy | sample16 均值 | best-of-16 | 每步熵 |
|---:|---:|---:|---:|---:|
| 4 | 1.079 | 1.670 | 0.597 | 2.21 |
| **8** | 1.079 | 1.226 | **0.380** | 1.52 |
| 16 | 1.079 | 1.123 | 0.452 | 0.96 |

  选定 **β0 = 8**。
- **附带发现：** fixed 臂在 β0 下 Chu 的 sample64 gap 为 **0.213%**，已经优于 main_v0 学习策略在 sample256 下的 0.233%（T0 中的 a）。多采样阶段的收益主要来自温度校准，而不是学习。这进一步支持“主表只报告 greedy”（P-4）。

### 1.2 B1：常数 α 扫描（u=0）

| α | 0.5 | 0.75 | 1.0 | 1.25 | 1.5 | 2.0 | 2.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 合成验证集 greedy gap | 4.76 | 2.18 | **1.08** | **1.08** | 1.59 | 2.85 | 3.91 |
| Chu N=100 greedy gap | 4.83 | 2.25 | **1.05** | 1.18 | 1.66 | 2.94 | 4.06 |

- α=2.5 比 α=1 差约 2.8 pp。这**证实了 T1 失败的机制**：α 顶到上界、被当作锐度使用，从而破坏了排序。
- 全局常数 α 的最优点就在 1 附近，调常数 α 没有额外收益。

### 1.3 B2：逐实例静态 θ 搜索（每个实例 512 次 greedy 评估）

| 数据 | fixed greedy | 搜索可达的 greedy | 配对差（均值 / 中位数） | 改善 >0.1 pp 的实例比例 |
|---|---:|---:|---:|---:|
| 合成验证集，128 例 | 1.079 | **0.450** | −0.629 / −0.501 pp | 79% |
| Chu N=100，60 例（诊断） | 1.051 | **0.434** | −0.617 / −0.414 pp | 82% |

- 分层：余量在紧容量（α=0.25）时最大，为 −1.0 至 −1.2 pp；在 α=0.75 时约为 −0.3 pp。
- 最佳 θ 的形态：α 的中位数为 1.0（80% 落在 [1.0, 1.25]），\|u\| 的中位数为 0.16，几乎从不碰到边界（0.2%）。也就是说，**有用的 θ 在初始化附近的小邻域内，而且因实例而异。**
- **解释边界：**
  - 这是静态 θ 可达余量的下界，说明接口的表达能力足够（Q-3：至少静态时可达）。
  - 但“每个实例取 512 次中最好的一次”本身就是一种搜索，带有选择效应。它**不能证明这份余量能被一个策略在实例之间摊销（amortize）**。下面的 T1-P 正是检验这一点的直接实验。

### 1.4 门控训练与 Chu 评测（seed 1234，greedy）

| 臂 | Chu greedy | p90 | worst | 合成验证集（训练前 → 最好） |
|---|---:|---:|---:|---|
| fixed | 0.611 | 1.23 | 3.56 | — |
| untrained | 1.233 | 2.47 | 4.74 | — |
| learned_static（10 ep，最好在 step 750） | 0.573 | 1.30 | 3.25 | 1.085 → 0.905 |
| learned_dynamic（**只训练了 250 步**） | 0.572 | 1.26 | 4.29 | 1.085 → 1.012 |

| 配对比较（Chu greedy） | 均值 pp | 胜/平/负 | p |
|---|---:|---|---:|
| learned_dynamic − fixed | −0.038 | 85/13/82 | 0.17 |
| learned_static − fixed | −0.038 | 86/4/90 | 0.79 |
| learned_dynamic − learned_static | −0.001 | 64/65/51 | 0.19 |

各分层（M、N、α）都不显著。最大的点估计是 dynamic 在 α=0.25 层的 −0.10 pp（p=0.16）。sample64 阶段三个臂也没有差别。

**训练动态：**
- static 臂的 θ 几乎没动：\|u\| 均值约 0.10，α 均值约 1.06；在 10 个 epoch 中，训练 gapUB 一直平坦在 1.44–1.66。
- 合成验证集从第 2 个 epoch 起就在 0.90–1.00 之间来回波动，没有改善的趋势。
- dynamic 臂只有 epoch 0 的记录，且验证结果优于训练前，没有触发早停，应当是被人为中止的。250 步全部处在 lr 的 warmup 期内，**所以 dynamic 臂的结果没有信息量。**

**结论：**

- **T1R-1：锐度修复有效。** α 和 u 不再触界（触界比例为 0），训练后不再比初始化差，崩坏问题已解决。
- **T1R-2：逐物品采样的 REINFORCE 不能把 B2 揭示的余量变成学习增益。** 余量约为 −0.63 pp，学到的只有 −0.04 pp，且不显著。我的判断（C 级，证据较强）如下：
  - **信用分配错位。** 在 β0=8 时，每一步对物品采样都会引入大量与 θ 无关的随机性，而 θ 的梯度只能经由逐物品的 log π 间接传递。“这条轨迹好”的功劳主要被分给了物品层面的随机选择，而不是 θ。
  - **训练目标与评测目标不一致。** 训练优化的是“采样轨迹的期望回报”，评测看的是确定性 greedy。B2 表明，有用的 θ 改动只在小邻域内（\|u\|≈0.16），而这些改动对采样期望回报的影响被采样噪声淹没了。
- **T1R-3：dynamic 每步重新计算 θ 的成本过高。** 剖析显示，每个构造步都要对 B·K=1024 条轨迹各跑一次控制器，梯度检查点又让前向计算翻倍（N=150 时，剖析器统计的单个训练步 CPU 总时间约 27 秒，其中反向约 11 秒、控制器前向约 4.4 秒）。在 1 小时/epoch 的成本下，按步更新的 dynamic 实际上无法训练足够的步数。

### 1.5 回答用户的问题

| 问题 | 回答 | 理由 |
|---|---|---|
| 做 T1 正式实验（单种子）吗？ | **现在不做** | 学习增量为 −0.04 pp 且不显著，没有可以固化的结果。先换训练信号；T1-P 通过门控后，它的单种子结果直接作为 T1 的结论 |
| 进入 T2 吗？ | **现在不进入；但本轮先写好 T2 的代码** | 现有 M+1 维已有约 0.63 pp 的余量没有用上，瓶颈在训练信号。在 T1-P 证明“θ 可以被学到”之前，加 G2 只会多一个学不动的维度。门控通过后可以立刻训练 T2 |

---

## 2. T1-P：θ 空间策略梯度（本轮核心任务）

### 2.1 研究定位

- **问题：** Q-1、Q-3 的后半部分（B2 揭示的余量能否被摊销）；I-6（θ 必须有明确的训练路径）；C1（在启发式系数空间中做控制）。
- **做法：** 把 MDP 的动作从“选哪个物品”改为“这一时刻的 θ 取什么值”。
  - 控制器输出一个高斯分布的均值，从中采样 θ；
  - 执行器在训练和推理时**都用确定性 greedy**；
  - 回报为 greedy 解的 profit/UB；
  - 梯度只经由 θ 的高斯得分函数传递。
- **这样做直接对应主路线的叙事：** 学习只负责 θ，执行器透明且确定。它同时消除了 T1R-2 指出的两个问题：
  - 探索只发生在 θ 上（M+1 维），不再混入逐物品的随机性；
  - 训练优化的就是评测指标本身（greedy 回报）。
- **它也是 B2 的摊销版：** B2 在每个实例上对 θ 做随机搜索；T1-P 用一个网络在实例之间学会 θ 的改进方向。

### 2.2 估计器（精确定义）

控制器的输出改为**激活前的值**：u 的 tanh 前 `v ∈ R^M`，α 的 sigmoid 前 `z ∈ R`。在每个决策点：

```
v' = v + σ_u · ε_u,   ε_u ~ N(0, I_M)
z' = z + σ_α · ε_α,   ε_α ~ N(0, 1)
u  = u_max·tanh(v') − mean(u_max·tanh(v'))
α  = 0.5 + 2·sigmoid(z')
log q = −½‖(v'−v)/σ_u‖² − ½((z'−z)/σ_α)²       # v'、z' 视为常数（detach），梯度只经 v、z
```

- **一条轨迹：** 在各决策点用采样得到的 θ，执行器每步取 argmax（z-score 标准化后的分数，hard_mask），最终 R = profit/UB。
- **损失：** `L = −mean_{b,k}[ A_{b,k} · Σ_{决策点} log q ]`，其中 `A = R − mean_K(R)`，保留现有的 `adv_normalize`。不加熵项（σ 固定）。
- **对偶采样：** K=16 条轨迹组成 8 对 (ε, −ε)。只对每对的第一个决策点做对偶；之后状态已经分叉，各点独立采样即可。
- **均值轨迹（只做日志，不进入损失）：** 每个实例额外跑 1 条无噪声的 greedy，记录：
  - `R_mean`；
  - `frac_better = mean_k 1[R_k > R_mean]`，即 θ 邻域里还有多少可改进的空间，这是 B2 余量的在线版本；
  - `adv_zero_frac`：同一实例 K 条轨迹的回报完全相同的比例。

**σ 的默认值**，按 B2 最佳 θ 的形态（合成数据）设定：
- `σ_u = 0.05`（tanh 前；u_max=3，在 v≈0 处 u 的标准差约 0.15，对应最佳 θ 中 \|u\| 的中位数 0.16）；
- `σ_α = 0.3`（sigmoid 前；在 α=1 处 α 的标准差约 0.11，覆盖 [1.0, 1.25]）。

**两种 θ 更新方式：**

| 配置 | 决策点 | 控制器调用 |
|---|---|---|
| `theta_update=static` | 只有 t=0 | 对 B 个实例**各调用一次**，再把均值复制 K 份后加噪声。不要像现在这样对 B·K 个相同的状态重复调用 |
| `theta_update=step` + `theta_interval=k` | t = 0, k, 2k, …（默认 k=8；θ 在两次决策之间保持不变） | 只在决策点对 B·K 条轨迹调用，成本约降到原来的 1/k |

- `theta_interval` 属于方法定义（I-4 中“明确定义的决策点”），**推理时必须使用与训练相同的 k**，并且要写入 checkpoint 的配置和评测协议的 SHA。
- 旧路径 `theta_interval=1` 保持现有行为不变。

### 2.3 文件级改动

**`mkp_config.py`**
- `TrainConfig.signal`：取值 `item_pg | theta_pg`。新运行默认 `theta_pg`；旧配置在没有该字段时视为 `item_pg`，保证旧 checkpoint 可以复现。
- 新增 `TrainConfig.theta_pg_sigma_u=0.05`、`theta_pg_sigma_alpha=0.3`、`theta_pg_antithetic=true`。
- 新增 `ModelConfig.theta_interval=1`（T1-P 的 dynamic 设为 8）。

**`mkp_Model.py`**
- `StrategyController.forward`：额外返回 `u_pre=v` 和 `alpha_pre=z`。把变换抽成一个纯函数 `theta_from_pre(v, z, u_max)`，供均值路径和噪声路径共用，保证二者数学一致。
- `ConstrainedMKPPolicy.forward`：新增参数 `theta_noise: Optional[dict]`，包含 sigma_u、sigma_alpha、antithetic 和 generator。
  - 非 None 时，强制 `decode="greedy"`，在决策点采样 θ，并累加 `theta_logp`（形状 B·K）后返回。
  - static 模式下，控制器只在 B 个实例上运行一次。
  - 支持 `theta_interval`：在两次决策之间复用上一个 θ（训练和推理都适用）。
  - trace 中记录每个决策点的 θ。
- `WeakDecoder` 不改。

**`mkp_Trainer.py`**
- `train_step` 按 `tc.signal` 分支。`theta_pg` 分支：
  - 先跑一次 `model(..., num_samples=K, theta_noise=...)`；
  - 再在 `no_grad` 下跑一次均值轨迹；
  - 按 §2.2 计算损失；
  - 日志新增 `R_mean`、`frac_better`、`adv_zero_frac`。
- 验证、checkpoint 选择、epoch −1 验证保持不变（都在均值 θ 下跑 greedy）。早停规则扩展到 static 和 dynamic 两臂。

**`Benchmark_zeroShot.py`**
- learned 臂从 checkpoint 的配置中继承 `theta_interval`，并写入 meta 和协议 SHA。不需要其他改动。

**`scripts/run_t1r.py`**（或新建 `scripts/run_t1p.py`）
- 生成两份配置：`t1p_static_s1234` 和 `t1p_dynamic_k8_s1234`。
- 从 `runs/t1r_static_s1234/config.json` 继承，只改 signal、σ、theta_interval 和 run_name。β0=8 保留（只影响 sample 阶段）。

### 2.4 必要测试（只保留下面 4 项，其余不加）

1. `theta_noise=None` 时，所有现有测试照常通过，fixed 臂和旧 checkpoint 的 greedy 不变（复用现有测试即可）。
2. 用有限差分检查 `theta_logp` 对 v、z 的梯度：在小实例上与解析式 `(v'−v)/σ²` 一致。
3. 对偶采样：每一对的首个决策点噪声互为相反数；static 模式下控制器只被调用一次（可用调用计数或 hook 检查）。
4. `theta_interval=k`：两次决策之间 θ 保持不变；k=1 时与现有的 step 模式逐步一致。

### 2.5 训练与评测（用户执行；单种子）

| 运行 | 配置 | 预算 |
|---|---|---|
| `t1p_static_s1234` | signal=theta_pg，theta_update=static | 10 epoch × 250 step，B=64，K=16，lr 等与 T1-R 相同 |
| `t1p_dynamic_k8_s1234` | signal=theta_pg，theta_update=step，theta_interval=8 | 同上 |

- 两个运行可以在两张 GPU 上并行。如果 dynamic 一个 epoch 仍超过 20 分钟，先报告剖析结果，不要擅自缩减预算。
- **Chu 评测：** 臂为 `fixed, learned_static, learned_dynamic`，**只跑 greedy**。
  - 不跑 sample、ILS，也不重跑 untrained：untrained 的 greedy 与本轮改动无关，沿用 1.233%。
  - 报告：整体与 α 分层的配对差、p90、worst，以及决策点上 θ 的统计。
- **不做的事：** β 重校准、余量诊断复跑、性能剖析（除非 dynamic 过慢）、画图（可选）、三种子。

---

## 3. T2 代码预先写好（本轮实现，门控通过后才训练）

- **目的：** 门控一旦通过，就可以立刻训练 T2，不必再等一轮实现。
- **范围：** 按迁移指导 §3 的 G2 来实现，加入 θ-PG 所需的噪声定义。

```
blocked_i = Σ_{k∈F, k≠i} p_k · 1[∃j: r_kj > c_rem_j − r_ij] / Σ_{k∈F} p_k     # 固定透明基函数，no_grad，按块计算，O(N²M)
s_i ← s_i − γ · blocked_i                                                        # 在 z-score 之前加入
γ = softplus(g),  初值 γ0 = 0.1，g0 = softplus⁻¹(0.1)
θ-PG 噪声：σ_γ = 0.3（在 softplus 之前的 g 上加）
```

- **配置：** `theta_groups` 允许包含 `"block"`，此时 θ 维度为 M+2；fixed 臂取 γ=γ0。
- **测试：** 只加一项，即在小实例上 `blocked_mass` 与朴素的双重循环实现逐元素一致。另外报告 N=500、B=1 时一次 greedy 的耗时和显存（一次性测量，不另做剖析）。
- **本轮不训练 G2，** 也不改变 T1-P 两个运行的默认 `theta_groups`（仍为 `["price","alpha"]`）。

---

## 4. 门控与下一步（由 Claude 根据回报判定）

主判据：T1-P 两臂在 Chu greedy 上相对 fixed 的配对差，辅以合成验证集相对 epoch −1 的变化。

| 结果 | 判断 | 下一步 |
|---|---|---|
| static 或 dynamic ≤ −0.10 pp 且 p<0.05 | θ 可以被学到，余量能够摊销 | 该单种子结果作为 **T1 结论**。主方法取两臂中较好的更新方式（若 dynamic 不比 static 差，就以 dynamic 为主）。**直接进入 T2 训练**：同一 signal 和更新方式，`theta_groups=["price","alpha","block"]`，单种子，同预算 |
| 两臂都在 −0.05 至 −0.10 pp 之间，或 p≥0.05 但验证集有明确改善 | 学到了一部分 | 由 Claude 根据 `frac_better` 和 `adv_zero_frac` 曲线判断是 σ 不合适还是预算不足，最多调整一次 |
| 两臂都在 \|Δ\|<0.05 pp 以内，且 `frac_better` 一直很高 | 余量在邻域里，但当前的状态表示预测不了 | 不进 G2，转到 **T3：残余 LP 通路**（x^LP 基函数和残余对偶锚点），这是信息最强的透明特征 |
| `frac_better` 很低或 `adv_zero_frac` 很高 | σ 太小，θ 的扰动改变不了 greedy 解 | 把 σ 放大 2 倍后重跑 static（成本低），不改其他设置 |

## 5. Codex 回报要求（精简版）

1. 修改清单与 commit/diff。T1-R 的改动请先提交到 `codex/t1r-sharpness-decoupling`，T1-P 在其后续提交上实现。
2. §2.4 的 4 项测试和 §3 的 1 项测试结果，以及 N=500 的耗时和显存。
3. 两个运行的配置路径、每个 epoch 的验证 gap（含 epoch −1）、`frac_better` 和 `adv_zero_frac` 的 epoch 均值、每个 epoch 的用时。
4. Chu greedy 的汇总、配对表与 α 分层。
5. 偏离计划之处和未验证的假设。单种子结果只能写成“单种子 T1 结论”，不能写成多种子显著性。
