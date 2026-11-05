# T1-P 结论与下一步任务 T3-a：执行器基函数扫描（固定臂，不训练）

> 日期：2026-11-05。指导方：Claude。执行方：Codex。
> 上游：[T1R_RESULTS_AND_T1P_GUIDANCE.md](T1R_RESULTS_AND_T1P_GUIDANCE.md)（T1-P 任务与 §4 门控表）、[WEAK_DECODER_T1P_EXECUTION.md](WEAK_DECODER_T1P_EXECUTION.md)（执行报告）。
> 研究边界仍以 [RESEARCH_HANDOFF_FROM_OLD_PROJECT.md](RESEARCH_HANDOFF_FROM_OLD_PROJECT.md) 第 2 节和 [CODE_GUIDANCE_WEAK_DECODER_MIGRATION.md](CODE_GUIDANCE_WEAK_DECODER_MIGRATION.md) §1 为准。用户决定：训练实验只做单种子（seed 1234），减少非核心验证。

**一句话决策：** G0+G1 上的 θ 学习到此为止，作为单种子 T1 结论收尾。T1-P 相对 fixed 仍为零增益；新增诊断 D1 进一步表明，B2 的 0.63 pp 是 greedy 在 θ 上“锯齿形景观”带来的**抽签式搜索收益**，任何输出单个 θ 的策略都无法摊销。G0+G1 中真正可摊销的余量只有 ≤0.1 pp，学习已经接近这个上限。

- 下一步不训练。先做 **T3-a**：在固定臂上扫描执行器基函数，包括 G3 的残余 LP 锚点和 x^LP 基函数；G2 的代码已经写好，顺带扫描。同时用交叉拟合估计“按实例类别条件化”的可摊销余量。
- 只有当某个新系数的可摊销余量 ≥0.1 pp 时，才进入 **T3-b** θ-PG 训练（代码本轮预先写好）。

---

## 1. T1-P 结果（seed 1234，Chu 180，greedy）

材料：`runs/t1p_{static,dynamic_k8}_s1234/`、`results/20261105_133820_726705/`。

### 1.1 训练

| 运行 | 每 epoch 用时 | 验证集 epoch −1 | epoch 0 | 最好（所在 epoch） | frac_better | adv_zero_frac |
|---|---:|---:|---:|---:|---:|---:|
| static | 约 155 s | 1.077 | 0.949 | 0.899（4） | 0.35 → 0.30–0.33 | ≤0.007 |
| dynamic k=8 | 约 285 s | 1.077 | 0.884 | 0.855（8） | 0.355–0.361，全程持平 | ≤0.0012 |

- 10 个 epoch 全部跑满，没有早停；dynamic 的成本也远低于 20 分钟/epoch 的上限。
- 验证集上的改善**几乎全部发生在 epoch 0**（250 步，仍处于 warmup 期），此后只在噪声范围内来回波动。

### 1.2 Chu greedy

| 臂 | gap % | p90 | worst | 与 fixed 的配对差 pp | 胜/平/负 | p |
|---|---:|---:|---:|---:|---|---:|
| fixed | 0.611 | 1.23 | 3.56 | — | — | — |
| learned_static | 0.584 | 1.33 | 2.97 | −0.026 | 84/9/87 | 0.97 |
| learned_dynamic (k=8) | 0.575 | 1.20 | 3.10 | −0.036 | 87/6/87 | 0.99 |

- dynamic 与 static 的配对差为 −0.009 pp，p=0.64。
- 分层中最大的点估计在 N=100：dynamic −0.125 pp（p=0.31）、static −0.058 pp。N=250 和 N=500 都约为 0。M=10 层 dynamic −0.074 pp（p=0.68）。所有分层都不显著。

**学到的 θ 是什么：**
- learned_static 在 180 个实例上 α∈[1.098, 1.100]，|u| 的均值在 0.001–0.06 之间，**等于一个全局常数**（α≈1.1，u≈0）。
- learned_dynamic 的 α∈[1.074, 1.112]，|u| 的均值为 0.007–0.17（M=10 时更大），实例间和步间的变化都很小。

### 1.3 D1：θ\* 是平滑的盆地，还是孤立的“幸运点”？（Claude 已执行）

`scripts/t1p_lottery_check.py`，输出见 `results/t1p_lottery_check/output.txt`。数据为合成验证集 128 例，θ\* 取自 B2 的 `best_theta.json`。在激活前空间加 32 次高斯扰动，scale=1 即 θ-PG 的 σ。

| 评测点 | 平均 gap % |
|---|---:|
| fixed（u=0，α=1） | 1.079 |
| θ\*（B2，每实例 512 次搜索） | **0.450** |
| θ\* + 0.25σ 扰动，32 次均值 | 1.047（增益只保留 **5%**） |
| θ\* + 1σ 扰动，32 次均值 | 1.568（比 fixed 还差） |
| fixed + 1σ 扰动，32 次均值 | 1.267 |
| fixed + 1σ 扰动，**32 次中最好的一次** | **0.321**（优于 θ\*） |

按容量比分层，结论在三层中都一致（见输出文件）。

### 1.4 B1 再分析：按类别条件化的常数 α

数据为已有的 `alpha_scan.csv`，按 M×容量比分层，取每层最优的常数 α：
- 合成数据：M=10 偏好 α=1.25，各层分别改善 0.07 / 0.24 / 0.07 pp；M=5 基本偏好 α=1.0。
- Chu N=100：M=10 的三层分别为 −0.10 / +0.13 / −0.14 pp，与合成数据不一致；M=5 全部偏好 α=1.0。

### 1.5 结论

- **T1P-1（单种子 T1 结论）：** 在 G0+G1 接口下，两种训练信号（item-PG、θ-PG）和两种更新方式（static、dynamic）都没有带来相对 fixed 的显著增益。Chu greedy 的配对差为 −0.03 pp，p>0.9。学到的 θ 坍缩为一个接近 fixed 的全局常数。
- **T1P-2（机制，B 级，证据较强）：** 单轨迹 greedy 的回报 R(θ) 关于 θ 是高度不连续的锯齿形景观。
  - B2 的每实例余量是**搜索收益**：在 θ\* 附近 0.25σ 的邻域内，增益就几乎全部消失；而在 fixed 附近随机取 32 次再挑最好的，结果比 θ\* 还好。
  - 一个输出单个 θ 的策略只能优化平滑后的期望回报 E_σ[R(θ)]，这个期望的最优点就在全局常数附近。这同时解释了 T1-R/T1-P 的零增益、θ 坍缩为常数，以及 `frac_better` 始终稳定在约 0.35。
- **T1P-3：** G0+G1 中**可被摊销的余量**（按实例类别条件化的常数 θ）不超过约 0.1 pp，学习得到的 −0.03 至 −0.07 pp 已经接近这个上限。继续调 σ、增加预算或改训练算法，在这个接口上都没有意义。
- **门控判定：** 对应 T1-P 指导 §4 的第 3 行（两臂 \|Δ\|<0.05，frac_better 高）。第 2 行（“验证集有改善，调整一次”）不采用，因为改善发生在 epoch 0、来自全局常数的平移，增加预算或调整 σ 无法产生实例自适应。
- **我原先门控设计的不足：** `frac_better` 无法区分“余量平滑但策略预测不了”和“余量本身是抽签”。D1 补上了这一判别，结果是后者。因此 T3 的立论要调整：
  - **新基函数的价值首先体现在固定臂上**，即执行器本身变强；
  - 学习只在“最优系数随可观测的实例类别系统性变化”时才有用武之地，这需要先测量，而不是直接训练。

### 1.6 对研究叙事的影响（需要用户知晓）

“学习 θ 来改进单条 greedy”在 G0+G1 上已经连续两次得到零增益，而且现在有机制性解释。如果 T3-a 显示 G2/G3 的可摊销余量同样 <0.1 pp，C1 中“学习优于固定”的主张就需要改写，可以考虑两个方向：
- 学习只负责少数随类别变化的系数（例如 η，旧项目 v6 中 η 按 M 学到 0.75 / 1.42）；
- 承认单轨迹余量主要来自搜索，把 θ 空间多起点作为一个单独命名的阶段来研究。D1 中 fixed 附近 best-of-32 的结果为 0.321，而 β0=8 下逐物品采样 best-of-16 为 0.380（A3），两者可以在相同轨迹数下比较。

这个选择留到 T3-a 结果出来之后再做，本轮不实现 θ 多起点。

---

## 2. T3-a：执行器基函数扫描（本轮核心，固定臂，不训练）

### 2.1 研究定位

- **Q-6 / Q-2：** 残余 LP 的锚点通路和评分通路，各自对执行器的贡献是多少；
- **Q-4：** 最小充分字典；
- 新增“可摊销余量”度量，用于决定是否训练（替代 B2 那种会高估的每实例搜索）。

### 2.2 执行器定义（在现有公式上增加两个开关）

```
s_i = α·log p̃_i − log Σ_j λ_j · r_ij/(c_rem_j+1)  − γ·blocked_i  + η·log(x^LP_i + 0.01)
λ_j = (base_j + floor·mean(base))·exp(u_j)
base = mu（root，现状）  或  mu_res（residual：当前残余问题的 LP 对偶，归一化在 c_rem 尺度上）
```

- 采用 residual 锚点时，mu_res 是“每单位 c_rem 比例”的价格。因此 `mu_res_j·r_ij/c_rem_j` 恰好等于原始对偶价格乘以 r_ij，**不要另外再除一次 c_rem**，沿用现有的代码路径，只替换 base 即可。
- `x^LP = sigmoid(red/τ_x)`，其中 τ_x=1e−3，即 lp_dual 末段的平滑温度；不可行或已处理的物品取 0。
  - `lp_source=root`：在 t=0 由根 mu 计算一次，之后保持不变（每步零 LP 成本）；
  - `lp_source=residual`：每隔 `lp_interval` 步按残余问题重算。

### 2.3 文件级改动

**`mkp_config.py`**
- `ModelConfig.theta_groups` 允许包含 `"lp"`。
- 新增 `lp_source: root|residual`（默认 `root`）、`lp_interval: int=8`、`lp_eta_init: float=0.5`、`alpha_init: float=1.0`。
- `price_anchor` 放开 `residual`。
- `EnvConfig` 新增 `residual_lp_iters=30`、`residual_lp_lr=0.02`。
- `TrainConfig` 新增 `theta_pg_sigma_eta=0.3`。
- 所有新字段都写入协议 SHA。旧配置缺少这些字段时取默认值，并且行为与 `4d3c376` 逐位一致。

**`mkp_env.py`**
- `lp_dual(..., mu_init=None, item_mask=None, scale=None)`：
  - 三个参数都为 None 时，行为逐位不变；
  - `item_mask` 外的物品不进入 red 和梯度项；
  - `scale` 固定为整实例的 `p.mean`，以保证热启动时的价格可比。
- 新增 `MKPEnv.refresh_lp(st)`：
  - c 用 `c_rem.clamp_min(1)`，mask 用 `st.fits`；
  - 热启动 `mu_init = mu_res_prev · c_rem_new / c_rem_prev`，即保持原始单位价格不变；
  - 写回 `st.mu_res`、`st.x_lp`。
- 在 t=0 时 `mu_res = mu`，不重复求解。
- **任何 μ≥0 都给出合法上界**；热启动后要按新的残余问题重新计算 dual_value（见 main_v1 `3c7391c` 的教训）。
- `MKPState` 新增 `mu_res`、`x_lp` 两个可选字段。

**`mkp_Model.py`**
- 在 `forward` 主循环中，当 `t>0 且 t % lp_interval == 0` 时调用 `env.refresh_lp(st)`，所有臂（包括 fixed）都调用，且放在决策和打分之前。
- `WeakDecoder.raw_scores` **只读取** `st.mu_res` / `st.x_lp`，不在内部求解 LP。
- **消除重复计算：** 目前日志统计会再调用一次 `raw_scores`，这会让 blocked_mass 和 LP 基函数各算两次。改为由 `__call__` 一并返回原始分数的 std，供统计使用。
- `fixed_theta` 和 `theta_override` 支持 `eta`；fixed 臂取 `α=alpha_init`、`γ=block_gamma_init`、`η=lp_eta_init`。
- 控制器的 `scalar_head` 支持 `lp`：`η=softplus(h)`，偏置初始化为 softplus⁻¹(η0)；α 的偏置改为由 `alpha_init` 反解。`theta_from_pre` 和 θ-PG 噪声加入 `eta_pre`。**本轮只写好代码，不训练。**

**`scripts/run_t3a_scan.py`（新建）**：只跑 greedy，不训练，见 §2.4。

### 2.4 扫描网格（16 个固定臂配置，一次运行完成）

| 编号 | 锚点 | x^LP 来源 | η | γ | lp_interval |
|---|---|---|---|---|---|
| F0 | root | — | — | — | — |
| A | residual | — | — | — | 8 |
| R(η) | root | root | 0.25 / 0.5 / 1 / 2 | — | — |
| S(η) | root | residual | 0.25 / 0.5 / 1 / 2 | — | 8 |
| AS | residual | residual | S 中按合成数据选出的最好 η | — | 8 |
| K1 | {A, S\*, AS} 中最好的一个 | 同左 | 同左 | — | **1** |
| G(γ) | root | — | — | 0.1 / 0.3 / 1 | — |
| BEST | G3 中最好的配置 + G 中最好的 γ | | | | |

α=1、u=0 保持不变。F0 必须逐实例复现 Chu greedy 的 0.611%。

**数据：**
1. 合成验证集 N=100，128 例，以 MILP 为参考（复用缓存）。**选择只在这一项上做。**
2. 合成 N=250 和 N=500，各 64 例（M=5/10 各 32 例，固定新种子，三种容量比）。不做 MILP，只报告**相对 F0 的配对目标值差**（换算成占 LP UB 的 pp），用来检查规模外推是否有害。
3. Chu 180，greedy，只作确认，不参与选择。

**每个配置报告：**
- 三个数据集的平均 gap；
- 相对 F0 的配对差、胜/平/负和 Wilcoxon p；
- 按 M×容量比分层；
- 每个实例的 LP 耗时和构造耗时，按 N 分列，N=500 时另报显存峰值。

### 2.5 可摊销余量（交叉拟合，用于决定是否训练）

只在合成 N=100 验证集上计算，按 M×容量比划分为 6 层：
- 在 η 网格（R 或 S 中较好的一族）、γ 网格以及 α∈{1.0, 1.25} 上，用**一半实例**选出每层最优的常数，在**另一半实例**上评估，然后交换两半再做一次，取平均（固定的二折交叉拟合）。
- 报告“分层最优常数”相对“全局最优常数”的配对差 pp 和 p 值，三个系数分开报告。α 只需在 F0 和最终的 BEST 两个配置下计算。
- **这就是对“策略最多能学到多少”的上界估计。** 它排除了 B2 中的抽签效应，只计入随可观测类别系统性变化的部分。

### 2.6 必要测试（只加这 3 项）

1. 默认配置（不含 lp、root 锚点、不含 block）下，fixed 和 T1-P 两个 checkpoint 在若干 Chu 实例上的 greedy 结果，与 `4d3c376` 逐位一致（复用现有测试和工具）。
2. 残余 LP 的正确性（小实例，N=30，M=5，取若干条随机轨迹的中间状态）：
   - UB ≥ HiGHS 残余 LP 值 − 1e−6（相对）；
   - UB 与 HiGHS 值的差距 ≤0.5%；
   - 30 次迭代的热启动结果与 300 次迭代的冷启动结果，UB 相差 <0.5%。
   - 如果做不到，增加 `residual_lp_iters` 并报告，不要放宽阈值。
3. 每个构造步中，LP 刷新、blocked_mass 和 raw_scores 各只计算一次（用调用计数检查）。

---

## 3. T3-b：θ-PG 训练（代码本轮写好；是否运行由 Claude 按 §4 判定）

- 配置从 `runs/t1p_dynamic_k8_s1234/config.json` 继承：
  - `theta_groups` 为 `["price","alpha"]` 加上 T3-a 中被保留的组（`lp`，可能还有 `block`）；
  - `theta_interval = lp_interval = 8`；
  - 初值（`alpha_init`、`lp_eta_init`、`block_gamma_init`）取 T3-a 选出的全局最优常数，因此 untrained 等价于新的 fixed；
  - 10×250，B64，K16，seed 1234，只训练 dynamic。
- **评测：** 只做 Chu greedy，两个臂为 `fixed`（新执行器）和 `learned_dynamic`。T1-P 的结果作为历史参照，不重跑。
- 本轮只需让 `scripts/run_t3b.py` 能生成配置；`--train` 由用户在门控通过后执行。

## 4. 门控（Claude 根据回报判定）

| T3-a 结果 | 下一步 |
|---|---|
| BEST 在合成 N=100 上比 F0 好 ≥0.1 pp（p<0.05），且 N=250/500 上不变差 | 把 BEST 采纳为新的执行器，Chu 结果作为 T3 的执行器结论（Q-6）。再看 §2.5： |
| └ 某个系数的可摊销余量 ≥0.1 pp | 运行 T3-b |
| └ 所有系数的可摊销余量都 <0.1 pp | 不训练。由 Claude 和用户一起确定 §1.6 的叙事走向 |
| 没有任何配置比 F0 好 0.1 pp | LP 基函数无效。停在这里回报，由 Claude 重新评估字典（Q-4） |

## 5. Codex 回报要求（精简）

1. 先单独提交 D1 的脚本和输出（`scripts/t1p_lottery_check.py`、`results/t1p_lottery_check/`，均由 Claude 生成，不要改数字）和本指导，再在其后实现 T3-a/T3-b。commit 和 diff 都要列出。
2. §2.6 的 3 项测试结果。
3. T3-a 的汇总表：16 个配置 × 三个数据集；相对 F0 的配对表；Chu 按 M×容量比分层；LP 和构造的耗时，N=500 时的显存。
4. §2.5 的交叉拟合表。
5. 偏离计划之处和未验证的假设。不要据此自行启动 T3-b 训练。
