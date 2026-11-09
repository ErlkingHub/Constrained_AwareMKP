# T7 结论与下一步任务 T8：把 v6 字典移植为新项目的执行器，对比两种训练信号（单轨迹）

> 日期：2026-11-09。指导方：Claude。执行方：Codex。
> 上游：[T6_RESULTS_AND_T7_GUIDANCE.md](T6_RESULTS_AND_T7_GUIDANCE.md)（T7 任务、§2 边界、§4 门控）、[WEAK_DECODER_T7_EXECUTION.md](WEAK_DECODER_T7_EXECUTION.md)（执行报告）。
> 约束：主线只做**单轨迹 greedy**；sampleK、SMS 和 ILS 冻结为参照行，本轮不运行。单种子 1234。`results/`、`runs/`、`checkpoints/`、`cache/` 不入 git。对 `D:\PhD_Doctor\Study\Codes\RL_ModelInhance`（下称“旧项目”）保持只读。Python：`D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe`。
> 用户已确认三个选择：T8 训练两个臂；一个模型同时训练 M∈{5,10}；B 臂保留续行排序辅助损失。

**一句话决策：**
- v6 学习的价值集中在**终盘（tau_cap≥0.8）、每步都要刷新、可达比例 ρ 很低的约束估值**，不是物品指针。
- E_full 接口的天花板（0.118%）远低于 L 的 0.479%，**瓶颈在学习，不在表达能力**。
- 去掉常数下“不重要”的项之后（E_min），接口**几乎没有可控性**。
- 因此 T1 失败的首要嫌疑是 **G0+G1 字典的可控性不足**。
- T8 在新项目中移植 v6 的完整字典作为执行器，然后用两种训练信号各训练一个单种子臂，对两个嫌疑做第一次单因素检验。

---

## 1. T7 结果（v6 checkpoint，只读，单轨迹 greedy）

材料：`results/t7_value/`。回收率 =(G − 臂)/(G − L)。L、G、S2 复用 T5/T6 的结果。

### 1.1 价值位置

| 臂 | 构造 | Chu gap | Chu 回收率 | 合成 gap | 合成回收率 | 每例控制器调用次数（Chu） |
|---|---|---:|---:|---:|---:|---:|
| L | v6 learned dynamic | 0.310 | 100% | 0.479 | 100% | 147.8 |
| G | 全局常数 θ | 0.514 | 0% | 0.774 | 0% | 0 |
| S2 | 按类别 × tau_cap 调度 | 0.449 | 31% | 0.687 | 29% | 0 |
| SL05 | tau_cap<0.5 用 S2，之后用 L | 0.314 | 98% | 0.500 | 93% | 74.8 |
| **SL08** | tau_cap<0.8 用 S2，之后用 L | **0.306** | **102%** | **0.451** | **110%** | 30.2 |
| SL09 | tau_cap<0.9 用 S2，之后用 L | 0.320 | 95% | 0.488 | 97% | 14.6 |
| LS08 | tau_cap<0.8 用 L，之后用 S2 | 0.469 | 22% | 0.717 | 19% | 117.6 |
| K2 | L 每 2 步更新一次 θ | 0.365 | 73% | 0.578 | 67% | 74.2 |
| K8 | L 每 8 步更新一次 θ | 0.433 | 39% | 0.724 | 17% | 18.9 |

- 按分层看，SL08 在 Chu 的 6 个 M×N 分层中回收率为 81–113%；LS08 在 −21% 至 53% 之间。
- SL08 与 LS08 的回收率之和为 1.24（Chu）和 1.29（合成），超过 1。也就是说，前期用闭环还是用调度可以互相替代，而终盘闭环无论接在哪种前期 θ 后面都能起作用。
- **学习价值几乎全部在终盘。** SL09 只在约 10% 的步上调用控制器（Chu 每例 14.6 次），就保留了 95% 的价值。
- 在终盘它**需要逐步刷新**：每 8 步更新一次就只剩 17–39%。

### 1.2 接口容量（`capacity_summary.csv`、`capacity_regions.csv`）

E_full 执行器，沿 L 的轨迹：

| 阶段（tau_cap 十分位） | 可行物品数中位数（合成 / Chu） | 局部可达集中位数（合成 / Chu） | ρ_loc 中位数（合成 / Chu） | 选择稳定度中位数（合成 / Chu） |
|---|---|---|---|---|
| d0（最早） | 98 / 473 | 15 / 15 | 0.15 / 0.06 | 0.12 / 0.06 |
| d5 | 64 / 288 | 13 / 17 | 0.20 / 0.09 | 0.08 / 0.03 |
| d9（终盘） | 30 / 137 | 6 / 11 | 0.21 / 0.09 | 0.46 / 0.20 |

- 按 ρ 分区，统计 L 与 S2 选择不同的步落在哪个区：
  - ρ<0.2：合成集 54%，Chu 93%；
  - 0.2–0.5：合成集 46%，Chu 7%；
  - ρ>0.5：只占 **0.2%**。
- **结论：终盘不是指针区。** 这否定了 T6 §2.2 中“后期可行物品少，所以 ρ→1”的预设。实际上终盘是“容量快用完、但可行物品仍然很多”的**紧约束补齐**问题，θ 在其中只能在少数极端候选之间做约束估值。这正好对应 PP-2：瓶颈随构造过程转移。
- 选择稳定度很低（中位数 0.02–0.14）：在 L 自身残差量级的扰动下，L 的选择大多会改变。这说明学习要做的是贴近边界的精细估值，也说明这项能力很难从常数或调度中得到。
- “L 与 S2 选择不同”的状态占 70–85%，但这主要是选择顺序不同（T6 显示最终解只差约 2 个物品），所以只作描述用。

### 1.3 接口天花板与字典可控性（`oracle_summary.csv`、`target_summary.csv`）

| 执行器 | 目标解 | 例数 | 接口内预言机 gap | 首次出现空交集的例数（平均 tau_cap） |
|---|---|---:|---:|---|
| E_full | 已证明的最优解 | 53 | **0.118** | 8（0.94） |
| E_full | MILP incumbent | 75 | 0.177 | 11（0.90） |
| E_min | 已证明的最优解 | 53 | 0.381 | 25（0.90） |
| E_min | MILP incumbent | 75 | 0.369 | 28（0.85） |

- 参照：合成 N100 上 L 为 0.479，G 为 0.774。
- 这个预言机只是“可达的下界”：可达集是抽样估计的，物品按 S2 顺序选取且不回溯。
- **E_full 接口远未饱和，学习还有 ≥0.3 pp 的余量（Q-3：瓶颈在学习）。**
- L 的选择有 94% 落在 x\* 中；抽样交集为空的状态只占 1.1%。
- **E_min 几乎没有可控性：** 局部可达集的中位数是 1，ρ≈0.01，天花板 0.38。
  - T5 中常数 θ 下“去掉影响 <0.05 pp”的那些项（价格、opportunity、reserve、interactions、lp_dual、bottleneck），**提供的是闭环控制需要的自由度**。
  - 因此 **Q-4 的判据要从“常数下的增益”改为“可控性 / 接口天花板”**；G-min 不能作为学习用的字典。

### 1.4 门控判定（对照 T6 指导 §4）

- **第 1 行（后期指针）不成立：** 价值确实集中在后期，但那里 ρ 很小，而且 K2 仍保留 67–73%。
- **第 2 行（各阶段的缓慢估值）不成立：** LS08 只有 19–22%，K8 只有 17–39%。
- **第 4 行（接口饱和）不成立：** 天花板 0.118 远低于 0.479。
- **落在第 3 行（介于两者之间）**，按第 2 行执行，但做两处修正：
  1. 字典用 **E_full**，不用 E_min；
  2. 控制器**每步**都给出 θ。
- 新的刻画：**v6 的学习价值 = 终盘、逐步、低 ρ 的约束估值。**
- “早期用开环调度，终盘才用闭环控制器”（终盘 θ 模式，可以省去约 80% 的控制器调用）留作后续的效率选项，不进入 T8。

### 1.5 自我更正（T6 §2.1）

T6 中我说“v6 学到动态，是因为梯度经由 softmax 回传”，这不对：T1-R 的 item-PG 本来就经由 softmax(score/τ) 回传，而且是逐步决策。T1 学不到动态的嫌疑因此重新排序：

1. **字典可控性：** G0+G1 只有价格加 α，共 M+1 个自由度，结构上与 E_min 类似。T1-P 中 dynamic α 只在 1.07–1.11 之间变化，与“没有可调的余地”是一致的。
2. **逐步信用分配配方：** PPO+GAE+独立 critic、LP 势函数整形、续行排序辅助损失。
3. **数据与模型协议：** v6 按 M 分别训练、只用 N=100、生成器中相关分布占 0.8、控制器是 BipartiteEncoder。

T8 用 2×1 的设计检验嫌疑 1 和 2；嫌疑 3 只在 T8 门控的最后一行才处理。

### 1.6 对“弱解码器边界”的影响

E_full 的 θ 有 M+8+M(M+1)/2+K 维（M=10 时为 76 维），比 G0+G1 大得多，但仍然满足 I-1、I-2、I-3、I-6。T7 的测量还表明，它的学习价值发生在 **ρ 小的估值区**，所以也满足 T6 §2.2 新增的 I-7。L-4 担心的“θ 膨胀”，今后改用“可控性”来回答，不再用“常数下的增益”。T9 的最终矩阵会给出“哪些项是提供可控性所必需的”。

---

## 2. T8：E_full 执行器移植 + 两种训练信号（单种子 1234）

### 2.1 执行器移植（核心工程）

- 新增模块 `mkp_v6exec.py`（或拆成 `mkp_v6_decoder.py`、`mkp_v6_lp.py`、`mkp_v6_basis.py`），**忠实移植**旧项目中的以下内容：
  - `mkp_Model.WeakDecoder.__call__`：全部评分项。包括逐资源价格的 log-cost（含 s_j^β 和 logsumexp）、α 作为收益指数、κ 作为瓶颈、opportunity/reserve（`opportunity_features`）、LP primal（`log(x_lp+floor)`）与 LP dual（截断的 reduced cost）、interactions（`interaction_loads`）、companion（`companion_basis`），以及温度 τ。
  - `mkp_lp.py`：`solve_lp_relaxation`（温度序列 (1,…,3e-4)，Newton 步数 5）和 `lp_potential`。**每步都在残余问题上求解**，并按状态缓存，供执行器、控制器和 critic 共用。
  - `mkp_basis.py`：`CombinationControlHead`、`interaction_loads`、`companion_basis`。
  - 来自 `StrategyTheta` 和 `ModelConfig`：θ 各字段的取值范围与激活函数，以及 eps、feature_clip、opportunity_bandwidth、companion_scales、lp_primal_floor 等常量。
- 每个复制过来的文件，头部都要注明来源路径和旧项目提交号 `4f93555`（v6 checkpoint 对应 `735183d`/`a933ff7`，计算逻辑相同）。
- 新增配置 `ModelConfig.executor: "g0g1" | "v6"`，默认 `g0g1`，**默认行为逐位不变**。
- 接入 `mkp_Model.forward` 的方式：在 `executor="v6"` 时，执行器接收 v6 形式的 θ 字典，在可行候选上打分；可行性仍然是 hard_mask。
- 必须保持的语义：
  - **greedy 取 argmax，平分时取编号最小的物品**（与 v6 `CandidatePolicy.choose` 一致）；
  - 训练时按 softmax(score/τ) 采样。
- 精度：v6 的状态用 float64、评分用 float32，移植时保持一致。

### 2.2 控制器

- 用新项目的 `StrategyController`（双轴 Transformer），新增 `theta_family="v6"` 输出头：
  - **价格：** `floor + (1−floor)·M·softmax(logits)`，与 v6 一致，即均值为 1 的凸组合；
  - **8 个标量：** sigmoid 映射到 v6 规定的取值范围；
  - **interactions 和 companion：** 移植 v6 的 `CombinationControlHead`（资源对等变），输入换成新编码器给出的资源 token 和池化向量；
  - **初始化：** 所有输出层的权重置 0，偏置按 v6 的 `defaults` 反解。这样**未训练的控制器 ≡ v6 的默认 θ ≡ D 臂**，不依赖编码器的权重。
- **动态特征：**
  - 在新项目已有的动态特征上，加入 v6 `state_features` 中的残余 LP 特征：lp_items（x_lp、reduced cost、对偶成本、4x(1−x)）、lp_resources（log 对偶、松弛量、对偶占比）、lp_global（两个 bound 比值）。
  - 这些特征直接复用 §2.1 中每步已经算好的 LP。
  - 用开关 `ModelConfig.ctrl_residual_lp_features: bool` 控制，**T8 中开启**，供 T9 的 L-blind 臂直接关闭。
- **更新方式：** `theta_update="step"`，`theta_interval=1`。

### 2.3 两个训练臂

| 臂 | `signal` | 内容 |
|---|---|---|
| **T8-A** | `item_pg` | 沿用新项目 T1-R 的 item-PG：同实例 K=16 次采样、共享均值基线（训练期方差缩减，B-4 允许）；终局奖励 = 收益/root UB；按 softmax(score/τ_θ) 逐步采样；熵系数 0.01；没有 critic，没有整形 |
| **T8-B** | `ppo_gae` | **v6 配方**，从旧项目的 `mkp_Trainer.py` 和 `mkp_ranking.py` 忠实移植并注明来源：PPO（clip 0.2，ppo_epochs 3，minibatch 128，target_kl 0.03，max_grad_norm 1.0，lr 1e-4）、GAE（γ=1，λ=0.95）、**独立的 critic 编码器**（value_coef 0.5，value_clip 0.2）、熵系数 0.01、**LP 势函数整形**（`shaped_rewards`，Φ=残余 LP bound/Σp）、**续行排序辅助损失**（全部 ranking_* 参数取 v6 默认值，系数 0.1），以及 v6 的 feasibility 辅助头（constraint_aux 0.05） |

- **两个臂相同的设置：**
  - 数据：新项目的生成器，N∈{50,100,150}，M∈{5,10}，容量比 ∈{0.25,0.5,0.75}；
  - 一个模型同时训练两种 M；
  - 验证集：新项目现有的合成验证集（val_sizes=[100]），以 MILP 为参考；
  - **checkpoint 只按验证集的 greedy gap 挑选**。
- **训练预算：** 两个臂都约 2500 次更新（A：10 epoch × 250 step，B=64；B：2500 次 PPO update，每次 B=64 个实例）。
  - 第 1 个 epoch 结束后报告每次更新的耗时。如果预估总时长超过 24 小时，A 臂把 B 降到 32，B 臂把 `ranking_states` 降到 8，并在 meta 中注明。
  - 两个臂都要报告总轨迹数、总 LP 求解次数和 GPU 小时数。
- **边界说明（写进 meta 和报告）：** 续行排序会在训练期从冻结策略做配对续行 rollout，它**只是训练信号**，推理时只走一条 greedy 轨迹；critic 和 feasibility 头在推理时不使用。

### 2.4 评测（只做单轨迹 greedy；评测入口中所有 sampleK、SMS 和 ILS 阶段一律禁用）

| 臂 | 说明 |
|---|---|
| D_port | 未训练的控制器，即 v6 的默认 θ |
| G_port | T5 的全局常数 θ（`results/t5_v6/attribution/theta_means.json`），在新项目中注入 |
| A | T8-A 的 best.pt |
| B | T8-B 的 best.pt |
| L_v6（引用） | T5 的数字（Chu 0.310，合成 0.479），不重跑 |
| F0（引用） | 新项目 G0+G1 的 fixed（Chu 0.611，合成 1.079） |

- **数据：** Chu 180（best-known 参考）、合成 N100 128 例（MILP 参考，与 T3–T7 相同）。
- **报告：**
  1. 每个臂的 gap、p90、worst；A 和 B 分别相对 G_port、相对 L_v6 的配对差、胜/平/负与 Wilcoxon p；A 与 B 之间的配对比较；按 M×N 和容量比分层；每个实例的耗时（LP、控制器、总时间分开）。
  2. **学到的 θ 是不是动态的：**
     - 在 Chu 上对 A、B 的 θ 迹，按 T5 的方式做三部分方差分解（类别之间、类别内实例之间、轨迹内）；
     - 报告 tau_cap≥0.8 阶段与之前相比，θ 各字段逐步变化量的均值；
     - 计算回收率 (G_port − 臂)/(G_port − L_v6)。
  3. **训练曲线：**
     - 验证集 gap；
     - A：R_mean、frac_better、熵；
     - B：critic 的解释方差、approx KL、clip 比例、ranking 标签的接受率和保留率（L-10 要求）；
     - 两个臂：θ 的统计量随训练的变化。

### 2.5 必要测试（只加 3 项）

1. **执行器与 LP 等价：**
   - 用独立子进程导入旧项目（`sys.path[0]`=旧项目根目录，并检查所有 `mkp_*` 模块都来自那里），在 Chu 的 6 个 (M,N) 类别中各取 1 例；
   - 分别注入 D 和 G 两组 θ，与新项目逐步比较：每步的候选 score 满足 allclose(1e-5)，残余 LP bound 满足 allclose(1e-6)，greedy 解逐位一致；
   - 另外，新项目中的 D_port 和 G_port 在 Chu 180 上的平均 gap 与 T5 相差都不超过 0.005 pp（0.571 和 0.514）。
2. **初始化等价：** `theta_family="v6"` 的控制器在任意编码器权重下，第 0 步输出的 θ 都等于 v6 默认 θ（allclose）；D_port 与“未训练控制器的 greedy”得到的解逐位一致。默认配置 `executor="g0g1"` 下，T1-P checkpoint 的 greedy 结果逐位不变。
3. **PPO/GAE/整形数值检查：**
   - 构造好的奖励和价值序列上，GAE 与手工计算一致（包括 padding 和终止步）；
   - `shaped_rewards` 沿轨迹的累加等于 objective − Φ(s0)（望远镜和性质）；
   - `ppo_epochs=1` 且 clip 不起作用时，PPO 的策略梯度与普通策略梯度一致。

## 3. 门控（Claude 根据回报判定；以 Chu 为主，合成 N100 作佐证）

| 结果 | 解释 | T9 |
|---|---|---|
| **A** 比 G_port 好 ≥0.1 pp（p<0.05），且学到的 θ 中轨迹内方差占主导 | T1 失败的原因是**字典可控性**；简单的 item-PG 在 E_full 上就能学到动态 | A、B 中较好的作为主方法 L，开始 T6 §2.3 的**最终消融矩阵**：L-res、L-blind、L-cap，以及跨规模、跨 M（Chu M=30）、跨分布（GK、Weish、相关分布的合成数据）。仍然单种子；结论成形后再补种子 |
| 只有 **B** 比 G_port 好 ≥0.1 pp | 字典是必要条件但不够，**逐步信用分配配方**也是必要的（C4） | 以 B 作为 L 开始最终矩阵；同时对 B 做单因素拆解：去掉 ranking、去掉 shaping、GAE 换成 Monte Carlo、去掉 critic，每个都是单种子 |
| B 达到或优于 L_v6（Chu ≤0.33） | 在新框架内复现了 v6，而且是一个联合训练两种 M 的模型 | 在上面两行的基础上，L 就是主方法 |
| A、B 都没有比 G_port 好 0.1 pp | 原因在控制器、特征或数据协议（嫌疑 3） | T9：把 v6 的 BipartiteEncoder + HighLevelController 作为控制器做单因素移植；必要时改为按 M 分别训练、使用 v6 的生成器 |

## 4. Codex 回报要求

1. 改动文件清单和提交号（docs 与代码分开提交；不提交 `results/` 等目录；用 `git -C D:\PhD_Doctor\Study\Codes\RL_ModelInhance status` 证明对旧项目零改动）。
2. §2.5 三项测试的结果，以及 D_port 和 G_port 的复现数字。
3. 训练运行的配置、实际预算、耗时，以及任何降配的说明。
4. §2.4 的报告表。
5. 偏离计划之处和未验证的假设。不要运行任何多轨迹或搜索阶段。
