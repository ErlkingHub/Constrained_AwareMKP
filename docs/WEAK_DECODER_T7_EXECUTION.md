# T7：学习价值位置、稳定度与有限采样接口容量

日期：2026-11-08。研究分析与门控见 Claude 的 [T6 结论及 T7 指导](T6_RESULTS_AND_T7_GUIDANCE.md)。本轮只交付推理诊断代码和必要验证，完整实验由用户运行；没有训练、物品采样策略、多轨迹 best-of 或 ILS。

## 1. 分支和修改

从 T6 的 `902dc69` 创建 `codex/t7-value-interface-capacity`。指导及原有 T6 报告更新单独保存为 `fef71e7`，代码为 `33331b3`；本执行报告另作 docs 提交，编号见分支日志。

| 文件 | 作用 |
|---|---|
| `scripts/t7_interface.py` | 合法 θ 探针、状态特征缓存评分、切换/间隔 actor、目标引导单轨迹诊断 |
| `scripts/run_t7_value.py` | 复用 T5/T6 的隔离、数据和基准；组织 policy/capacity/oracle 三阶段并逐例落盘 |
| `scripts/t7_report.py` | 配对、回收率、切换、调用耗时、容量分布、ρ 分区和分目标类型的 oracle 报告 |
| `tests/test_t7_value.py` | 指导要求的恰好两项测试，通过独立子进程读旧模块 |
| `README.md`、本报告 | 入口、补充口径、已验证范围与完整运行命令 |

新项目模型/解码器、T5/T6 代码及旧项目均不修改。用户提供的 T6 报告改动已保留。结果、cache 和检查点不入 git，不使用 `git add -f`。

## 2. 输入和只读边界

- 旧源码默认 `D:\PhD_Doctor\Study\Codes\RL_ModelInhance`，本项目检查点默认 `checkpoints/lp_dual_v6/mkp_M{5,10}_best.pt`。
- 读取完整的 `results/t5_v6/`、`results/t5_v6/attribution/`、`results/t6_phase/`；不再生成数据或解 MILP。
- 核对 T5/T6 完成且非 smoke、导出 manifest、被冻结的 T6 逐实例结果/迹/相位表 SHA、实例值及参考的哈希、旧源码和两个 best checkpoint SHA。主比较直接复用 L/G/S2 基准，不重新推理这些比较臂。
- MILP 目标向量只读已有 `cache/milp_*.npz` 的 `solution`，校验缓存 SHA、原始批次哈希、索引、binary/可行性/目标值以及 `optimal` 标记。缓存缺失或不匹配就停止，不自动求解。
- 旧模块只进入旧版独立进程；新项目的同名 `mkp_*` 不导入。关闭字节码写入，旧 Git 审计禁用 optional locks；统计单独运行。
- 前后对比旧项目 commit/branch/status 与九个源文件 SHA，所有输出必须在旧项目之外。

## 3. 实现协议与补充约定

### 3.1 T7-a/b：阶段与决策间隔

新增 `SL05/SL08/SL09/LS08/K2/K8`，对应指导的四个切换臂与 k=2/8。τ 使用当前决策前的 `tau_cap`，严格按 `<阈值` 的两侧分配来源；回归端点 0/1 明确定义为全 L/全 S2，避免零需求尾项在 τ=1 时破坏端点一致性。

- 记录首次阈值到达的 step、此时 feasible 数量、剩余步数。`switch_remaining_steps` 包含第一次切换后的当前决策，另有减去该步的 `switch_remaining_after_action`。
- 未达到阈值记 null，不假定在终点切换；汇总同时给到达率和已切换实例的条件均值。
- K2/K8 在 step=0 及 k 的倍数更新 L 的完整激活后 θ，中间保持全部 θ 不变。旧 decoder/LP 仍每步更新，代码断言 LP 求解次数等于实际决策数、控制器次数等于 ceil(steps/k)。
- 控制器耗时分别记含 LP 的前向时间、减去本次前向 LP 求解的时间。后者仍含其他特征计算，并非纯神经算子时间；含 LP 的列与 LP 总时间重叠，不能相加。复用 L/G/S2 的调用数由原有逐步/固定/调度契约推得，标明来源，不伪造未测的控制器分项耗时。

### 3.2 T7-c：一次特征缓存与两个字典

只沿 L 采集状态：合成 128 例全部状态，Chu 按 M×N×容量比取原文件顺序前两例，共 36 例。使用旧 `prepare_state` 缓存相同的 θ 无关几何、opportunity、reserve、LP 与组合基函数；L、S2 和所有 θ 探针共享该状态缓存。

`StateScores` 照旧公式批量评分，调用旧 `CandidatePolicy` 保留相同的归一化和最小物品编号 tie-break。L 与 S2 的已知锚点单独评分，避免批量形状影响已知选择。每个 L 状态断言单点缓存选择等于旧 decoder 选择；每条状态源轨迹最终逐项匹配 T5 L 的选择序列。

- E_full 保留全部项。
- E_min 保留 profit_exponent/scarcity/lp_primal/companion，价格为 1，其余系数为 0；temperature 保留，执行器配置和特征算法不改。
- 全域为 2048 个随机 θ，另加 L/S2 两个锚点；局部为 512 个 θ。它们只在**固定状态**上打分，不展开候选 continuation，更不从多条策略轨迹挑最好值。
- 随机数以单种子 `20260910`、数据集、实例 id、已选择集合和 global/local 标签构造 SHA 种子；两个字典共用同一批 θ，分块仅控制内存。默认每块 128，可用 `--theta_chunk` 调整并记录。

**当前默认的补充抽样口径：** v6 的 interactions 上三角之和及 companion 总和有预算，不能逐分量矩形抽样后声称合法。这里令总权重在 [0,max] 均匀，分配比例为 Dirichlet(1)；interactions 对称存储且上三角只计一次。该分布是“均匀总权重 × 均匀比例”，**不是整个预算单纯形的体积均匀分布**。标量在规定区间均匀，价格 logits 在 U[-4,4]，经旧版 floor+softmax 规则激活。

局部中心为 S2 的激活后 θ。只从 T6 合成拟合迹，按类别、tau_cap 重新计算十个分位段的 L−S2 残差人口标准差（ddof=0），不是用原 θ 的方差。标量逐字段、companion 逐分量；价格在可交换资源维共享尺度，interactions 对上三角共享尺度。空段用该类别全体残差尺度。加独立 Gaussian 后，标量截到合法区间，价格截下界后归一，组合权重截为非负并在超预算时径向缩放。**投影后的分布不再是无约束 Gaussian。** 所有抽样批次检查有限、范围、价格均值和组合预算。

### 3.3 报告定义

- 每状态记录 F、两类可达集合大小、ρ、与 L 选择相同的局部频率、L/S2 选择差异。
- 两字典的主要 stability/difference 统一以原始 E_full 的 L/S2 为参照，便于同状态比较；另保留 E_min 投影后的 L/S2 选择，以及 `stability_projected_L`，不混淆两种定义。
- 按各数据集 L 状态的 tau_cap 十分位、N，以及 L/S2 不同步单独汇总；输出均值、p10/median/p90。ρ 分区为 `<.2`、`.2… .5`、`>.5`，同时报告区内不同步比例和全部不同步落在该区的比例，分母明确。
- 状态分布是描述性统计，不把同一实例的多步视作独立训练样本。策略比较才用逐实例 Wilcoxon，未作多重比较校正。
- 回收率是平均 gap 差之比，不是逐实例比率的均值；不截到 [0,1]，近零分母留空、非正分母标记。SL08 与 LS08 的回收率之和及其相对 1 的偏差另列。

### 3.4 Oracle 目标与“天花板”的限制

**已核对的缓存事实：128 例只有 53 例已证最优（M5=49，M10=4），另 75 例为 incumbent。** 当前默认全部运行，但两类严格分表：只有前者记 `proven_optimum`，其余记 `MILP_incumbent`，后者不称 x* 或最优天花板。可用 `--oracle_targets proven` 只运行 53 例。

每个字典每例只构造一条目标引导诊断轨迹。在当前状态的抽样可达集合与未选目标物品交集中选 S2 排名最高者；为空则回到**旧版完全相同的 S2 归一化排名**，记录第一次空交集的 step、tau_cap、step/N 和 F。目标已全部选完则进入单独标记的 S2 补齐阶段，不把“目标耗尽”误记为接口失败。目标阶段始终非空时断言该前缀 ⊆目标，之后继续到可行极大；incumbent 可能非极大，因此补齐后可以包含目标外物品。

这个数值只是**已达到的诊断性能/接口最优能力的下界**：有限抽样未穷尽 Θ；指定目标解可能不唯一；即使每状态可达集合已知，S2 在目标交集内的贪心顺序也未枚举所有可达轨迹。因此空交集不是不可达证明，结果也不是经证明的“最多能达到多少”。不据此自动判 Q-3 或 T8 门控。

## 4. 已完成验证

恰好两项必要测试在隔离子进程中实际运行通过，最终复核 **2/2，6.673 s，未跳过**：

1. 单点 L 的缓存 logits 与旧版逐位一致，可达集合只有 L 的选择；SL(0) 与 L、SL(1) 与 S2 的目标值、选择序列及解集合逐位一致。
2. 两个字典的 oracle 每步经旧环境可行性检查，最终经独立目标值/可行性/极大性核验；全程交集非空时保持目标内前缀。另覆盖 S2 补齐及首步空交集回退。

有限 smoke（`results/t7_code_checks/smoke/`）：

- 两个合成 N100、两个 Chu N100，新增 24 次切换/间隔单轨迹，复制 12 条 L/G/S2 参照，共 36 行策略结果。
- 回放 4 条 L，共 211 个状态、422 个状态/字典记录，选择序列全部复现。只用 32 个全域和 16 个局部探针，明确 `smoke_only`，不冒充 2048/512 的正式测量。
- 两个合成目标各跑两个字典，共 4 次 oracle，覆盖一例已证最优和一例 incumbent；所有结果可行且极大。
- 对齐 S2 排名后，只额外复查这 4 次 oracle（`results/t7_code_checks/oracle_ranking/`），未重复切换或容量测量；目标值及选择序列与首次结果一致。统计报告重生成不涉及模型推理。
- 两次推理运行均 `status=complete, legacy_unchanged=true`，旧项目 HEAD 为 `4f935551e62d91d55d998fa6255a67f9b0fafeb7`；前后 branch/status/九个源码 SHA 一致，`git -C <legacy_root> status --porcelain=v1` 为空。

未运行完整训练或正式 T7 评估，不从上述 smoke 的 gap、ρ 或稳定度作研究结论。全域分布、局部投影及目标分组是明确记录的补充口径，尚需在正式分析中结合抽样预算解释。

## 5. 用户运行

完整三阶段，复用当前 T5/T6 产物：

```powershell
python scripts/run_t7_value.py --device cuda:0
```

默认输出 `results/t7_value/`，拒绝覆盖；旧项目、checkpoint、T5 数据/结果、T6 结果和 cache 可分别通过 `--legacy_root`、`--checkpoint_dir`、`--instances_dir`、`--t5_results`、`--t6_results`、`--reference_cache` 指定。

长任务可拆开运行，各次使用不同输出目录：

```powershell
python scripts/run_t7_value.py --stages policy --output results/t7_policy --device cuda:0
python scripts/run_t7_value.py --stages capacity,oracle --output results/t7_capacity --device cuda:0
# 若只接受已证最优的 oracle 目标：
python scripts/run_t7_value.py --stages oracle --oracle_targets proven --output results/t7_oracle_proven --device cuda:0
```

正式新增预算：policy 为 6×(128+180)=1848 条；capacity 为 128+36=164 条 L 状态源轨迹（每状态静态评分，不额外展开）；oracle 为 256 条，或 proven-only 的 106 条。阶段可独立执行，不自动恢复或覆盖已有目录。

主要交给 Claude：`report.md`、`summary.csv`、`paired_summary.csv`、`recovery.csv`、`additivity.csv`、`capacity_summary.csv`、`capacity_regions.csv`、`target_summary.csv`、`oracle_summary.csv` 和 `meta.json`。原始状态/策略/目标诊断逐例保留，报告检查重复、状态轨迹覆盖与抽样预算。

```powershell
python scripts/t7_report.py --input results/t7_value
python -B -m unittest discover -s tests -p test_t7_value.py -v
python scripts/run_t7_value.py --limit 2 --output results/t7_code_checks/smoke_new --device cuda:0
```

测试可用 `T5_LEGACY_ROOT`、`T5_CHECKPOINT_DIR`、`T6_RESULTS_DIR` 调整外部依赖位置；缺失时明确跳过，本机两项均实际执行。没有训练入口，也没有多轨迹策略采样或 ILS 参数。
