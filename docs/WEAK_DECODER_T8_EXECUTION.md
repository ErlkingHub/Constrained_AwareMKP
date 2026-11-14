# T8 执行说明：E_full 移植与两种训练信号

依据 [T7_RESULTS_AND_T8_GUIDANCE.md](T7_RESULTS_AND_T8_GUIDANCE.md) 实施。Claude 负责研究判断；本轮完成实现、局部验证和运行交接，不判定 T8 研究门控。

分支：`codex/t8-v6-executor-training-signals`。指导文档提交：`e24cbb8`，代码提交：`e20004f`。代码和本执行说明分开提交。

## 0. 2026-11-13 补充：当前评估改为 Chu270

按用户最新要求，`scripts/eval_t8.py` 现在加载完整 **M∈{5,10,30} × N∈{100,250,500}，九组各 30 例，共 270 例**。D_port、A、B 必须覆盖全部 270 例才算完成；M30 的 90 例作为跨约束数测试，直接使用原联合 M5/M10 模型，不改变训练/验证协议或要求重新训练。新建配置的通用 `EvalConfig.dims` 默认也包含 30。

T5 的 G 常数 θ、L_v6 历史结果只覆盖 M5/M10。G_port 继续评估其有参数的 180 例，L_v6 继续引用原 180 例；没有伪造 M30 θ、补默认值或运行旧项目来补 L。A/B、A/B 对 D 的配对覆盖 270；A/B 对 G/L 的配对及回收率只在相同的 180 例上计算。`paired_coverage.csv` 列出各比较的分母与未配对数，`summary.csv` 显示每行的实际 M 范围和例数；M30 单独列出 gap、耗时、A/B 与 D 的比较，缺少历史对照的回收率留空。T5 的约 0.571/0.514% 和 0.005 pp 复现门槛仍只用于原 180 例共同子集，不能作为 Chu270 参考均值。

**已有 `results/t8_port` 的完整 180 例基线可直接复用。** 新入口核对其 SHA、逐例身份与参考值，只补跑缺少的 90 条 M30 D 轨迹，然后运行 A/B 的完整 270 例；合成128仍保持原协议。现有结果和 checkpoint 不覆盖。已完成训练时直接执行：

```powershell
$py = 'D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe'
& $py scripts/eval_t8.py --arms A B --device cuda:0 --port_reference results/t8_port --output results/t8_best_270
```

若从头生成基线，使用 `--output results/t8_port_270`，之后把 `--port_reference` 指向该目录。`--limit 2` 保留为工程检查开关：现在代表 Chu3（每种 M 一例）+合成2；G 只有其中 M5/M10 两例。它不代表正式270评估。

本次修改的验证：5 项覆盖/迁移/配对测试和 31 项既有回归通过，含真实 M30 实例的单轨迹 greedy；另对现有 A/B checkpoint 做 Chu3+合成2 的入口检查。完整270评估仍由用户运行。下文第1–3节为最初实施时的工程记录，不代表后续训练或评估的最新进度。

## 1. 已实现的代码

| 文件 | 核心修改 |
|---|---|
| `mkp_v6exec.py` | 原式移植 StrategyTheta、CandidatePolicy 和 E_full WeakDecoder；平分取最小物品编号，训练按物品概率采样 |
| `mkp_v6_lp.py`、`mkp_v6_basis.py`、`mkp_v6_config.py` | 原 LP Newton 温度序列/步数、势函数、组合头、固定组合基及 v6 常量 |
| `mkp_v6_env.py`、`mkp_v6_cache.py` | float64 物理状态、按状态共享 LP memo；携带根节点编码上下文 |
| `mkp_v6_model.py`、`mkp_Model.py` | 新项目 StateEncoder 与双轴 StrategyController，新增 v6 输出头、零初始化、残余 LP 动态特征及独立 critic；旧 G0+G1 路径保留 |
| `mkp_v6_ppo.py`、`mkp_v6_ranking.py` | 旧版 rollout、PPO、GAE、势函数整形、feasibility 辅助损失及冻结策略续行排序配方；增加训练统计 |
| `mkp_t8_train.py`、`scripts/run_t8.py` | 联合 M 的 A/B 配置、2500 更新预算、greedy 验证选 best、时间门控、轨迹/LP/GPU 小时计数 |
| `scripts/eval_t8.py`、`scripts/t8_report.py` | 仅 greedy 的 D_port/G_port/A/B 评估、历史配对、分层统计、θ 方差/变化量、训练曲线 |
| `mkp_config.py`、`mkp_Trainer.py`、`train_ConstrainedMKP.py` | 配置和训练入口分流；旧 checkpoint 补齐新增字段的默认值 |
| `Benchmark_zeroShot.py` | 对 v6 提示使用专门入口，阻止误入 sampleK、SMS 或 ILS 阶段 |
| `tests/test_t8_v6.py`、`scripts/t8_legacy_parity.py` | 指导要求的三个测试；旧项目仅在隔离子进程内导入 |
| `requirements.txt` | 增加训练曲线导出所需 matplotlib |

复制模块头部保留旧项目来源和提交 `4f935551e62d91d55d998fa6255a67f9b0fafeb7`。运行时不依赖旧项目导入。旧版 checkpoint 的计算来源说明沿用指导：`735183d` / `a933ff7`，执行器计算逻辑相同。

`executor` 与 `theta_family` 默认均为 `g0g1`，`ctrl_residual_lp_features=False`，`v6_critic=False`。T8 显式选 `v6`、每步刷新 θ、打开残余 LP 特征。价格和八标量输出采用 v6 激活函数，interactions/companion 采用原 CombinationControlHead。资源 LP 特征中的对偶量使用原版 `log1p(clamp(dual,0,clip))`；物品对偶成本使用原版截断 log。

新编码器的原有静态特征采用移植 LP 在根节点求得的 dual/bound，之后固定；新增 LP 动态特征每步更新。执行器状态及收益结算为 float64，评分与 θ 为 float32。A/B 配置均关闭 AMP。Actor 和 Critic 共享该状态的 LP 解，但不共享网络参数；greedy 入口不执行 Critic 或 feasibility 头。

## 2. 训练协议和预算

共同设置：种子 1234，原新项目生成器不变；N∈{50,100,150}、M∈{5,10}、容量比∈{0.25,0.5,0.75}。每次更新选择一组 N/M，联合训练一个模型。验证使用原种子 20240607 生成的 N100、每种 M 64 例，复用 MILP 缓存和最优性标记；只按平均 greedy gap 选 `best.pt`，平分保留较早 checkpoint，`last.pt` 分开保存。允许初始化模型成为 best。

| 项目 | T8-A | T8-B |
|---|---|---|
| signal | item_pg | ppo_gae |
| 更新数/实例批量 | 2500 / 64 | 2500 / 64 |
| 主训练轨迹 | 同实例 K=16，共享均值基线 | 每实例一条采样轨迹 |
| 奖励 | 终局 profit / 移植 LP root UB | 逐步 profit / Σp，叠加 LP 势差 |
| 梯度配方 | 原新项目 item-PG 与按微批次归一化优势 | 原 v6 PPO/GAE/独立 Critic/feasibility/ranking |
| 熵系数 | 0.01；沿用新项目原始 categorical entropy，按实际决策步平均 | 0.01；沿用 v6 按候选数量归一化 entropy |
| 默认主训练轨迹总数 | 2,560,000 | 160,000 |

A 采用完整 K 组的微批次，最多 64 条轨迹一次前向/反向；不拆开同实例共享基线。B 的 PPO clip=.2、epochs=3、minibatch=128、target KL=.03、GAE γ=1/λ=.95、value_coef=.5、value_clip=.2、constraint_aux=.05，ranking_* 全部沿用 v6 默认值。两臂 lr=1e-4、max_grad_norm=1。

第 250 次更新后，以已观测更新计算时间估计 2500 次更新。超过 24 小时则 A 的实例批量 64→32，或 B 的 ranking_states 16→8；写入 `meta.json` 的 reductions 和 actual_config。该预测不含尚未执行的周期验证/写盘开销。A 若触发此规则，主训练轨迹总数变为 1,408,000。未实施其他自动降配。

训练计数包含：主训练轨迹、A 的 greedy 诊断轨迹、固定验证轨迹、ranking 的所有配对分支（包括被拒绝标签、以及保留率诊断）。`ranking_diagnostic_trajectories` 是 ranking 总数的子集，不能再加一次。LP 同时报告批量调用次数及状态行数，后者包含 padding 行的计算；统计包含初始化/周期验证，排除 MILP 参考值准备。`GPU_hours` 为 CUDA 运行的墙钟占用小时，非 kernel 活跃小时。

续行排序标签来自冻结策略，是训练信号，不进入 PPO buffer、GAE 或 critic target；不是最优 Q 标签。B 每 25 次更新（含第 1 次）在接受的配对状态上重新进行更新后策略的终局续行：

- `ranking_acceptance`：接受标签数 / 提议配对数。
- `ranking_retention`：更新后仍通过原接受阈值且收益差同号的比例；无接受标签时留空。
- `ranking_return_sign_retention`：只检查更新后收益差同号，不要求重新通过阈值。
- `ranking_post_greedy_aligned_return_gap`：按原标签方向对齐的更新后 greedy 终局收益差，单位为 objective/Σp。
- `ranking_post_accuracy`：更新后分数次序与原标签同号的比例，单独记录，不能代替终局收益保留率。

A 的 `R_mean` 是每 log_every 次更新前对当前训练批次进行的 greedy 诊断，`frac_better` 是采样轨迹超过它的比例；两者不参与基线或训练目标。

## 3. 本轮实际验证

**没有运行正式 A/B 训练，也没有运行完整 Chu180/合成128 评估。以下仅为工程检查，不能支持方法增益或门控结论。**

| 检查 | 实际结果 |
|---|---|
| 必要测试 1 | 在 Chu 六种 (M,N) 各取一例，D/G 共 12 条轨迹；隔离旧模块来源检查通过；逐步候选 score allclose(1e-5)、残余 LP bound allclose(1e-6)、选择序列及最终解逐位一致 |
| 必要测试 2 | 两个随机编码器种子、M5/M10 的零初始化 θ 与 T5 保存的原 D θ 一致；未训练 greedy 与 D 固定注入解逐位一致；两个现有 T1-P checkpoint 与不可变旧源码比较通过 |
| 必要测试 3 | GAE 的终止/padding 手算、整形望远镜和、clip 未起作用的 PPO/普通 PG 梯度一致，均通过 |
| 既有回归 | weak_decoder、training、T1-R、theta_pg、T4 SMS，共 32 项通过 |
| 梯度/排序分支 | 两臂各一次极小更新，梯度和损失有限；B 实际接受排序标签并完成优化；Actor/Critic 参数集合不相交 |
| CUDA 小检查 | 两臂各一次 B1/N9 的有限梯度更新；通过 forward hook 断言 greedy 推理不执行 Critic/feasibility 头 |
| 写盘和曲线 | 两臂各 1 次极小更新，N17、B2、16 维模型，验证每 M 仅 1 例；best/last、预算统计、36 张适用曲线导出通过 |
| 新评估入口 | D/G 及上述小模型 A/B，分别在 Chu 两例和合成两例完成 greedy、配对、θ 方差与逐步变化报告；小模型无科研效果解释 |

形状等价测试使用 CPU；未把 CPU 结果扩大解释为完整 GPU 基准复现。

入口 smoke 的 D/G 对照如下，仅每个数据集 M5/M10 各一例；四组都与 T5 的**同一批实例**结果一致，不能与全量均值比较：

| 数据集 | D_port gap(%) | G_port gap(%) | 两组相对同实例 T5 差值(pp) |
|---|---:|---:|---:|
| Chu，2 例 | 1.091704 | 1.336675 | 0 / 0 |
| 合成 N100，2 例 | 0.600693 | 0.600693 | 0 / 0 |

正式 Chu180 的 0.005 pp 复现门槛**尚未验证**。完整入口将与 T5 原始逐例 D/G 行比较，超差则失败；指导给出的全量参考约为 D=0.571%、G=0.514%。L_v6 只引用字节及参考值匹配的 T5 行，不重新运行。F0 只引用指导中的 Chu=0.611%、合成=1.079%，不构造其新配对证据。

旧项目检查：`git --no-optional-locks -c safe.directory=D:/PhD_Doctor/Study/Codes/RL_ModelInhance -C D:/PhD_Doctor/Study/Codes/RL_ModelInhance status --porcelain=v1` 在检查前后均无输出，HEAD 均为 `4f935551e62d91d55d998fa6255a67f9b0fafeb7`。没有写入旧项目；隔离子进程禁用 bytecode。

## 4. 用户运行命令

在本分支项目根目录执行，已有 T5 正式导出及 attribution 结果应保留。新入口会验证其 manifest、数据字节哈希、实例数量和参考值。请勿复用已有输出目录；失败或中断后换新目录保留证据。

```powershell
$py = 'D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe'

# 仅生成配置，不训练
& $py scripts/run_t8.py --cuda_device 0

# 新生成基线：D为Chu270，G为有T5参数的Chu180；各加合成128；仅greedy
& $py scripts/eval_t8.py --device cuda:0 --output results/t8_port_270

# 上一步 status=complete 且 reproduction.json 通过后，再分别启动两臂
& $py scripts/run_t8.py --train A --cuda_device 0
& $py scripts/run_t8.py --train B --cuda_device 0

# best 正式比较：A/B各Chu270；复用D/G并在共同子集引用T5的L
& $py scripts/eval_t8.py --arms A B --device cuda:0 --port_reference results/t8_port_270 --output results/t8_best_270

# 训练曲线
& $py scripts/t8_report.py --runs runs/t8_a_jointM_s1234 runs/t8_b_jointM_s1234 --output results/t8_training_curves
```

如需检查 last，单独输出，不能混进 best 表：

```powershell
& $py scripts/eval_t8.py --arms A B --tag last --device cuda:0 --port_reference results/t8_port_270 --a_checkpoint checkpoints/t8_a_jointM_s1234/last.pt --b_checkpoint checkpoints/t8_b_jointM_s1234/last.pt --output results/t8_last_270
```

必要测试与入口检查的独立命令：

```powershell
& $py -B -m unittest discover -s tests -p test_t8_v6.py -v
& $py scripts/eval_t8.py --limit 2 --device cuda:0 --output results/t8_code_checks/port_smoke_new
```

`--limit 2` 明确标注 smoke，不通过正式复现门槛。评估入口没有 sampleK、SMS、ILS 参数，推理每例始终只构造一次 greedy 解。A 的 K16 与 B 的续行均只在训练期存在。

## 5. 给 Claude 的结果

提供两臂的 `runs/.../meta.json`、`train_log.csv`，以及 `results/t8_port/reproduction.json`、`results/t8_best/report.md`、`summary.csv`、`paired_summary.csv`、`paired_per_instance.csv`、`recovery.csv`、`theta_variance.csv`、`theta_step_changes.csv`、`per_instance.csv` 和训练曲线。评估报告提供总体、M×N、M×N×容量比分层；差值统一为“臂−对照”，负值有利于该臂，Wilcoxon 为双侧且未校正。总耗时为带同步/trace 的实测墙钟时间；controller 时间扣除其内部 LP 时间，不能直接与历史硬件计时作速度增益比较。

偏离及未验证项：

- 不移植旧 memory24 的整批 rollout 几何缓存优化，保留原数值路径与状态 LP memo。PPO 重算状态仍会求 LP，所有实际调用均计数；耗时可能比旧项目高，等待正式首 epoch 测量。
- 新控制器、Critic 均用本项目双轴骨干；静态特征使用移植根 LP，新增动态特征含指导要求的 4/3/2 维 LP 项，没有增加学习式物品评分。
- 没有恢复训练功能；入口拒绝覆盖既有 run/checkpoint，用户需保留中断结果后以新 run_name 启动。完整预算及首 epoch 降配条件未执行，不能给出正式耗时估计。
- 训练标签保留率按上文给出明确操作定义，周期性增加训练诊断续行，完整计入预算，不影响 PPO/critic 目标。
- 正式 D/G 均值复现、A/B 效果、动态 θ 是否学成及 T9 门控均等待用户完整实验。

## 6. 提交记录

指导文档：`e24cbb8`。实现：`e20004f`（`feat: 移植 v6 完整执行器并接入 T8 双训练信号`）。本执行报告与 README 索引另作 docs 提交；实验产物目录不入 git。
