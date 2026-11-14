# T9 实施与运行交接：new_theta、LP 证书与核心集

日期：2026-11-14。依据 [RESEARCH_HANDOFF_TO_CODEX.md](RESEARCH_HANDOFF_TO_CODEX.md)。分支 `codex/t9-new_theta-certified-core`，指导与进展快照提交 `c783526`，代码提交 `cf5f4d5`；本说明另作 docs 提交。

用户补充优先：**新训练只做 cert，不重训 v6/v6c；以后完整评估均为 Chu270；先估算训练时间，暂不优化训练速度。** v6 仅作为既有 checkpoint、固定参照和离线接口探针。T9-5 的 beam、多轨迹推理等搜索未准备、未运行。

## 1. 实施范围

| 文件 | 行为 |
|---|---|
| `mkp_cert_basis.py` | float64 的 sur/reg/blk/fill/pair、稳定排序的 rank/core 标记、按状态缓存；blk/pair 共用分块配对遍历；提供原始单位 I1 分解 |
| `mkp_cert_exec.py` | CertTheta 的 5 个 simplex 权重和温度；K-rc/K-bound/K-profit/K-pair；沿用 CandidatePolicy 的 greedy 平分规则及训练采样 |
| `mkp_cert_model.py` | 复用新项目编码器和双轴控制器表示；单个 Linear(d,6) 零权重初始化，无 price/CombinationControlHead；仅支持 item-PG |
| `mkp_v6_lp.py` | 新增 unclamped dual_value、可行缩放后的 primal_value、certificate_gap、scale；额外保留 float64 raw_prices，供恒等式计算使用。原有七个输出不改运算式或返回精度 |
| `mkp_Model.py`、`mkp_config.py`、`mkp_Trainer.py` | cert 分流、特征维度开关、旧 checkpoint 默认字段兼容；默认 G0+G1/v6 架构与数值保留 |
| `mkp_v6_model.py` | theta_patch、冻结组开关、默认关闭的 prefix(δ)。ctrl_basis_features 打开时追加 5+rank+core 项和 3 个全局量 |
| `mkp_t8_train.py` | 复用原 A 臂 item-PG 配方，支持 cert 的统计和独立 checkpoint schema；cert 正式 fit 要求门控凭据，拒绝 PPO |
| `scripts/run_t9_measure.py` | T9-0 的精确 LP、I1 分解、core 诊断、冻结组；T9-4 的 prefix 诊断。HiGHS 整数解只出现在参考/诊断中 |
| `scripts/t9_interface.py`、`scripts/run_t9_dictionary.py` | 合成校准、固定臂、移植执行器的分数探针、随机基对照、按顺序尝试/拒绝扩展、oracle 和两种 ρ 分母 |
| `scripts/run_t9.py`、`scripts/eval_t9.py`、`scripts/t9_report.py` | cert 配置、门控训练入口、Chu270 greedy 评测和配对/动态 θ/I1 报告 |
| `scripts/profile_t9.py` | 少量完整更新的计时外推，不建立正式 run，不做速度优化 |
| `scripts/t9_common.py`、`tests/test_t9_cert.py` | 数据/参考值校验、报告公用逻辑，以及指导要求的五项测试 |

保留并提交上一轮尚未提交的 `scripts/t8_data.py`、`tests/test_t8_coverage.py`、评估/README/说明修改；T9 没有再次改写 `eval_t8.py` 的主流程。T9 直接复用 Chu270 数据读取与旧 T5 合成128交换文件。

`ctrl_basis_features=False`、`v6_frozen_groups=[]`、`lp_warm_start=False` 是默认值。冷 LP 路径保留。warm start、缩短 Newton 日程、AMP、额外几何缓存和批量调优均未实施；打开 lp_warm_start 会明确报错，避免配置与实际行为不一致。训练 B64→B32 仍只是原 T8 预算规则，不是本次新增优化。

cert 头的初始 π 暂为均匀值，正式值等待新合成集校准。softmax 无法精确表示零权重，默认配置转有限 logits 时以 1e-8 为下限；校准与 K* 使用相同 default_theta 激活。未训练头不依赖随机编码器权重。证书约束作用于 LP、I1/I2 和 P1 的上下界，任意五项加权分数本身不宣称为 Q* 的数值界。

## 2. 训练时间：已先计时，未启动正式训练

本机 `cuda:0 = NVIDIA GeForce RTX 5090 Laptop GPU`。完整保持 T8-A 的宽度/深度、K16、最多64轨迹的微批次、AdamW、梯度 checkpoint 和原生成器，仅换 cert 接口/特征。每个 N/M/B 配置做 **1 次独立计时更新**，每组从初始化开始，共12次，不是正式 fit、2500步训练或收敛实验。

| M | N | B64 秒/更新 | B32 秒/更新 |
|---:|---:|---:|---:|
| 5 | 50 | 42.94 | 24.58 |
| 5 | 100 | 96.56 | 44.30 |
| 5 | 150 | 149.49 | 67.60 |
| 10 | 50 | 46.34 | 18.53 |
| 10 | 100 | 98.59 | 38.16 |
| 10 | 150 | 143.56 | 66.40 |
| 六种形状等权均值 | | **96.25** | **43.26** |

原训练随机等概率选择 N/M。若全部保持 B64：2500×96.25/3600 ≈ **66.8小时**。按原规则，第250步预测超过24小时后改 B32：

`(250×96.2457 + 2250×43.2625) / 3600 = 33.72 小时`。

**完整单臂训练暂估约34–42小时**：下限为实测主更新外推，上限增加25%的工程余量，覆盖 greedy 诊断、验证、写盘、轨迹长度和长时间频率变化。该区间不是统计置信区间或运行上限。它基于尚未正式校准的均匀 π/τ=0.2；最终校准值和学习后的轨迹可能改变耗时，正式校准后建议再计时一次。当前 T8-A 的实际记录为47.71小时、2500更新、末期B32；此处只是新实现的预测，不是已验证的加速比。

数据在 `results/t9_profile/updates.csv` 与 `meta.json`。原规则触发后主训练轨迹数为1,408,000，另有 greedy 诊断和验证。未跑正式训练，不能提供真实总 LP 次数或最终 GPU 小时；正式 meta 会记录这些值。门控实验和最终 Chu270 评估的耗时**不含在34–42小时内**。

```powershell
$py = 'D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe'
& $py scripts/profile_t9.py --device cuda:0 --output results/t9_profile_recheck
# 正式校准后再测，输出目录须为新的目录：
& $py scripts/profile_t9.py --device cuda:0 --calibration results/t9_dictionary/calibration.json --output results/t9_profile_calibrated
```

## 3. 本轮实际验证与证据边界

- 现有全部测试加新增五项，**66 项通过**。新增五项覆盖任意非负乘子的 I1、逐动作 I2、枚举 Q* 的 P1/P3、零初始化与分数适配器兼容、原始 LP 证书夹住精确 LP。证书保留 float64 原始乘子，避免用截断/float32 reduced cost 验证严格恒等式。
- D/G 原 smoke 同实例结果相对 T5 差值0，旧源码隔离逐步测试也通过；新增 M30 的覆盖/合法性通过。旧项目 git status 前后均为空，没有写入旧项目。
- T9-0 数值/core smoke：Chu3+合成2。样本 p95 bound error 为0.00627个平均物品收益，0个 LP 回放失败；这**不是正式 p95 门控结论**。core MILP 使用0.2秒的小预算，超时 incumbent 与已证最优分别标记。
- 冻结组、prefix 三种 δ、固定臂、温度校准、随机基探针、证书分解报告和新模型 checkpoint 读写均做了局部检查。正式训练拒绝 smoke 门控的负向检查通过。另建单次小模型更新仅验证学习模型的加载与 θ 轨迹，不作为训练效果。
- 2048θ 的完整探针预算只测了两个合成实例，整个 oracle 检查约109秒；按同预算粗略线性外推128例约1.9小时，考虑状态/扩展差异可预留2–4小时。不是完整128例的实测。
- 该局部探针里，基础 cert 在一个 M10 incumbent 目标上弱于 E_full，随机基和 scar 扩展可以降低该例的 oracle gap。这是需要回报的负面提示，**不能用两个实例宣布字典通过、宣称泛化或批准训练**。正式门控未执行，decision 保持 pending_Claude。

## 4. 完整实验命令与顺序

所有命令在项目根目录执行，已有 `results/t8_best_270`、T5 交换文件和 MILP 缓存保留。新的输出目录不得已存在。每个新评估臂都检查完整 Chu270（九种 M/N）和合成128；G/L 只有其真实共同子集，不伪造 M30 参考。

### T9-0：先测量，再作研究判断

```powershell
& $py scripts/run_t9_measure.py --tasks numerics core freeze --device cuda:0 --output results/t9_measure
& $py scripts/t9_report.py --input results/t9_measure
```

输出 `accuracy(_summary).csv`、`integrality(_summary).csv`、`identity(_paired/_summary).csv`、`core.csv`、`policy_(summary/paired_summary/retention).csv`、`gates.json`、`report.md`、`tables.md`。根 LP/integrality 用 best-known 或缓存 MILP reference，不将 LP 当评估 gap 参考。core 限制 MILP 按指导只在合成集做；Chu 只有核心覆盖/fixing统计，缺少参考解向量的对称差不填数。640次限制 MILP 每次最多10秒，时间预算上界约1.8小时，另外加 LP 测量和回放。冻结组原指导估计6–8 GPU小时，完整作业本轮未计时。

P1 在正式 Chu 的真实 G 共同子集上若丢失超过10%增益，程序停止余下 F-mean/F-zero/prefix，并回报矛盾。F-mean **不使用 Chu 拟合常数**：M5/M10 用 checkpoint 自己的合成轨迹；M30 另生成32例 N100合成轨迹（种子2026111402）按决策加权拟合对角/非对角均值和两个标量。这个补充使用户要求的 M30 诊断也可运行；只在评估期干预，不更改 checkpoint。

LP p95 超过0.01时，字典入口拒绝继续；没有擅自做 warm start/final polish。应把完整测量交 Claude 后再决定准确性处理。

### T9-2：合成校准、固定臂和字典 oracle

```powershell
& $py scripts/run_t9_dictionary.py --measure results/t9_measure --device cuda:0 --output results/t9_dictionary
```

默认 π 只在新合成集选：种子2026111401，每M32例、N100，70个0.25网格点、0.1局部细化、两折交叉拟合及全校准集最终拟合。该集合独立于固定验证集；另建其 MILP reference 并保留最优性标记。τ沿用 calibrate_beta 的 synthetic best16方法，β=1/2/4/8/16/32/64，平分选较大β，范围为选定τ的1/4到4倍。该采样只校准训练温度，固定臂推理仍是一条 greedy。

固定臂 K-rc/K-bound/K-profit/K-pair/K* 全部跑 Chu270；v6c-G 只在 G 有真实常数的维度作固定诊断，不重训。oracle 仅在合成128做，已证最优53与incumbent75分表，2048个 global/local θ探针加常数；随机基按实例/物品稳定播种，带符号系数，尺度取可行项 reg 的标准差。扩展按给定顺序尝试，未通过随机基停止规则的项先移除再试下一项，总维度不得超过10。ρ_core 使用精确 LP 对偶的 top(2M+10) 与精确 fractional 变量并集，另给全部可行项分母。

输出 `calibration.json`、`temperature.csv`、`fixed_summary.csv`、`oracle_summary.csv`、`random_control.csv`、`pointer.csv`、`extension_decision.json` 和 `gate.json`。校准/固定臂预计另需数小时，尚未完整计时；oracle 预算估计见第3节。可以用 `--stages calibration` 分开生成校准，再用新目录、`--stages fixed oracle --calibration <path>`复用；入口把校准参数复制到最终 dictionary 目录并记录输入 SHA，原文件不改写。

**当前训练实现是基础六维 cert。扩展槽位已实现离线探针，但尚未选择扩展进入训练头。** 若正式结果只允许某个扩展通过，必须先由 Claude 确认所选扩展及对应默认/温度，再接入其训练头/特征并复验；不会将 probe 自动冒充为可训练模型。若基础 cert 不通过，当前入口停止，不回退重训 v6c（用户补充要求）。

### T9-3：仅在完整门控通过且用户/Claude选定基础cert后

```powershell
# 准备配置，不训练；没有正式校准文件时会明确写为 provisional
& $py scripts/run_t9.py --dictionary results/t9_dictionary --output results/t9_configs

# 查看并确认正式 gate.json 后，显式选择 cert 才能启动
& $py scripts/run_t9.py --train --decision cert --dictionary results/t9_dictionary --measure results/t9_measure --cuda_device 0

# 仅一条 greedy：D_cert、K*、T9-A，新臂各Chu270+合成128
& $py scripts/eval_t9.py --dictionary results/t9_dictionary --device cuda:0 --output results/t9_best_270
# last 单独报告，不与 best 混合
& $py scripts/eval_t9.py --dictionary results/t9_dictionary --checkpoint checkpoints/t9_a_cert_s1234/last.pt --tag last --device cuda:0 --output results/t9_last_270
# 新训练的验证、R_mean/frac_better/entropy及六字段 θ 曲线
& $py scripts/t9_report.py --run runs/t9_a_cert_s1234 --output results/t9_training_curves
```

T9-A 与 K*、T8-A、L_v6 的配对保留实际分母（K*/A: Chu270；L: Chu180），同时报告 mean/p90/worst、分层、LP/控制器/总计时、六字段 θ 方差和终盘变化、I1 gap 分解及逐步自身 regret/blocked regret/dual reoptimization gain。仍只按固定合成验证选 best，初始模型可成为 best；不使用 Chu 选 checkpoint。正式 fit 要求门控 SHA 和 decision 凭据，不支持 PPO 或 v6 warm start。

### T9-4：既有 checkpoint 的 core 前缀，仅推理诊断

```powershell
& $py scripts/run_t9_measure.py --tasks prefix --device cuda:0 --output results/t9_prefix
```

δ=0.05/0.1/0.2，A/B 各覆盖完整270，G保留真实180。前缀批次内逐物品检查可行性，既不重求 LP 也不调用控制器；首次无 eligible 项后永远进入 core 阶段，每步一个 LP/控制器调用。默认关闭，训练路径从未启用。报告逐例收益、保留率及调用数。完整作业耗时本轮未测，不把本轮小例子的节省宣称为普遍加速。

## 5. 尚未验证和交付记录

正式 T9-0、完整字典校准/oracle、T9-A 2500更新、完整 Chu270 新臂评估和 T9-4 保留率均交给用户运行。未执行 T9-5，未改写研究纲领文件。没有宣称字典成功、学习优于常数或已达目标 gap。训练时间估计尚不涵盖可能选定的扩展头；基本 cert 未过正式门控时不应启动训练。

旧项目验证命令：`git --no-optional-locks -c safe.directory=D:/PhD_Doctor/Study/Codes/RL_ModelInhance -C D:/PhD_Doctor/Study/Codes/RL_ModelInhance status --porcelain=v1`，前后输出均为空。指导/进展 `c783526`、代码 `cf5f4d5`、执行说明分开提交；实验产物不入 git。
