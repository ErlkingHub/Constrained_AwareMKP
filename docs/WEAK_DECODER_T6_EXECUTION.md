# T6：单轨迹相位调度替换交付

日期：2026-11-07。研究问题、判断与后续门控以 Claude 的 [T5 结论及 T6 指导](T5_RESULTS_AND_T6_GUIDANCE.md)为准。本轮交付推理代码及必要验证，完整 T6 实验由用户运行。没有训练，也没有运行或修改 SMS/sampleK/ILS。

## 1. 分支与文件

从 T5 的 `b3c16ef` 创建 `codex/t6-phase-schedule`。指导文档独立提交为 `324d1c2`，代码为 `3f6fedc`；本执行报告另作 docs 提交，编号见分支日志。

| 文件                      | 修改                                                                        |
| ------------------------- | --------------------------------------------------------------------------- |
| `scripts/t6_schedule.py`  | 三种可部署时钟、分位数节点拟合、分段线性插值、S1–S5 actor、完整实例二折诊断 |
| `scripts/run_t6_phase.py` | 复用 T5 数据/基准、重新采集拟合迹、单轨迹评估、逐实例落盘与旧项目只读审计   |
| `scripts/t6_report.py`    | 双参照配对表、回收率与 M×N 分层、时钟/分组/价格对照、R² 和报告校验记录      |
| `tests/test_t6_phase.py`  | 指导要求的两项测试，旧模块只进入独立子进程                                  |
| `README.md`、本报告       | 当前入口、验证范围和用户运行命令                                            |

新项目的模型、训练器、解码器和 T5 实现均不修改。所有新推理为 `stochastic=False`，每例每臂一条 greedy 轨迹；没有 `--train`、采样数或搜索参数。`results/`、`runs/`、`checkpoints/`、`cache/` 不入 git，不使用 `git add -f`。

## 2. 数据、隔离与复用

- 旧源码默认路径 `D:\PhD_Doctor\Study\Codes\RL_ModelInhance`；检查点默认本项目 `checkpoints/lp_dual_v6/mkp_M{5,10}_best.pt`。
- 输入复用 `results/t5_v6/instances_eval.npz`（128 例 N100）和 `instances_trace.npz`（同种子 `20261107` 的 144 例）。不重新生成数据或 MILP 参考。
- 默认 T5 基准在 `results/t5_v6/attribution/`。要求 T5 已完成、非 smoke、180 例 L 全部复现，并核对导出 manifest、实例值/参考/哈希、旧源码、两个 checkpoint SHA 和模型配置。原始结果还须匹配 T5 `report_manifest.json` 保存的输入 SHA，避免误用改动过的参照文件。
- T6 重新在合成迹集上运行 L，只增加记录每个实际决策前的三种时钟。全部拟合和交叉拟合只接受 `synthetic_trace`，不使用评估集或 Chu。
- Chu 的 L/G/C/D、合成集的 L/G/D 逐实例直接复用 T5 原结果，不重跑。**T5 没有评估合成 C，因此不补造或补跑该参照。** S4 需要的类别常数 C 从 T5 的 `theta_means.json` 读取。
- 旧版推理在独立进程内将 `legacy_root` 放在 `sys.path[0]`，验证所有 `mkp_*` 实际来自旧项目；只通过专用 T5/T6 工具计算 θ。统计另起进程，避免同名模块冲突。
- 关闭字节码写入，不调用会写旧目录的旧 Benchmark 主入口；Git 审计禁用 optional locks，只在当前项目写结果。前后比较旧项目 commit/branch/status 及九个源码文件 SHA。

## 3. 实际协议

### 3.1 时钟、节点与字段

时钟全部由当前决策前的状态计算：

- `tau_step = selected_count / N`；没有使用最终轨迹长度 T。
- `tau_cap = 1 − mean(remaining / capacities)`。
- `tau_feas = 1 − feasible_count / N`。

三者检查有限且在 [0,1]，并在新轨迹内检查单调不减。只容忍浮点端点的 1e-12 误差，再截到端点；没有使用未来信息。

每个分组池内按实际决策等权，在激活后的 θ 空间中拟合。时钟分位数用 numpy 的线性分位数规则分 10 段，节点为非空段的时钟中位数，值为段内均值。**补充的确定性数值规则：** 跳过空段；重复节点按样本数加权合并，避免除以零；一节点表作为常数表。最多保留 10 个不同节点。节点间线性插值，范围外使用端点值，节点处准确返回节点值。

全部 8 个标量（包括 temperature）和 companion 每个分量共用该分组/时钟的节点。interactions 取分组常量均值，并对称化；检查各字段区间、均值价格、interactions 上三角预算和 companion 总预算。插值为凸组合，每次部署输出也进行范围断言。

**不按资源位置平均价格向量。** 拟合函数从均值计算中直接排除 resource_prices，调度表中设为全 1。S5 的价格来自原 actor 在当前 S5 状态上的实时输出；其余 θ 输出由调度给定。为获得价格而运行原 actor 的全部计算开销计入 S5。

### 3.2 五个臂

| 臂  | 分组           | 时钟     | 字段                                                                         |
| --- | -------------- | -------- | ---------------------------------------------------------------------------- |
| S1  | M,N,名义容量比 | tau_step | 所有标量 + companion，价格均匀                                               |
| S2  | 同上           | tau_cap  | 同上                                                                         |
| S3  | 只按 M         | tau_cap  | 同上，interactions 也按 M 取常数                                             |
| S4  | M,N,名义容量比 | tau_cap  | 仅 profit_exponent/scarcity/lp_primal/companion；其余保留 T5 C，价格改为均匀 |
| S5  | M,N,名义容量比 | tau_cap  | 同 S2，价格替换为实时 L 控制器的价格通道                                     |

三种时钟、两种分组都生成表和交叉拟合诊断；tau_feas 只做指定的模型诊断，不增加新的评估臂。旧 decoder、LP、可行性、终止条件和 greedy tie-break 均不改变。

### 3.3 二折 R²、配对与耗时

- 在每个类别内按实例 id 排序后交替分到两折，完整轨迹作为分割单位。正式 8 例每类是 4/4；拟合某折时，分位数、节点和所有均值都只使用另一折。
- **R² 口径明确区分两列：** `raw_R2` 是原始值上的 1−SSE/SST；`within_trajectory_R2` 对观察序列和预测序列分别减去该条留出轨迹自身均值，再计算 1−SSE/SST。后一列只用于轨迹内动态诊断，中心化从不进入部署时钟或 θ。零方差字段留空，不强填 R²。
- 8 个标量、companion 每个分量及其总和均报告。当前两个 v6 配置各有 3 个 companion 分量，所以共 12 个统计量 × 2 分组 × 3 时钟 = 72 行诊断。
- 对所有 S 臂分别与 L 和 G 做同实例配对，报告平均/中位 gap 差、胜平负、双侧 Wilcoxon p；p 未作多重比较校正。均值/p90/worst 和回收率均按 M×N 分层。
- 回收率取 **平均 gap 的差之比** `(mean(G)−mean(S))/(mean(G)−mean(L))`，不是逐实例比率的均值；不截到 [0,1]。分母近零时留空，非正分母另列标记，避免误读。
- 另报 S2−S1、S3−S2、S4−S2、S5−S2，供 Claude 判断时钟、分组和价格的作用；不自动选臂或判研究门控。
- 记录 LP 实际求解耗时与总构造耗时。新臂耗时含时钟/插值/断言和 S5 网络，剥离拟合迹复制及文件序列化；同步记录会影响延迟，按 instrumented timing 解读。复用的 T5 耗时保留原运行环境，不据此自动宣称跨环境加速比。

## 4. 已执行验证

仅新增、运行指导要求的两项测试，**2/2 通过，11.340 s，未跳过**：

1. 读取实际 T5 G 和 v6 M5/M10 checkpoint，在四个 N9/N11 合成小实例上，常数节点调度与 T5 固定 θ 注入的 selected/sequence/objective 逐位一致。此回归夹具保留完整 G（含 G 的非均匀价格）；正式 S1–S4 仍严格均匀价格，不把回归夹具当作评估臂。
2. 插值在节点、区间中点、超出两端时准确；三种时钟在实际合法状态更新中始终在 [0,1] 且单调不减；重复分位数的一节点表合法，拟合价格为全 1。

有限 smoke 已完成（`results/t6_code_checks/smoke/`）：

- 只在 N100 的 M5/10 × 三容量比各两例上重采 L，共 12 条拟合迹、617 个实际决策；两折每类各一例。
- 每个评估数据集只选 M5/M10 各一例，S1–S5 共 20 次新单轨迹推理；直接复用 T5 的 14 条参照结果，不重跑参照。
- 34 行结果、60 个实例配对、72 行交叉拟合诊断，报告生成成功。所有新构造通过目标值、可行性和极大性独立检查，时钟范围和单调性断言通过。
- `status=complete, smoke_only=true, legacy_unchanged=true`。旧项目前后 HEAD 均为 `4f935551e62d91d55d998fa6255a67f9b0fafeb7`，branch/status 和源码 SHA 一致；`git -C <legacy_root> status --porcelain=v1` 输出为空。

本轮未运行正式 144 条拟合迹、128 例合成和 180 例 Chu 的完整 S1–S5 评估。smoke 的 gap/R²/回收率不作为研究结论；是否可由相位表替代控制器、T7 采用哪种 θ，均待正式结果由 Claude 判断。

## 5. 用户运行

使用已有完整 T5 产物，不需要重跑导出或 T5 基准：

```powershell
python scripts/run_t6_phase.py --device cuda:0
```

默认读取 `results/t5_v6/`、`results/t5_v6/attribution/` 和本地 v6 best checkpoints；输出到 `results/t6_phase/`，拒绝覆盖。其他位置可显式指定：

```powershell
python scripts/run_t6_phase.py --legacy_root D:\PhD_Doctor\Study\Codes\RL_ModelInhance --checkpoint_dir checkpoints/lp_dual_v6 --instances_dir results/t5_v6 --t5_results results/t5_v6/attribution --output results/t6_phase --device cuda:0
```

正式新增推理预算为 **144 条拟合 L + (128+180)×5=1540 条 S 评估**。每条均单轨迹；没有重新推理任何评估参照。主要给 Claude 的文件：`report.md`、`summary.csv`、`paired_summary.csv`、`recovery.csv`、`contrasts.csv`、`crossfit_R2.csv`、`schedules.json` 和 `meta.json`；原始 `traces.jsonl`、逐例结果和拟合进度也保留。

只重做统计与复现有限检查：

```powershell
python scripts/t6_report.py --input results/t6_phase
python -B -m unittest discover -s tests -p test_t6_phase.py -v
python scripts/run_t6_phase.py --limit 2 --output results/t6_code_checks/smoke_new --device cuda:0
```

`--limit 2` 仍要求原始完整 T5 数据/基准，但缩小为上面的 N100 smoke 设计，所有产物标为 smoke。测试可用 `T5_LEGACY_ROOT`、`T5_CHECKPOINT_DIR`、`T5_RESULTS_DIR` 环境变量调整外部依赖路径；依赖缺失时明确跳过，本机已实际执行。
