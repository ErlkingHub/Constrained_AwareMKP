# T5：旧版 v6 增益归因代码交付

日期：2026-11-07。研究判断与门控以 Claude 的 [T4 结论与 T5 指导](T4_RESULTS_AND_T5_GUIDANCE.md)为准。本轮只实现推理归因；不启动训练，完整实验由用户运行。

## 1. 分支、范围与文件

当前 T5 分支为 `codex/t5-v6-attribution`，承接 T4 的 `d314451`；指导文档已单独提交为 `18e0452`，代码提交为 `3a49910`。本执行报告另作 docs 提交，编号见分支日志。

| 文件 | 用途 |
|---|---|
| `scripts/t5_export_instances.py` | 在新项目进程中导出相同的 N100 评估实例、独立的拟合迹实例集及参考值/哈希 |
| `scripts/t5_v6_attribution.py` | 独立旧版进程；检查点/数据校验、L 重现预检、θ 记录、D/G/C/I 和字典消融 |
| `scripts/t5_theta.py` | 激活后 θ 的均值、范围/组合预算断言、默认 D、常量 actor 注入及中性化 |
| `scripts/t5_report.py` | 独立后处理进程；配对统计、分层表、方差分解和 Markdown 报告 |
| `tests/test_t5_v6.py` | 指导第 4.6 节要求的两项测试，通过子进程隔离旧模块 |

新项目模型、训练器和固定执行器均不修改。T4-b 保留代码但本轮不训练；T5 也没有训练入口。实验产物位于被忽略的 `results/`，检查点不入 git，没有使用 `git add -f`。

## 2. 路径与只读边界

- 旧项目：`D:\PhD_Doctor\Study\Codes\RL_ModelInhance`。本次开始时分支为 `40_lp-dual-bipartite-bound-shaping`，HEAD 为 `4f935551e62d91d55d998fa6255a67f9b0fafeb7`，`git status --porcelain=v1` 输出为空。
- 检查点：当前新项目下的 `checkpoints/lp_dual_v6/mkp_M5_best.pt` 与 `mkp_M10_best.pt`。用户路径中的 `Zlb08.codex` 实际对应 `Zlb08\.codex`；四个 best/last 文件均存在，本协议只用 best。
- 历史基准：旧项目 `results/20261022_195937_430788/manifest.json` 和 `solutions.jsonl`。两个 best 的完整 SHA 必须与历史 manifest 一致；Chu 数据也核对历史 SHA。
- 不调用旧 Benchmark 的 `main/benchmark`，它会向旧目录写结果。只调用旧 `load_checkpoint/construct_solution/verify_solution` 等只读接口。设置 `sys.dont_write_bytecode=True`，旧源码不产生 pycache。
- 旧进程把 `legacy_root` 放到 `sys.path[0]`，不导入新项目的 `mkp_*` 模块；导出与统计分别在其他进程执行。后处理需要 scipy，旧推理进程只导入标准库、numpy/torch 和旧模块及本次专用 θ 工具。
- 运行前后记录旧项目 Git 状态，要求与初始状态相同；不修改全局 Git 配置。CLI 输出路径须位于旧项目之外。

## 3. 数据与 θ 协议

### 3.1 交换格式

默认导出到 `results/t5_v6/`：

- `instances_eval.npz`：相同生成器、相同 `val_seed=20240607`，先分别生成 M5/M10 的完整 64 例 N100 批次，合计 128 例；复用原 MILP 缓存。保存 incumbent/已证最优标记，不将未证 incumbent 称为最优解。
- `instances_trace.npz`：新的固定种子 `20261107`，M5/10 × N100/250/500 × 容量比 .25/.5/.75，每类 8 例，共 144 例；只拟合 G/C，不评估或选择消融。
- `manifest.json`：源配置/代码、生成种子、参考缓存与 NPZ 的 SHA、各类数量、T3/T4 已有 manifest 的实例哈希比较。

异构实例各存 `p_i(N), w_i(N,M), c_i(M)`，float64，JSON 标量 `metadata` 保存逐例 id/M/N/tightness/reference/data_sha256；使用 `allow_pickle=False`。新项目 float32 的整数原始值上转 float64，`r(M,N)` 转置为旧版 `w(N,M)`，不归一化。另保留 `source_data_sha256`，区分转置/上转后的交换格式哈希与原项目哈希。

### 3.2 臂与消融

- L：原始 learned_dynamic greedy；actor hook 保存每个实际决策的完整 11 字段 θ，不重新调用 actor。
- D：用每个 M 的实际 v6 配置，新初始化 actor 在 t=0 读 θ；换两个随机种子，逐字段断言相同。
- G：按 M 对拟合迹集的所有实际决策求均值；每个决策等权，长短轨迹不先归一为同权实例。
- C：同样在拟合迹集上按 (M,N,名义容量比) 求均值；Chu 容量比映射到 .25/.5/.75。
- I：本实例 L 轨迹的时间均值。使用了该实例原始 L 的轨迹，是事后诊断，不是可部署的静态策略。
- 所有均值均在激活后的空间中求；prices 检查均值接近 1 后归一为 1，interactions 对称化。除逐字段范围，还检查 interactions 的上三角总预算及 companion 的总预算。
- 固定 θ 仅替换 actor；feasibility 输出为 0。旧 decoder、残余 LP、状态更新、可行性和 greedy tie-break 保持原样。
- 在合成 N100 上以 G 为基点做 10 个逐项消融；prices 中性值为均匀，profit_exponent 为 1，其余项为 0，temperature 保留均值。G-min 只保留移除后平均 gap 变差 ≥0.05 pp 的项。G-F0 仅保留 G 的 prices/profit_exponent。

**G-F0 的解释限制：** 按指导实施该消融，但它与新项目 F0 并非同一个公式。v6 在 scarcity=0 时使用 w/C 的成本，新 F0 使用 root-LP 价格锚点并按剩余容量归一；v6 prices 也不是该 LP 锚点上的乘性修正。因此“应该接近 1.08%”只作为待检验预期，不能预先宣称等价或将差异直接归因于学习。

### 3.3 重现预检与报告

先跑 Chu L，再跑其他臂。逐实例对比历史 `method=learned_dynamic` 的 objective，保存不一致的 id、数量及平均 gap 差。正式全 180 例的平均偏差绝对值超过 0.02 pp 时中止，保留结果供排查；不自动继续其他臂。

全部结果逐实例落盘。报告包含各臂 gap 均值/p90/worst、相对 L（Chu）或 G（合成）的配对差、胜平负和双侧 Wilcoxon p，Chu 另按 M×N 分层。p 未做多重比较校正；单种子不代表训练种子稳定性。LP 与总推理时间分别记录，剥离 θ 复制和报告序列化时间；同步记录仍可能影响执行延迟，因此标为 instrumented timing。常量 actor 省掉网络开销是该臂本身的行为，不额外绕过 decoder/LP。I 的 `time_total_s` 仅为注入后的轨迹，另列 `time_with_required_prepass_s` 计入它必需的同实例 L 前置轨迹；G/C 的离线拟合成本另存于 `fitting_progress.jsonl`。

方差分解只使用 Chu L 的逐决策迹，按等权决策的嵌套平方和分为类别间、类别内实例间、轨迹内三部分，验证三者之和等于总平方和。逐标量，加上 prices 的总体 std、interactions 的矩阵迹（对角线之和，区别于上三角总权重）、companion 的和；完全常量字段的方差份额留空。

## 4. 验证情况

两项必要测试均通过（`test_t5_v6.py`，CPU，10.782 s）：

1. 实载 M5 best checkpoint，在 N9 的同一状态读取 L 的 θ 后通过常量 actor 注入，candidate owners/items/logits 与原结果逐位相同，feasibility 辅助输出为 0。
2. 实载 M5/M10 best checkpoint，在少量 N9/N11 状态上检查 G/C 激活后均值；字段范围、prices 均值为 1、interactions 对称及上三角/companion 总预算正确。D 在两个初始化种子下逐字段完全一致。两个测试的旧模块仅存在于字节码关闭的独立子进程。

导出 smoke 已完成：2 例真实合成 N100 评估数据（M5/10 各一例），加完整 18 个类别各一例的拟合迹输入。保持正式批次的 RNG 消耗顺序，T3-a/T4-a 各两组的源实例哈希全部匹配。两个参考中一例未证最优，正确保留为 MILP incumbent，没有冒充 optimum。

有限推理 smoke 已完整结束（`results/t5_code_checks/attribution_smoke/`）：

- 18 次拟合迹推理，以及 40 次评估（合成 15 臂 × 2 例，Chu 5 臂 × 2 例），全部为 greedy。所有构造均通过独立 float64 可行性、极大性和目标值重算检查。
- Chu id1/id91 的 L 与历史目标值完全一致：不一致数 0/2，平均 gap 偏差 0 pp。产物明确标记 `partial_smoke`、`gate_assessed=false`；这不是完整 180 例的重现结论。
- 已生成 `report.md`、逐实例/汇总/配对/分层 CSV，以及 11 个 θ 统计量的方差分解。报告器检查臂/实例覆盖、无重复配对、差值和参考一致、数值有限、迹连续与平方和分解相加正确；I 的前置 L 耗时也另列。`report_manifest.json` 独立记录报告阶段实际源码、原始输入及报告产物的 SHA，避免把推理开始时的代码快照误作后处理代码版本。
- `meta.status=complete, smoke_only=true, legacy_unchanged=true`。旧项目前后 HEAD/branch/status 和九个源码文件哈希完全相同；`git -C <legacy_root> status --porcelain=v1` 前后均为空。没有旧项目文件改动或提交。

未运行正式 144 例拟合/128 例消融/180 例重现与对照，不报告新的研究收益或门控结论。

## 5. 用户运行

在本仓库根目录、具备 torch/numpy/scipy 的环境中：

```powershell
# 第一步：新项目进程，仅导出数据和 MILP 参考。
python scripts/t5_export_instances.py

# 第二步：独立旧版进程，无训练。数据、检查点和历史基准均会校验。
python scripts/t5_v6_attribution.py --legacy_root D:\PhD_Doctor\Study\Codes\RL_ModelInhance --checkpoint_dir checkpoints/lp_dual_v6 --device cuda:0
```

脚本拒绝覆盖已有导出/推理目录。需要另存时给导出器 `--output <新导出目录>`，并在旧 runner 中通过 `--instances_dir <该目录>`、`--output <新结果目录>` 配套指定。正式运行不得使用 smoke 的交换文件。

默认推理输出在 `results/t5_v6/attribution/`。主要交给 Claude 的材料为 `report.md`、`summary.csv`、`paired_summary.csv`、`theta_variance.csv`、`reproduction.json` 和 `selection.json`；原始结果与逐步迹也保留。若只需重新统计已有推理产物：

```powershell
python scripts/t5_report.py --input results/t5_v6/attribution
```

有限检查的复现命令（测试使用本地实际旧项目及检查点）：

```powershell
python -B -m unittest discover -s tests -p test_t5_v6.py -v
python -B scripts/t5_export_instances.py --limit 2 --output results/t5_code_checks/export_smoke_new
python -B scripts/t5_v6_attribution.py --limit 2 --instances_dir results/t5_code_checks/export_smoke_new --output results/t5_code_checks/attribution_smoke_new --device cuda:0
```

测试支持环境变量 `T5_LEGACY_ROOT` 和 `T5_CHECKPOINT_DIR` 指定其他机器上的实际位置；缺少这些不入 git 的外部依赖时会明确跳过。本机上述两项均实际执行通过，未跳过。`--limit 2` 将 2 例评估、18 例拟合及 2 例 Chu 的所有结果标记为 smoke；正式 runner 拒绝把这种交换数据当作完整实验。旧项目始终只读；完整评估及研究门控由用户与 Claude 执行和判断。
