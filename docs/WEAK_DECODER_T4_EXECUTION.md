# T4：SMS 与 max@K 提议分布学习交付

日期：2026-11-06。研究分析与任务定义见 Claude 的 [T3 结论及 T4 指导](T3_RESULTS_AND_T4_GUIDANCE.md)。本报告记录工程实现和验证，不将代码检查或两实例 smoke 作为正式实验结论。

## 1. 分支与交付范围

- 从 `codex/t3-residual-lp-basis` 的 `814aaab` 创建 `codex/t4-strategy-multistart-maxk`。
- 指导文档单独提交：`2abcaa9`；代码提交：`894ebb6`；本执行报告另作 docs 提交，最终编号见分支日志。
- T3 的 LP/block 扩展保留且默认关闭。T4 固定使用 F0：root 价格、`price/alpha`、归一化分数、hard mask、α=1。
- 本轮不运行正式 T4-a、不启动 T4-b 训练或完整 Chu 测评。用户先运行 T4-a，Claude 根据结果判断门控，再决定是否训练。
- `results/`、`runs/`、`checkpoints/`、`cache/` 均被忽略。取消跟踪历史 `results/t1p_lottery_check/output.txt`，保留本地原文件；没有使用 `git add -f`。

| 文件 | 核心修改 |
|---|---|
| `mkp_config.py` | SMS 评测配置、可选 σ 头及基准尺度、固定验证噪声种子 |
| `mkp_sms.py`（新增） | 独立 latent θ 抽样、中心候选、θ 前缀、hybrid 的 θ 优先顺序和逐轨迹物品种子 |
| `mkp_evaluation.py` | `thetaK`、`hybAxB` 及 `+ILS`；旧 stage 字典保持原样；`learned_mu_only` 臂 |
| `Benchmark_zeroShot.py` | SMS 候选评测、可选返回全部候选、T4 三臂、协议 SHA；保留旧 sampleK 语义 |
| `mkp_Model.py` | 默认关闭的 σ 头、张量 σ 的 detached-action logq、静态提议读取与 σ 覆盖 |
| `mkp_Trainer.py` | `theta_maxk` 留一法优势、独立训练提议、诊断日志、best-of-16 验证与检查点选择 |
| `scripts/run_t4a_sms.py`（新增） | 正式网格、合成集选择、Chu 确认、配对/分层/多样性/耗时报告与逐实例落盘 |
| `scripts/run_t4b.py`（新增） | 严格读取正式选择；默认只生成配置，显式 `--train` 才训练 |
| `tests/test_t4_sms.py`（新增） | 指导第 7 节要求的恰好 3 项测试 |

## 2. 实际协议与补充规则

### 2.1 候选与随机数

- θ 族固定中心为 `v=0, z=logit(0.25)`，尺度为 `s×(0.05,0.3)`；dims 为 `u/a/ua`。第 0 条为中心 greedy，其他 θ 独立，无 antithetic。
- 每个 θ 索引使用 `paired_seed(seed, instance_id, "sms", f"{tag}:{index}")` 的独立 CPU generator，按资源后 α 的顺序生成 M+1 个标准正态数。先生成所有 θ，再切分构造批次。索引种子和激活后的前缀不依赖总 K。
- hybrid 按 θ 优先顺序展开；整个池仅第 0 条强制中心 greedy，其余均为物品采样。每条物品轨迹有独立种子，逐条调用，保证改变分块或扩大 Kθ 不改变已有候选；其调用开销计入构造耗时。
- 历史 `sampleK` 仍是 K 条采样，不隐含 greedy。T4-a 专用入口显式启用 `centered_samples=True`，用中心 greedy 加 K−1 条采样，与 θ 族对称。两种语义在产物中区分。
- T4-a 每个 θ/sample 配置每例只构造一次 K256，按前缀报告 1/4/16/64/256；额外读取 ua/s1 的 K32 锚点。前缀行的耗时是实测完整池耗时，`prefix_runtime_measured` 标明是否为完整预算，未冒充单独小 K 耗时。

### 2.2 选择与报告

- 合成 N100 共 128 例，复用 T3-a 的生成顺序和 MILP 缓存；仅此集合选择 Θ*、S*。Chu 固定 180 例（M5/10 × N100/250/500），每组必须 30 例且 id 不重复，否则正式运行报错；只作确认。
- Θ* 和 S* 分别最小化 K16 与 K64 的平均 gap 之和；并列保留网格顺序。
- **用户已确认的补充规则：** H* 按各混合方案在其原始总预算 K=16/64/256 下相对 S* 同预算的平均 gap 差选择；并列优先较小预算，再按网格顺序。Chu 固定所选 Ks，扩展 Kθ 到总计 256 条，按轨迹前缀报告；`selection.json` 同时保存原始 H* 和实际扩展的 Kθ/Ks。该扩展改变了原始 Kθ，必须在正式报告中注明。
- 合成集 3 个 H 方案仅报告各自原生预算点。因此输出包含 17 个配置，但不伪造 17×5 矩形表中的其他 H 点。Chu H* 的小 K 为 θ 优先顺序的轨迹前缀，可能只覆盖部分 θ。
- `paired_summary.csv` 报平均/中位 gap 差、胜平负、双侧 Wilcoxon p；K16/K64 另按 M×N 分层。p 未作多重比较校正，单种子结果不推断训练种子稳定性。
- 逐实例记录于 `progress.jsonl`，全部构造候选及目标值无损存于 `candidates.jsonl`，汇总为 `per_instance.csv`、`summary.csv`、`paired_per_instance.csv`、`paired_summary.csv`。
- 正式锚点若偏离预期超过 0.1 pp，则保存已有数据并中止在 Chu 确认之前，供排查；smoke 标记为 `not_assessed_smoke`，不解释这些参考数字。

### 2.3 T4-b 的学习边界

- 仅 static 控制器，均值头沿用旧初始化。资源 σ 为并列的单层小头，标量头新增 σ_α 通道；新增权重/偏置为 0。`σ=σ0 exp(1.5 tanh(s))`，不扰动的维度固定为 0、不产生 logq 项。
- σ 头开启时，标量及 σ 读出使用 float32，保证 AMP 下初始中心和注册 σ0 一致；关闭时不增加参数、不改变旧分支计算。
- 可学习张量 σ 的 logq 包含 `−log σ`，统一省略每个活跃维度的 Normal 常数 `−log(2π)/2`；动作抽样 detached。旧固定浮点 σ 路径保留原有常数约定。
- 训练 K16 为独立抽样，不强制中心，不用 antithetic、不加熵项。优势 `max(R)−max(R_except_k)` 只在唯一赢家上非零，并列全零。`R_center` 是额外无梯度诊断。
- 验证和评测 K16 按指导 6.3 包含均值 θ 的中心 greedy 加 15 个独立提议；验证噪声用 `val_noise_seed=20261106` 加集合索引固定，epoch −1 与后续验证使用同一噪声。best.pt 按这个 best-of-16 gap 选择。**训练的独立 16 抽样与验证的含中心 16 候选并非同一候选协议。**
- 日志增加 `R_maxK/R_center/frac_adv_nonzero`、两类 σ 的均值/极值、均值头 `|u|`。`learned_mu_only` 保留 μ，覆盖为注册 σ0；fixed 同样读取注册 dims/σ0，不再额外乘评测默认尺度。

## 3. 已执行的验证

新增 3 项测试均通过：

1. `theta1` 与 fixed greedy 逐位一致；theta256 前 16 条与 theta16 完全一致；分块 64 与 7 的候选/目标值完全一致；旧 stage spec 与 `814aaab` 逐字段相同。
2. 构造奖励矩阵的唯一赢家、最优次优差和并列情况正确；maxk + antithetic 在训练开始前报错。
3. 张量 σ logq 与 `Normal.log_prob` 仅差统一常数，σ 梯度有限；实载 T1-P static/dynamic 检查点，与 `4d3c376` 代码在小实例上的 greedy 解及 log likelihood 逐位一致，state_dict 字段无新增。禁用 u 扰动时 σ_u=0，σ_α 初值准确。

现有及新增测试合计 **47 项通过**（约 6.9 s，CPU）。另用小内存实例核对 sigma-head 初始 fixed/learned_static/learned_mu_only 候选一致，maxk 单次训练步的损失/梯度有限，验证不受训练全局 RNG 推进影响。这些均为接口检查，未运行正式训练。

真实 N100 的 `--limit 2` smoke 已完成，目录 `results/t4_code_checks/sms_smoke_real_n100/`：

- 合成 M5/10 各一例；Chu id1/id91 各一例，均为 N100，未截断物品。
- meta 与 selection 均为 `status=complete, smoke_only=true`；17 个主配置、46 个实际配置/实例候选池、10,144 条候选、192 条前缀表行。
- 候选预算及 bitpack 长度正确；目标值/耗时有限；候选可行且极大；前缀 best 单调且不低于中心。两条低预算 ILS 路径均跑通。
- ILS 仅在 smoke 中减为 rounds1/iters1/pool2/drop2；正式配置仍为 rounds100/iters200/pool16/drop32。
- `run_t4b.py --selection results/t4_code_checks/sms_smoke_real_n100/selection.json` 如期报错，未生成训练配置、未启动训练。
- 本次 smoke 曾在 θ/sample 池完成后中断并显式恢复，复用了 28 个已落盘池。恢复仅用于有限 smoke；没有重复正式实验。

额外修复 fixed-only 入口忽略 `--static_checkpoint` 的问题，并在现有评测测试方法中补断言；该方法复跑 1/1 通过，未新增第 4 项测试。完整 T4-a 未运行，以上计数只说明工程路径跑通。

## 4. 用户运行

以下命令在仓库根目录、具备项目依赖的 Python 环境中执行；源配置为本地既有 `runs/t1p_static_s1234/config.json`，不入 git。

```powershell
# 正式 T4-a：128 个合成 N100 + 180 个 Chu，不训练。
python scripts/run_t4a_sms.py --device cuda:0
```

默认输出 `results/t4a_sms/`。脚本拒绝覆盖已有目录；再次运行须选择新的 `--output`，并将对应选择路径传给 T4-b。正式实验保留原配置的 ILS 预算（rounds100 / iters200 / pool16 / drop32），这部分可能明显慢于构造。

将 T4-a 的 selection、汇总和配对表交给 Claude 判断门控。获准后：

```powershell
# 仅准备配置，不训练。
python scripts/run_t4b.py

# 由用户明确启动训练：K16、B64、10×250、seed1234。
python scripts/run_t4b.py --train --cuda_device 0

# 训练后完整 Chu 测评，best 与 last 分别运行并报告。
python Benchmark_zeroShot.py --arm fixed,learned_static,learned_mu_only --static_checkpoint checkpoints/t4b_static_maxk16_s1234/best.pt --stages theta16,theta64,theta64+ILS
python Benchmark_zeroShot.py --arm fixed,learned_static,learned_mu_only --static_checkpoint checkpoints/t4b_static_maxk16_s1234/last.pt --stages theta16,theta64,theta64+ILS
```

T4-b 读取选择文件中的 dims/scale；拒绝缺失、未完成、smoke、非 128 例选择，以及源配置 SHA 不匹配。默认只写配置，门控不会自动触发训练。评测中的 fixed 自动匹配检查点注册的 Θ*，无需猜测尺度参数。

smoke 复现命令：

```powershell
python scripts/run_t4a_sms.py --limit 2 --device cuda:0 --output results/t4_code_checks/sms_smoke_new
python -m unittest discover -s tests -v
```

`--limit 2` 使用真实 N100 的 M5/10 各一例，Chu 也各一例；只为接口检查，ILS 自动减为 rounds1/iters1/pool2/drop2。只有显式 `--resume-smoke` 可恢复中断的有限 smoke 池，检查源配置、实例身份及 limit/device 等一致；正式运行不提供恢复功能。smoke 选择永远不能用于 T4-b。

## 5. 尚未验证

正式网格性能、运行时间、统计门控、Θ*/S*/H* 选择、完整 ILS 优势，以及 max@K 学习是否得到有效提议分布，均须由用户正式运行后再判断。本轮没有新的实验收益结论。
