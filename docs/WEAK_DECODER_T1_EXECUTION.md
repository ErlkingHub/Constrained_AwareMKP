# 弱解码器迁移执行报告：T0 工具与 T1 代码

日期：2026-11-02。执行依据：[CODE_GUIDANCE_WEAK_DECODER_MIGRATION.md](CODE_GUIDANCE_WEAK_DECODER_MIGRATION.md)。
起始代码提交：`738201e`（本次未提交代码；各次运行的 meta 保存其实际 commit、diff 与未提交源码）。

## 范围与阶段状态

| 阶段 | 当前状态 | 证据边界 |
|---|---|---|
| T0 | 旧源码隔离加载、a/b/c/c_fullfixed 干预工具及缩小预算集成检查完成 | 尚无完整 180 实例 × sample256 × 原 ILS 预算的结果，不能定量归因历史 0.074% |
| T1 | G0+G1 代码、对照评测和工程验证完成 | 尚未完成三个独立训练种子、静态/动态匹配预算正式实验，不能声称学习带来显著增量 |
| T2 / T3 / T4 | 未实施 | 遵循逐组消融、逐阶段回报的要求，未把阻断、残余 LP 或表示改造混入 T1 |

指导文件要求 T0 先行、各阶段单独回报。本次先完成可审查的 T1 接口和工程检查；
T0 全量因果拆解仍是未完成实验，不能把工具试跑当作其验收，不能据此决定 G2/G3。
未启动文档中建议的 40 epoch × 250 step × 2 个训练臂 × 3 种子的正式训练。

## 代码修改

- `mkp_Model.py`：删除逐物品残差、基元混合和状态温度头；保留物品–资源关系编码；
  资源读出 `u_max*tanh` 后去均值，alpha 初始化为 1。参数化接口与 N 无关，默认 M+1。
  执行器只有显式对数代理比公式，在当前可行集合选择，无搜索或可训练参数。
- 温度是策略拥有的全局标量，支持 learned/fixed，初值 1/3，不计入 theta。
  固定臂跳过编码器及控制器；static 模式在 t=0 读一次 theta，保留整条轨迹的训练梯度。
- `mkp_config.py` / `mkp_env.py`：主方法默认 hard_mask，奖励 `profit/UB`；
  perceive/soft_mask 仍是具名对照模式。新增 theta/temperature 配置及显式 CPU 选项。
  T1 请求 block/lp 组或 residual anchor 会直接报错。
- `mkp_Trainer.py`：保留 REINFORCE + 同实例 K 均值基线；记录 u/alpha/T、梯度状态和数据 SHA。
  只用固定合成验证集选 checkpoint。严格拒绝旧 delta checkpoint 和不兼容的静态/动态恢复。
- `mkp_reference.py`：HiGHS MILP 合成验证参考和缓存；区分限时 incumbent 与已证最优值，
  校验缓存设置、二元性、可行性、目标值和上界。MILP 不进入执行器，也不用 Chu 选择权重。
- `mkp_artifacts.py`：保存 commit/diff、未提交源码、完整配置、种子、数据/参考 SHA 与 checkpoint SHA。
  恢复运行会核验历史 best 的归属，避免把共享目录的其他模型当作本次 best。
- `Benchmark_zeroShot.py` / `mkp_evaluation.py`：fixed、fixed_dualgreedy、untrained、
  learned_static、learned_dynamic；greedy、sampleK、greedy+ILS、sampleK+ILS 分阶段输出。
  配对要求同实例、种子、执行器协议和推理预算，包含 Wilcoxon、百分点差、M/N/alpha 分层、worst/p90。
  `mkp_refine.py` 原样保留。
- `legacy_t0.py`：从不可变 `6a4cf82` 加载旧源码，严格加载原 checkpoint，保存源码快照。
  a=原策略；b=delta0；c=delta0+u0+w初值，**保留 learned kappa**；
  c_fullfixed 额外把 kappa 设初值。c 的采样结果不是完全固定启发式，报告必须区别。
- README 更新当前接口；旧算法文档保留历史内容并增加快照提示。

## 已执行验证

解释器：`D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe`，PyTorch `2.11.0+cu128`、SciPy `1.15.3`。
系统 Python 缺少 PyTorch，因此使用已有环境，未安装新依赖。

静态检查通过：训练/评测 `--print_config`、Python 编译、`git diff --check`。
已验证默认 theta 在不同 N 下始终 M+1，decoder 无逐物品学习头或搜索。
全套回归测试 **31 项通过**（`python -m unittest discover -s tests -v`，3.716 秒）。
覆盖显式公式、固定复现、排列等变（避开分数相等的歧义）、static 冻结、有限梯度、
已完成行、非法 theta 拒绝、MILP 缓存、断点恢复、预算/协议配对与随机种子。
测试过程中发现并修复了 masked -inf 在可学习温度除法中产生 NaN 梯度的问题。

### 两套独立 smoke

命令：`python scripts/smoke_weak_decoder.py`。脚本会拒绝覆盖已有 smoke 产物。

| 项目 | 设置 / 结果 |
|---|---|
| 更新方式 | step、static 分别从头训练，同 seed=1234 |
| 训练 | 各 2 epoch × 20 step，B=2，K=16 |
| 模型 | embed=32，heads=4，encoder/controller 各 1 层，edge_hidden=8 |
| 实例 | 训练 N=20/30，M=3/5；合成验证 N=20，M=3/5，每组 2 个 |
| LP / 验证参考 | dual_iters=30，HiGHS MILP，限时 2 秒 |
| 结果 | 共 80 步 loss/gradient 均有限，两套验证解均可行且极大 |
| 耗时 | step 15.96 秒，static 2.85 秒；首次加载成本不同，不能当成本消融 |

产物：

- `runs/weak_decoder_t1_smoke_dynamic/{config,meta,smoke_report}.json`、训练/验证 CSV；
- `runs/weak_decoder_t1_smoke_static/` 对应产物；
- `checkpoints/weak_decoder_t1_smoke/weak_decoder_t1_smoke_dynamic/best.pt`；
- `checkpoints/weak_decoder_t1_smoke/weak_decoder_t1_smoke_static/best.pt`。

另使用默认完整网络（embed128，encoder3/controller2），在 CUDA:0/bf16、B2/N20/M5/K4
上检查 step/static 各一次反向传播，loss 和梯度均有限，解可行且极大；记录在
`results/t1_cuda_smoke.json`。这是 GPU 兼容检查。

### 固定臂：Chu 180 个实例，单轨迹 greedy

复现命令（本次 CPU 线程数设为 2）：

```powershell
python Benchmark_zeroShot.py --arm fixed,fixed_dualgreedy --stages greedy --device.force_cpu true --eval.results_dir results/t1_fixed180
```

默认 root LP 300 次迭代，M=5/10，N=100/250/500。全部 360 个解通过可行且极大审计。

| 臂 | 数量 | mean best-known gap (%) | p90 (%) | worst (%) |
|---|---:|---:|---:|---:|
| fixed（新动态余量公式，固定 theta） | 180 | 0.61070129 | 1.23424737 | 3.56061633 |
| fixed_dualgreedy（历史静态容量公式） | 180 | 0.71124146 | 1.55399836 | 3.93687131 |

上述是**无学习基线**；不是学习改进，不是全局最优 gap。整次运行约 32.34 秒。
结果：`results/t1_fixed180/20261102_140130_251675/`，含 per_instance、summary、paired CSV、
meta、source diff、逐实例解向量及 theta 轨迹。

### 评测阶段集成

T1：五臂 × `greedy,sample3,greedy+ILS,sample3+ILS`，仅第 1 个 Chu 实例，
使用上述 smoke checkpoint、ILS rounds=1、max_iters=1、drop_k=2、pool=2。
所有结果可行且极大，产物：`results/weak_decoder_t1_cli_smoke/20261102_140306_160768/`。

T0：同一实例及缩小预算，a/b/c/c_fullfixed 共 16 个阶段结果通过审计；产物：
`results/legacy_t0_cli_smoke/20261102_140327_903476/`。
原权重读取自 `D:\PhD_Doctor\Study\Codes\Constrained_AwareMKP\checkpoints\20261021_164411\best.pt`，
SHA 与旧源码版本保存在该 meta 中。不能从这一实例推断 delta 的总体贡献。
c_fullfixed 用 hook 覆盖旧控制器输出，计时仍包含被覆盖的网络计算；只能用于数值与同搜索预算对照，
不能把该耗时当作省去网络后的固定启发式成本。独立配对报告 CLI 也已跑通，产物位于
`results/weak_decoder_t1_cli_smoke/paired_report_cli/`。

## 后续正式实验命令

以下尚未执行；需按指导文档先完成 T0 全量拆解和确定训练预算。

```powershell
# T0：原始三个干预 + 完全固定采样补充，默认 180 实例和原搜索预算
python legacy_t0.py --checkpoint D:/PhD_Doctor/Study/Codes/Constrained_AwareMKP/checkpoints/20261021_164411/best.pt --arms a,b,c,c_fullfixed --device cuda:0

# T1：每个seed单独训练两种theta更新方式；以下只展示一个seed
python train_ConstrainedMKP.py --model.theta_update step --device.seed 1234 --train.epochs 40 --train.run_name dynamic_s1234
python train_ConstrainedMKP.py --model.theta_update static --device.seed 1234 --train.epochs 40 --train.run_name static_s1234

# 推理预算严格分开；搜索结果不能作为弱解码器greedy效果
python Benchmark_zeroShot.py --arm fixed,untrained,learned_static,learned_dynamic --dynamic_checkpoint checkpoints/dynamic_s1234/best.pt --static_checkpoint checkpoints/static_s1234/best.pt --stages greedy,sample64,sample256,greedy+ILS,sample256+ILS

# 汇总3个独立训练seed，保留逐实例配对单位
python mkp_evaluation.py --inputs RUN1/per_instance.csv RUN2/per_instance.csv RUN3/per_instance.csv --output results/paired_3seeds
```

推理主阶段 greedy 不受温度影响。默认 sampling 对照中 fixed/untrained 使用初始全局 T，
learned 使用训练后的全局 T，因此 sampling 增量包含全局温度学习的作用；meta 已记录这一口径。
若研究仅隔离 theta 的采样贡献，应在独立实验中使用共同固定温度。
每阶段总耗时计入一次 root LP；构造时间包含 theta trace 捕获，排除 JSON 序列化与解审计。

当前未验证：T0 全量贡献拆解、T1 三种子显著性与跨规模学习收益、单轨迹 greedy <=0.1%。
G2/G3 的 blocked_mass/残余 LP 正确性与成本检查属于后续阶段，当前没有冒充完成。


## T0 全量与 T1 seed1234 结果

2026-11-04 更新。分析与研究决策来自 Claude 的
[T0_T1_ANALYSIS_AND_T1R_GUIDANCE.md](T0_T1_ANALYSIS_AND_T1R_GUIDANCE.md)，本次不重新跑全量结果。
本文此前“未运行 T0/正式 T1”是 11-02 交付时的历史状态，现由后续运行更新：

- T0 全量：`results/legacy_t0/20261102_141830_091197/`，180 实例；a/b/c/c_fullfixed 的
  greedy 约为 0.437/0.450/0.583/0.583%，sample256+ILS 为 0.065/0.064/0.064/0.116%。
  具体统计与解释以 Claude 指导为准。
- T1：`results/20261103_130436_240755/`；fixed/untrained/static/dynamic 的 greedy 约为
  0.611/1.233/2.153/1.958%。dynamic 只训练至 ep10 it201，static 为 40 epoch，不能判断动态与静态优劣。
- `results/paired_3seeds/meta.json` 已补 `training_seeds:[1234]` 和 `single seed; name is legacy`，保留原路径。
- 本次没有从日志时序推断历史 dynamic 的停止原因，也无法据此确认历史 GPU 共享情况。

## T1-R

实施日期：2026-11-04。按用户分工，Codex 修改实现并做小规模验证；Claude 分析与指导；全量训练与评估由用户运行。

### 版本与改动

实施前已保存当前 T1 实现：分支 `codex/t1-baseline-before-t1r`，提交 **`4d5e686`**。
这是本次开始时源码的快照，不冒称为历史训练时已记录的 commit。
当前实施分支 `codex/t1r-sharpness-decoupling`，T1-R 改动尚未提交；实验 meta 会保存 commit/diff。

- 新配置默认 `score_normalization=true, temperature_mode=fixed`。原始分数在可行集上做 float32、
  总体标准差 z-score，不截断，std 梯度保留；零方差、单候选与 done 行处理保持有限。
- 旧 checkpoint 没有新配置字段时显式使用 raw scores，支持前后回归；新固定温度的默认数值只是占位，实验必须用校准 β0。
- 增加 alpha 上/下界比例、u 触界比例、原始可行分数 std；epoch −1 验证参与 best 选择。
  dynamic 在 epoch 0–3 全部比初始化差 >0.3 pp 时保存 early_stop.json 并停止。
- 评测 `--beta` 对全部臂统一逆温度，要求 normalized checkpoints 且 sample_temperature=1。
  标准化和温度配置纳入协议 SHA，旧 T1 与 T1-R 不能混配；记录所选 checkpoint 的 global_step。
- 新增合成 β 校准、常数 alpha 扫描、逐实例静态 theta 搜索（256 随机包含 fixed + 256 坐标搜索）、
  greedy 回归、配置准备/训练入口、曲线导出、性能 trace 脚本。搜索结果仅作可达余量下界诊断。
- REINFORCE、同实例 K16 基线、lr、entropy、theta 范围保持原设计；不加入 G2/G3 或数学上的工程加速。

### 已验证内容

原有 31 项测试及新增 6 项 T1-R 测试全部通过（先整套 36 项通过，随后新增配置准备保护测试通过）。
Python 编译与 `git diff --check` 通过。

小规模脚本产物在 `results/t1r_code_checks/`：

- `beta/`：N8/M2 一个合成实例，校准 CSV/meta 写入通过，标记 smoke_only，不能用于正式 β0。
- `headroom/`：同一实例，随机4+坐标4次评估，alpha 扫描、可行极大解审计与见证写入通过。
- `regression/`：仅 Chu 第1个实例，fixed/旧 dynamic/旧 static 的目标分别为 24003/23314/23789，
  标准化前后均一致；不替代 180 实例验收。
- 门控早停使用可控验证值做单元测试，梯度另用真实小实例验证；没有启动全量训练。
- 配置准备脚本默认只写配置，不训练；拒绝 smoke 校准和不完整或有差异的 greedy 回归。

### 用户运行顺序

从仓库根目录运行，python 指安装 PyTorch 的环境。已有结果目录会拒绝覆盖，可用 --output 改名。

```powershell
# A1：完整180实例，fixed及两个旧checkpoint的前后greedy回归
python scripts/t1r_regression.py --device cuda:0

# A3：仅128个固定合成验证实例、MILP参考，选择beta0
python scripts/calibrate_beta.py --device cuda:0

# B1/B2：主合成诊断，默认256+256；次要Chu诊断单独保存，不能参与选择
python scripts/theta_headroom.py --device cuda:0
python scripts/theta_headroom.py --dataset chu --device cuda:0 --output results/t1r_headroom_chu

# 只准备配置供检查；通过完整回归和校准后再由用户启动两臂训练
python scripts/run_t1r.py
python scripts/run_t1r.py --mode step --train
python scripts/run_t1r.py --mode static --train

# 从两份新checkpoint继承相同固定beta，主greedy和sample64，不跑ILS
python Benchmark_zeroShot.py --arm fixed,untrained,learned_static,learned_dynamic --dynamic_checkpoint checkpoints/t1r_dynamic_s1234/best.pt --static_checkpoint checkpoints/t1r_static_s1234/best.pt --stages greedy,sample64 --eval.sample_temperature 1

# 曲线导出需要matplotlib（缺少时在自己的环境安装）；性能脚本不更新权重
python scripts/plot_t1r.py
python scripts/profile_t1r.py --config results/t1r_configs/step.json --device cuda:0
```

`run_t1r.py` 从 `runs/dynamic_s1234/config.json` 继承设置，仅按指导改变归一化、固定温度、
10 epoch、theta_update、运行名、初始化验证和 dynamic 门控；不加载旧权重。
配置写入 `results/t1r_configs/`，校准与回归 meta 的 SHA 一并保存。
原运行已存在时拒绝覆盖；准备好的相同配置可以在后续 --train 时复用。
两臂计划预算一致；若 dynamic 早停，实际预算不同，回报时须明确，不能声称已作匹配预算的 Q-1 结论。

全量 greedy 回归中，任一臂差异超过2个实例会立即停止。
任何差异都会阻止自动训练前置检查，应把逐实例结果交给 Claude 检查浮点并列原因。

性能工具导出 Chrome trace 与算子表，标记 LP、controller forward、executor、environment、total backward。
controller backward 具体归属需要结合 autograd 事件检查；total backward 还包含 encoder/score。
本次没有实测或解释历史16–18倍耗时差、停止原因和历史GPU共享，也未实施需要逐步log-prob一致性证明的加速。

尚待用户执行：完整A1、正式β校准及选择、B1/B2正式预算、两臂门控训练、全量greedy/sample64、
曲线与性能剖析。效果假设未验证，后续研究决策仍由Claude按指导矩阵给出。


## T1-R 结果与后续 T1-P

2026-11-05：Claude 的结果分析与新任务见 [T1R_RESULTS_AND_T1P_GUIDANCE.md](T1R_RESULTS_AND_T1P_GUIDANCE.md)。
T1-R 已提交并推送为 `5b390e4`；后续代码交付与用户运行命令见 [WEAK_DECODER_T1P_EXECUTION.md](WEAK_DECODER_T1P_EXECUTION.md)。
