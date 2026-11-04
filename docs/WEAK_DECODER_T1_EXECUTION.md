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
