# Constrained_AwareMKP · main_g1

面向 MKP 跨规模、跨数据分布泛化与长程信用分配的研究实现。依据
[研究交接纲领](docs/RESEARCH_HANDOFF_FOR_NEW_PROJECT.md)重新设计：学习器仅输出
`M` 个资源价格和 `4` 个评分基函数权重，固定执行器保证每一步可行。

## 直接运行

在本项目根目录打开 PowerShell，使用已有环境：

```powershell
& "D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe" .\train_ConstrainedMKP.py
& "D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe" .\Benchmark_zeroShot.py
```

第一条只用在线合成数据训练，以固定合成验证集选择 `best.pt`。第二条自动读取
`checkpoints/main_g1/latest.json` 指向的最佳模型，通过现有 `read_mat.py` 读取
`MKP_dataset/Benchmarks_Chu.mat` 的 **全部 270 个实例**，生成五个对照臂的单轨迹 greedy 结果。
Chu 不进入训练、验证或 checkpoint 选择。
若后续更新没有超过初始验证表现，`best.pt` 可以保留为 update=0；`last.pt` 始终记录最近保存的训练状态。

| 内容 | 输出位置 |
|---|---|
| 日志、配置、训练指标、实例种子/SHA、选择序列、源码快照 | `runs/<version>/<时间戳>_seed<seed>/` |
| 最佳模型、最近模型及 SHA | `checkpoints/<version>/<时间戳>_seed<seed>/{best,last}.pt` |
| 当前版本最近一次保存模型的指针 | `checkpoints/<version>/latest.json` |
| 逐实例指标、分层汇总、配对差异、解序列与 θ 轨迹 | `results/<时间戳>_<version>/` |

模型版本和全部可调参数集中在 [mkp_config.py](mkp_config.py)。默认：`N=100`，
`M∈{5,10}`，弱相关合成数据，容量比 `{0.25,0.5,0.75}`，2000 次 PPO 更新，每次 8 个独立实例。
`device=auto` 使用可用 CUDA，也可传 `--device cpu`。默认预算是实验起点，不是已验证的收敛预算。

## 配置、续训和实验

导出完整默认配置，然后编辑 JSON；也可仅提供需要覆盖的字段。未知字段会报错。

```powershell
& "D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe" .\train_ConstrainedMKP.py --write-default-config configs/my_experiment.json
& "D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe" .\train_ConstrainedMKP.py --config configs/my_experiment.json --seed 43 --version main_g1_seed43
```

续训恢复 actor、独立 critic、两个优化器、初始 actor、随机数状态和累计更新计数：

```powershell
& "D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe" .\train_ConstrainedMKP.py --resume checkpoints/main_g1/<run_id>/last.pt --updates 4000
```

`--updates` 是**累计目标**，不是追加量。保留同目录的 `best.pt`；续训会建立新的时间戳目录。
续训仅允许覆盖更新预算和设备。相同 CPU/runtime 的连续更新与续训已做逐张量一致性测试；
更换设备属于有效继续训练，但不保证逐位一致。使用自定义版本后评估时传同一配置，或显式指定 checkpoint：

```powershell
& "D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe" .\Benchmark_zeroShot.py --checkpoint checkpoints/main_g1/<run_id>/best.pt
& "D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe" .\Benchmark_zeroShot.py --checkpoint checkpoints/main_g1/<run_id>/best.pt --synthetic
& "D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe" .\Benchmark_zeroShot.py --checkpoint checkpoints/main_g1/<run_id>/best.pt --best-of-k 16
```

`--synthetic` 默认测试 `N={50,100,250,500}`、`M={5,10,30}`、四个数据族、三个容量比，
每格 5 个独立实例。测试生成参数与训练参数分开保存；实例按测试种子和实例 ID 生成，不随其他测试格的增删而改变。
`--best-of-k 16` 额外输出独立的 **16 条采样取最好**阶段，greedy 不混入采样结果。
`--n 100 250 --m 5 10` 可筛选规模，`--limit 2` 仅用于小范围检查。

对照臂：`dynamic`（每步重算 θ）、`static`（训练模型在根状态的 θ 固定到终局）、
`fixed`（均匀资源价格/密度规则）、`untrained`（同一次训练保存的初始 actor）、
`lp_dual`（根 LP 资源价格/密度规则）。所有臂共享执行器、候选可行性规则和轨迹预算。
未加入局部搜索或修复。

`instances.csv` 的 `best_known_gap_pct = 100*(best_known-objective)/best_known`，保留符号；
`lp_gap_pct` 单独使用 LP 上界。`summary.csv` 按 N、M、容量比、数据族和组合分层，含均值、
中位数、P95、最差 gap、时间与可行性；`paired.csv` 的 gap 差值单位是**百分点**。
合成测试没有 best-known 标签，该列留空。利用率高不是高解质量或“约束理解”的独立证据。

## 检查与设计说明

```powershell
& "D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe" -m unittest discover -s tests -v
& "D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe" .\train_ConstrainedMKP.py --config configs/smoke.json
& "D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe" .\Benchmark_zeroShot.py --config configs/smoke.json --device cpu --limit 1
```

`smoke.json` 使用单独的 `main_g1_smoke` 版本，不会替换默认正式版本的模型指针。
这些检查验证接口、梯度、可行性、序列化和指标，不证明 gap 改善或泛化能力。
算法公式、表示需求对照、信用分配消融和证据边界见 [实现与实验说明](docs/MAIN_G1_IMPLEMENTATION.md)。
