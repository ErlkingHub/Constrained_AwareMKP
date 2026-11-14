# T9 训练执行加速与服务器命令

2026-11-14。此补充取代上轮“暂不做速度优化”的运行建议；训练框架和研究门控仍按 [T9执行说明](WEAK_DECODER_T9_EXECUTION.md)。没有启动正式训练，未改变单种子、模型、奖励、K16、优势归一化、熵系数、梯度裁剪、优化器、2500更新或合成验证选best规则。完整评估仍为Chu270。

## 实现与等价性

新增可选 `--lp_backend cuda_graph`，默认 `eager` 保留参考路径。改动集中在 LP 的执行调度：仍冷启动 y=0、原8个温度×5次Newton、原回溯候选、float64物理/LP计算与原返回精度。没有 warm start、少算 LP、减少迭代、AMP、TF32 或改变 batch/K 的优化。

`mkp_lp_runtime.py` 按设备、形状、dtype、stride和LP配置缓存CUDA Graph，最多16个。捕获使用相同的 `solve_ex(check_errors=False)` 运算，并在每次LP返回前检查全部40次求解的info；未取消数值错误检查。原 `solve` 每次会同步CPU，[PyTorch solve_ex说明](https://docs.pytorch.org/docs/stable/generated/torch.linalg.solve_ex.html)允许把检查移至LP末尾。图内运算顺序保持原式。

输入和回溯常量持续持有；返回的12个LP字段都复制到独立存储，避免下一次重放覆盖根节点dual/bound或反向传播所需的值。这遵循 [PyTorch CUDA Graph内存约定](https://docs.pytorch.org/docs/main/notes/cuda.html#cuda-graphs)。捕获不支持时记录原因并退回eager；CPU和非常规非稠密view使用eager。未引入Triton/编译器安装依赖。

新增 `tests/test_lp_runtime.py` 验证：

- CPU回退完全相同；CUDA上M5/M10/M30、不同B/N的所有LP字段逐位一致。
- 再次重放和修改物理状态后，旧输出及根上下文保持不变。
- 非零读出头、K16、跨微批次的完整更新，采样序列和CUDA RNG状态一致；loss一致至1e-6，梯度rtol=1e-5/atol=1e-6、更新后参数rtol=1e-6/atol=1e-7。

本机完整回归共69项通过（含上述3项CPU/CUDA测试）。另完成加速入口的Chu3（M5/M10/M30）+合成2、D_cert/K*工程检查：10条轨迹、466次重放，无回退；smoke配置尝试训练仍在门控处按预期拒绝。这些是实现检查，不是收敛或基准收益实验。

`LP_runtime` 写入训练和计时meta，记录captures/replays/fallbacks与实际预热求解次数。`LP_SOLVE_*`仍为逻辑求解计数；每张图的额外3次预热单独列入warmup_batches/warmup_states，录制本身不算训练轨迹。不将这些执行准备当作采样或训练预算。

## 本机计时结果

RTX 5090 Laptop GPU，PyTorch 2.11.0+cu128，原模型宽度/深度、相同生成种子与初始θ。每个形状/批量各计时一次独立更新，包含首用图准备。正式校准值与长期训练尚未测。LP小测试约49.4→6.4ms；该约7.7倍**不能**作为端到端训练倍数。

| M | N | 原B64 秒 | 新B64 秒 | 原B32 秒 | 新B32 秒 |
|---:|---:|---:|---:|---:|---:|
|5|50|42.94|17.90|24.58|10.40|
|5|100|96.56|42.04|44.30|19.57|
|5|150|149.49|71.82|67.60|36.62|
|10|50|46.34|20.96|18.53|10.53|
|10|100|98.59|49.12|38.16|23.46|
|10|150|143.56|79.61|66.40|42.05|
|等权均值||96.25|**46.91**|43.26|**23.77**|

B64/B32各自的六形状均值约快2.05/1.82倍。另做同日顺序复测（M5/N100/B32、相同种子、各1次完整更新），eager 34.49秒、cuda_graph 20.01秒，约1.72倍；原始记录在 `results/t9_profile_eager_recheck/` 与 `results/t9_profile_graph_recheck/`。单次计时受GPU频率等影响，不能把历史表当成稳定的精确倍数。

按原“第250步预计总时长>24h才B64→32”的规则，本机两种后端均会触发原降配：

`(250×46.9071 + 2250×23.7727) / 3600 = 18.12小时`。

相同计划下原值33.72小时；新增后端的主计算外推约快1.86倍。加上25%工程余量，暂估**18–23小时**，不含前置字典门控/校准和最终评估。这不是完整训练实测或时间保证。正式运行仍以首250步实测决定降配；4090服务器更快时可能按原规则继续B64，实际轨迹预算会写入meta。没有为了展示更快而强制提前降批量。

原/新完整形状数据分别在 `results/t9_profile/` 与 `results/t9_profile_cuda_graph/`。不据此宣称训练质量、gap、收敛速度改善。

## 本机命令

保持已有正式门控，不使用smoke结果批准训练。使用新的配置输出目录，避免覆盖原配置：

```powershell
$py = 'D:\Zlb08\.envs\RL_MKPtest\Scripts\python.exe'
& $py -B -m unittest discover -s tests -p test_lp_runtime.py -v
& $py scripts/profile_t9.py --lp_backend cuda_graph --device cuda:0 --calibration results/t9_dictionary/calibration.json --output results/t9_profile_fast_calibrated

# 仅在完整T9-0/T9-2门控通过且已选定基础cert后启动
& $py scripts/run_t9.py --train --decision cert --lp_backend cuda_graph --cuda_device 0 --dictionary results/t9_dictionary --measure results/t9_measure --output results/t9_configs_fast

# 评估也可显式选择该后端；每个新臂仍完整Chu270
& $py scripts/eval_t9.py --lp_backend cuda_graph --dictionary results/t9_dictionary --device cuda:0 --output results/t9_best_270_fast
```

去掉 `--lp_backend cuda_graph` 或改为 `--lp_backend eager` 即恢复参考实现。版本化checkpoint的模型参数格式不变；评估meta记录实际选择的后端，不因checkpoint保存了训练执行选项而混淆。

## 双4090服务器可选运行

下面是Linux命令，假设仓库、CUDA版PyTorch及依赖已经准备好。把被git忽略的正式门控/校准结果、MAT数据、T5交换文件、T8基线、必要checkpoint与MILP缓存一同复制到相同的仓库相对路径，不能只clone代码。保留文件字节以通过SHA校验。SciPy版本应与参考缓存保持一致（本项目缓存为1.15.3），避免重新生成一套验证incumbent。

本机只测试了一张5090 Laptop，没有实测4090性能。先在服务器检查CUDA等价性和完整更新计时：

```bash
CUDA_VISIBLE_DEVICES=0 python -B -m unittest discover -s tests -p test_lp_runtime.py -v
CUDA_VISIBLE_DEVICES=0 python scripts/profile_t9.py --lp_backend cuda_graph --device cuda:0 --calibration results/t9_dictionary/calibration.json --output results/t9_profile_4090

# GPU0：一个cert模型，保持原全局batch/K及梯度逻辑
CUDA_VISIBLE_DEVICES=0 python scripts/run_t9.py --train --decision cert --lp_backend cuda_graph --cuda_device 0 --dictionary results/t9_dictionary --measure results/t9_measure --output results/t9_configs_4090
```

第二张卡可以在另一个终端同时处理独立的T9-4诊断：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_t9_measure.py --tasks prefix --device cuda:0 --output results/t9_prefix_4090
```

这里每个进程只看到一张卡，因此都使用进程内cuda:0。**当前入口是单模型单卡；上述两卡用法是独立任务并行，不是同一模型的DDP训练。** 未提供未经验证的torchrun双卡训练命令，也不将第二张卡算作已测得的训练加速。若需要单模型双卡分担K组采样，应另做分布式梯度/随机数/全局batch验证。
