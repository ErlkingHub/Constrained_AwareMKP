# T1-P 实施与交付记录

日期：2026-11-05。研究分析与下一步裁决由 Claude 负责，依据
[T1R_RESULTS_AND_T1P_GUIDANCE.md](T1R_RESULTS_AND_T1P_GUIDANCE.md)。
本次只实施代码、必要小实例检查和指定的一次 G2 成本测量；完整训练与评估由用户执行。

## T1-R 提交和推送

已先将 T1-R 实现、工具和 Claude 分析指导提交为 **`5b390e447620a8471e435ef75adaf436189f0f40`**：

> feat: 修复 T1-R 锐度混淆并记录单种子结论

提交正文概括：可行集标准化、共享固定 beta、初始化验证和门控早停；合成校准 beta=8，
静态 theta 搜索揭示约 0.63 pp 可达余量；单种子 Chu 增益约 0.04 pp、不显著，dynamic 仅 250 步。
这些结果的解释以 Claude 指导为准，不能当作充分训练的动态与静态对照。

远程同名分支：`origin/codex/t1r-sharpness-decoupling`，已通过 HTTPS 成功推送并 fetch 核验。
用户通知 SSH 修复后再次尝试：22 端口仍被关闭；SSH 443 仍报 host key verification failed。
未绕过主机密钥校验，也未改变 origin URL；已成功的 HTTPS 推送不受该问题影响。
T1-P 在这个提交之后单独实现，保持两个阶段可追溯。

## 修改范围

- `mkp_config.py`：新运行默认 `signal=theta_pg`；旧配置缺该字段时恢复为 `item_pg`。
  新增 sigma_u=.05、sigma_alpha=.3、sigma_gamma=.3、antithetic=true；theta_interval 默认1。
- `mkp_Model.py`：控制器额外输出激活前 v/z（启用 G2 时另有 g）；纯函数 theta_from_pre 统一激活。
  噪声动作 detach，Gaussian logq 只对均值求梯度；首决策相邻正负对偶，后续决策独立。
  theta_noise 非空时执行器强制 greedy，返回各决策点 active logq 之和。
  static 先对 B 个实例读一次均值，再复制 K 份加噪；step 按 t=0,k,2k 刷新，中间冻结。
  trace 保留每步，theta_decision 标出决策点；噪声决策点记录 pre_mean/pre_sample。
- `mkp_Trainer.py`：theta_pg 使用同实例 K 均值 baseline 和原 adv_normalize，无熵项；
  额外 no_grad 均值 greedy 只作日志，记录 R_mean、frac_better、adv_zero_frac。
  static/dynamic 都支持原 epoch 0–3 门控；epoch_log.csv 汇总验证 gap（含 -1）、三项诊断均值及用时。
  旧 checkpoint 的模型默认值兼容；resume 校验训练信号与噪声契约，避免混用旧 best。
- 评测继承每个 checkpoint 的 k，拒绝 CLI 偷改。完整方法 SHA 保留 update/k，另有明确的比较 SHA
  用于预定 static/dynamic 调度对照；配对表同时保留双方完整 SHA、k 和 matched_update_schedule。
  报告包含整体、M/N/容量比分层、p90/worst、决策点 theta 统计。
- `scripts/run_t1p.py`：从 runs/t1r_static_s1234/config.json 继承，固定 beta=8、seed1234、
  10×250、B64/K16、原 lr/调度/数据。只生成配置时不训练，--train 才启动。
- T2 预写：`mkp_basis.py::blocked_mass`，no_grad 分块 O(N²M)，只统计当前可行其他物品的被阻断收益。
  groups 加 block 后，gamma=softplus(g)，初始化 .1，噪声在 g 上、sigma_gamma=.3；
  分数先减 gamma*blocked 再标准化。fixed 也取 gamma=.1。默认两份 T1-P 配置仍为 G0/G1，M+1；
  G2 只有显式启用时为 M+2。本次没有训练 G2。

## 必要检查与成本

原有 37 项回归全部通过；新增仅指导要求的三个 theta-PG 测试和一个 blocked_mass 测试，合计 **41 tests passed**。
没有新增其他实验性测试。编译与 git diff --check 通过。

| 指导检查 | 结果 |
|---|---|
| 无 theta_noise 的旧行为 | 复用原测试；另以不可变 5b390e4 源码与相同权重核对 k=1 的每步 theta、动作解、收益、item log-likelihood，完全一致 |
| Gaussian 得分函数梯度 | 小例 v/z：autograd 与 (draw-mean)/sigma²、固定 draw 的中心有限差分均一致 |
| 首决策对偶与 static B 调用 | 每对噪声互为相反数；hook 验证只调用一次且输入 batch 为 B；请求 sample 仍执行 argmax |
| theta_interval | k=3 时窗口内 theta 不变；k=1 与旧 step 完全一致 |
| blocked_mass | 唯一新增 block 测试与朴素双循环逐元素一致，含空可行集和已处理项 |

训练器只作三个小实例单步检查（CPU、B2/K4、N12/M3），item_pg、static theta_pg、dynamic k2 的 loss/gradient
均有限，三个新诊断正常；没有完整 fit 或新方法的全量 Chu 测试。

按指导完成一次 N500/B1 成本测量，产物 `results/t1p_code_checks/block_n500.json`：

- M10、容量比 .5、完整128维/encoder3/controller2、G2 block64、dynamic k8、bf16、seed1234。
- 实际设备 cuda:0，RTX 5090 Laptop GPU；当时只有一个可见 GPU。
- LP 0.411768秒；greedy 不含 LP 5.765856秒；合计6.177624秒。
- PyTorch 峰值 allocated **24.71 MiB**，reserved **38 MiB**；259步，解可行且极大。
- 无 warmup、无重复、无训练；包括 resident model/instance、LP与rollout，不是整机/驱动总显存。
- 当前日志会再次计算 raw_scores，因此含每步两次 blocked_mass；这是未优化实现的实测成本。
  本轮没有据此额外改数学、调预算或再做剖析。

## 用户运行入口

两份配置已准备好（未启动训练）：

- `results/t1p_configs/static.json` → `t1p_static_s1234`，theta_update=static，k=1。
- `results/t1p_configs/dynamic_k8.json` → `t1p_dynamic_k8_s1234`，theta_update=step，k=8。
- 同目录 meta.json 记录继承配置 SHA、source commit/diff；正式启动时会刷新 provenance。

```powershell
# 只准备配置，不训练
python scripts/run_t1p.py

# 单个可见GPU时顺序运行；若确有两张可见GPU，可在两个终端分别选 static/0 与 dynamic/1
python scripts/run_t1p.py --mode both --train --cuda_device 0

# 训练后仅跑三臂 greedy；不重跑 untrained、采样或 ILS
python Benchmark_zeroShot.py --arm fixed,learned_static,learned_dynamic --static_checkpoint checkpoints/t1p_static_s1234/best.pt --checkpoint checkpoints/t1p_dynamic_k8_s1234/best.pt --stages greedy
```

已有运行目录会被拒绝覆盖。若 dynamic 一个 epoch 超过20分钟，按指导回报剖析信息，不擅自缩减预算。
训练日志使用 runs/<name>/epoch_log.csv：包含 epoch -1、每 epoch 验证 gap、R_mean/frac_better/adv_zero_frac 均值与 time_s。
若早停导致实际预算不同，应如实报告；checkpoint 的最佳权重步数也保存在评测 meta 中。

完整训练、完整 greedy 评测、两臂对 fixed 的单种子门控结论尚待用户运行，不能用本次工程检查替代。
本轮不重做 beta 校准、余量诊断、多种子、绘图或全量性能剖析。是否进入 G2 训练由 Claude 按指导矩阵裁决。
