# T3-a / T3-b 实施记录

日期：2026-11-05。分支：`codex/t3-residual-lp-basis`。
用户指定的 `docs/T1R_RESULTS_AND_T3_GUIDANCE.md` 在工作树不存在；实际指导为
[Claude 的 T1P_RESULTS_AND_T3_GUIDANCE.md](T1P_RESULTS_AND_T3_GUIDANCE.md)，本次按该文件实施。
完整扫描、训练与评估由用户运行，研究解释与门控决定由 Claude 给出。

## 提交边界

先创建新分支，再按指导单独提交 D1 脚本、原始输出与文档：**`f04bcfb`**。
该提交包含 `scripts/t1p_lottery_check.py`、`results/t1p_lottery_check/output.txt` 及指导指针。
没有改动 Claude 的脚本或结果数字。T3 实现在其后的独立提交中保存。

## 实现

- `lp_dual` 扩展 mu_init/item_mask/scale，三参数均为 None 时保持旧 root 求解算式。
  warm multiplier 在新残余问题上重新计上界；mask 外物品不参加梯度/dual 项，整实例 p.mean 固定收益单位。
- 状态缓存 root 与 residual 两组 LP 基础量；t=0 复用 root mu，不重复求解。
  refresh_lp 在当前 fits 与 c_rem.clamp_min(1) 上求解，按新旧容量比例换算热启动价格。
  保留 st.mu 给原编码器；x_lp_root 固定，residual x_lp 每 lp_interval 更新，已处理/不可行项清零。
- 执行器支持独立 price_anchor=root/residual、lp_source=root/residual；只读状态、不在打分函数里求解。
  新 eta*log(x_lp+.01) 在标准化前加入，alpha_init/eta_init 和 gamma_init 支持固定臂及控制器初始化。
  eta 通过 softplus，并已接入 theta-PG 的 sigma_eta=.3、latent logq、日志与 trace。
- forward 对所有臂（含 fixed/override）在规定时刻先刷新 LP，再决策、打分。
  raw_scores 与 blocked_mass 每构造步只计算一次，原始 std 由 decoder 返回给日志复用。
- 新配置字段进入配置及评测协议 SHA；旧 checkpoint 缺字段使用兼容默认值，保留旧行为。
  常规评测补充 eta 统计、root/residual LP 分项计时；总时间包含二者。
- 源快照明确排除 runs/results/cache/checkpoints 等生成目录，避免在目录被取消忽略时递归嵌入历史 meta。

### 残余 LP 精度所需的配置调整

指导初值 residual_lp_iters=30 在必要测试中失败：最坏 HiGHS 差约20.05%、warm/cold差约19.94%。
按指导增加到 **300**（lr仍 .02），没有放宽精度阈值。新默认值和扫描配置都会明确保存300。
这会增加残余 LP 成本，正式扫描的逐实例分项计时用于评估该代价；不冒称30次热启动已达标。

## 必要验证

原41项回归与限定新增3项 T3 检查均通过，总计 **44 tests passed**。Python 编译和 diff 检查通过。

| 检查 | 实际结果 |
|---|---|
| 默认行为与 `4d3c376` 比较 | fixed 和两份实际 T1-P checkpoint，各取 M5/M10 的 N100 Chu 实例；解向量、收益、item log-likelihood 逐位一致；root mu/UB 默认求解也逐位一致 |
| 残余 LP 对 HiGHS | N30/M5，3种子、随机轨迹3/7/11步，共27状态；全部上界满足相对1e-6合法性；最大UB gap **0.413788%**；warm300/cold300最大差 **0.295446%** |
| 调用次数 | fixed与均值策略，lp_interval=1：t=0不重复LP，之后每步refresh一次；raw_scores和blocked_mass每步各一次；所有解可行且极大 |

有限小例的精度检查不构成所有规模/分布上的精度保证；正式扫描仍会报告成本和结果。
没有运行完整 Chu、T3-a 正式网格、交叉拟合实验或 T3-b 训练。

脚本数据流试跑使用 `--limit 2 --smoke-items 12`，产物位于 `results/t3_code_checks/scan_smoke/`：
16主臂、224条小实例结果、16条交叉拟合记录全部写出；所有解可行且极大，fit/heldout集合不相交。
这些是改为N12的工程fixture，不是真实Chu成绩或正式N100校准；meta/selection均标记smoke_only。
T3-b准备函数已实际拒绝该smoke选择，不会生成正式训练配置。

## 用户运行

```powershell
# T3-a：16个固定臂，synthetic N100选择，250/500外推和Chu只确认
python scripts/run_t3a_scan.py --device cuda:0

# 扫描完成后，只准备T3-b配置，不启动训练
python scripts/run_t3b.py

# Claude和用户确认门控通过后，用户才显式训练
python scripts/run_t3b.py --train --cuda_device 0

# 训练后仅做新fixed与learned_dynamic的greedy评估
python Benchmark_zeroShot.py --arm fixed,learned_dynamic --checkpoint checkpoints/t3b_dynamic_k8_s1234/best.pt --stages greedy
```

T3-a 默认输出 `results/t3a_scan/`。16主臂严格为 F0、A、4个R、4个S、AS、K1、3个G、BEST；
全局 Sη、K1基础臂、G3最优与gamma组合均只用合成N100选择。250/500仅报告相对F0、占root LP UB的目标差，
不把它写成MILP gap；Chu用best-known参考，既不选系数也不挑checkpoint。
F0会核对历史固定臂逐实例目标，缺参考文件时明确标记未核对，不能伪称复现。
若发现F0目标变化，会在写出完整选择文件之前报错，避免后续训练消费失败结果。
`progress.jsonl`逐实例落盘，`completed_configs/`保存已完成配置结果，终端中断仍可审查部分产物；暂不自动resume。

交叉拟合限定在合成N100：按M×容量比分层的固定两折，fold内选择、另一fold评估并交换。
eta的R/S族选择及alpha_BEST的配置选择链也放在fit半，避免泄漏。
alpha辅助变体和必要缓存测量单独标记，不混入16主臂。
该报告是给定分层和网格下的条件化常数余量估计，不能替代一般策略的全局可达性结论。

T3-b 从真实 `runs/t1p_dynamic_k8_s1234/config.json` 继承，并消费
`results/t3a_scan/selection.json` 中的全局系数、LP路径与实际求解器配置。
缺选择文件或仅有smoke结果时拒绝生成虚构配置。
生成路径为 `results/t3b_configs/t3b_dynamic_k8_s1234.json`；保留10×250、B64/K16、seed1234及原优化器/β。
16主臂均保持alpha=1；供T3-b使用的alpha_init另在合成N100的BEST辅助alpha∈{1,1.25}诊断上取全局最优，
eta/gamma与LP路径沿用BEST链。此辅助选择不改16主表结果，也不使用Chu。
按指导强制theta_interval=lp_interval=8；若扫描K1(interval1)获胜，这个训练间隔变化会在meta中注明。
不会按指标自动启动训练，是否训练仍由Claude和用户决定。

## 待用户运行并回报

完整16臂表、逐实例配对与分层、250/500外推、交叉拟合和N500显存/耗时仍待正式扫描。
本次只交付代码和必要工程验证，不以“实现完成”冒称效果已验证。
