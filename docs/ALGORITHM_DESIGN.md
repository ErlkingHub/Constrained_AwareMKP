# 算法设计与实现契约

本文解释当前代码中的状态、策略参数 θ、固定评分公式、LP 对偶、训练算法以及训练/推理边界，可作为后续开发和 AI 智能体的实现索引。研究定位和已有实验结果见 [RESEARCH_OVERVIEW.md](RESEARCH_OVERVIEW.md)。

核对日期：2026-10-23；分支 `main_v0`；提交 `e7d1def91236ac8bd0e73153a1f4504af695c6bd`。以下是当前实现，不把未来设计当作现有功能。源码行号对应本次快照，后续以符号名定位为准。

**首先澄清 PPO：当前代码没有实现 PPO，实际训练为 REINFORCE + 同实例多采样均值基线。** 第 8 节给出实际 loss，并单列 PPO 尚缺的接口和组件。依据：[mkp_Trainer.py:4](../mkp_Trainer.py#L4)、[train_step](../mkp_Trainer.py#L157)。

## 1. 符号、数据结构与调用关系

为避免代码中两个 K 混淆，本文用 `K` 表示每实例 rollout 数，用 `P` 表示评分基元数；默认分别为 16 和 4。`B` 为实例数，`N` 为物品数，`M` 为资源维数，`D` 为 embedding 维数，`H` 为 attention head 数；默认 `D=128,H=8`。进入同一次环境或模型调用的实例具有相同 N、M。

| 符号/字段 | shape / 类型 | 含义与来源 |
|---|---|---|
| `inst.p` | `(B,N)`，float32 | 原始收益 `p_j`。 |
| `inst.r` | `(B,M,N)`，float32 | 原始消耗 `r_ij`；资源轴在物品轴之前。 |
| `inst.c` | `(B,M)`，float32 | 原始容量 `c_i`。 |
| `inst.best` | 可选 `(B,)` | 基准文件中的 best-known；在线生成实例为 `None`。 |
| `mu, ub` | `(B,M), (B,)` | 静态归一化 LP 对偶价格、原始收益单位的上界。 |
| `x_s, y_s` | `(B,N,D), (B,M,D)` | 静态物品与资源表示。 |
| `num_samples=K` 后的批维 | `B*K` | `repeat_interleave`：同一实例的 K 条轨迹相邻。 |

依据：[MKPInstances / repeat](../mkp_env.py#L48)、[feature sizes](../mkp_env.py#L357)、[ModelConfig](../mkp_config.py#L81)、[policy.forward](../mkp_Model.py#L336)。后文省略 batch 下标；涉及 rollout 时，环境中的批维已经变为 `B*K`。

```text
train.main -> MKPTrainer.fit -> train_step
  -> env.generate -> env.precompute -> policy.forward(sample, K)
  -> terminal reward / trajectory log-likelihood / mean entropy
  -> shared baseline -> loss.backward -> Adam.step

benchmark.main -> load_policy -> read_mat -> evaluate_instance
  -> env.precompute -> policy.forward(greedy, 1)
                    -> policy.forward(sample, chunk K) [可多次]
  -> policy_obj -> [可选 iterated_local_search] -> final_obj -> CSV
```

调用入口：[train_ConstrainedMKP.main](../train_ConstrainedMKP.py#L18)、[MKPTrainer.train_step](../mkp_Trainer.py#L157)、[benchmark.main](../Benchmark_zeroShot.py#L125)。

## 2. 优化问题、环境状态与转移

### 2.1 原始问题与训练实例

代码求解：

\[
\max_{x\in\{0,1\}^N}\sum_jp_jx_j,
\qquad \sum_jr_{ij}x_j\le c_i\quad(i=1,\ldots,M).
\]

默认在线生成：

\[
r_{ij}\sim U\{0,\ldots,1000\},\quad q_j\sim U[0,1),\quad
p_j=\max\left(1,\left\lfloor\frac1M\sum_ir_{ij}+500q_j\right\rfloor\right),
\]
\[
\alpha\sim U\{0.25,0.5,0.75\},\qquad
c_i=\left\lceil\alpha\sum_jr_{ij}\right\rceil.
\]

同一实例所有资源共享本次抽到的 α；每个实例独立抽取。权重上限、收益噪声与 α 列表可配置。生成在 CPU 上进行，随后搬至目标设备。依据：[generate_instances](../mkp_env.py#L77)、[MKPEnv.generate](../mkp_env.py#L196)。

### 2.2 `MKPState` 字段

| 字段 | shape / 类型 | 不变量或语义 |
|---|---|---|
| `inst`, `mu`, `ub` | 见第 1 节 | 当前实例及静态 LP 结果，episode 内不重新求 LP。 |
| `c_rem` | `(B,M)` | 剩余容量，只在接受动作时减少。 |
| `status` | `(B,N)`，int8 | `0=AVAILABLE,1=SELECTED,2=REJECTED`。 |
| `profit` | `(B,)` | 已接受物品的原始总收益。 |
| `n_reject` | `(B,)` | 活跃轨迹中的拒绝尝试数。 |
| `fits` | `(B,N)`，bool | 未处理且同时满足全部资源约束。 |
| `done` | `(B,)`，bool | `~fits.any(dim=1)`。 |
| `t` | Python int | 整批已经调用 `step` 的次数，不是每条轨迹的独立长度。 |
| `available` | 属性，`(B,N)` | `status == AVAILABLE`。 |
| `solution` | 属性，float32 `(B,N)` | `1[status == SELECTED]`。 |

依据：[MKPState](../mkp_env.py#L163)、[reset / _refresh](../mkp_env.py#L212)。

### 2.3 动作、接受/拒绝、终止与回报

动作是每行一个物品下标 `action[B]`。以 `ε=1e-6` 表示环境容差：

\[
fits_j=available_j\land\bigwedge_i(r_{ij}\le c^{rem}_i+\varepsilon),
\qquad done=\neg\bigvee_j fits_j.
\]

`reset()` 令全部物品未处理、`c_rem=c`、收益和拒绝数为 0，再计算 fits/done。`step()` 对未结束行执行：

1. 当前物品未处理且满足所有容量约束：标为 SELECTED，`c_rem -= r[:,j]`，`profit += p[j]`。
2. 当前物品未处理但不满足容量：标为 REJECTED，容量和收益不变，拒绝数加 1。
3. 已结束行保持不变；调用结束后全批 `t += 1`，刷新 fits/done。

正常策略用结构 mask 避免重复选择。若外部直接调用 `step()` 传入已处理物品，活跃行仍会计一次拒绝，原 status 不变。没有回退、换出、修复或 STOP 动作。没有可放物品即结束，因此最终仍可能有 AVAILABLE 但不可放的物品。依据：[MKPEnv.step](../mkp_env.py#L230)。

训练用终局回报：

\[
R=\frac{profit}{\max(UB,\varepsilon)}
   -\rho\frac{n_{reject}}N,
\qquad \rho=0.1\ \text{（默认 Config）}.
\]

不返回逐步 reward，也没有折扣因子。`MKPEnv()` 不传配置时内部 fallback 的 ρ 是 0.05；正常入口传入 `cfg.env`，采用默认配置中的 0.1。依据：[MKPEnv.__init__](../mkp_env.py#L189)、[get_reward](../mkp_env.py#L250)、[EnvConfig](../mkp_config.py#L72)。

最终解可行性来自环境只接受可放物品；默认模式允许策略尝试不可放物品。容量单调减少，当前不可放的物品以后也不可放，故丢弃该物品不损失其未来可行机会；这一性质不保证此前选择的集合优质或最优。

## 3. LP 松弛对偶：价格、上界与计算边界

### 3.1 实际求解目标

令 `s=mean_j p_j`、`v_j=p_j/s`、`a_ij=r_ij/max(c_i,ε)`。正容量下，归一化 LP 松弛为：

\[
\max_{0\le x_j\le1}\sum_jv_jx_j,
\qquad \sum_ja_{ij}x_j\le1.
\]

`lp_dual()` 不调用通用 LP 求解器，而是近似最小化消去物品上界乘子后的对偶目标：

\[
D(\mu)=\sum_i\mu_i+
\sum_j\left[v_j-\sum_i\mu_ia_{ij}\right]_+,
\qquad \mu_i\ge0,
\quad UB=sD(\mu).
\]

任意非负 μ 都在数学上给出上界：对任意 LP 可行 x，`vᵀx ≤ μᵀ1 + Σ_j[v_j-a_jᵀμ]_+`。有限迭代未求到对偶最优会使上界更松；返回值不是已经证明最优的 LP 值，更不是整数最优值。代码使用 float32，并未提供严格数值认证。依据：[lp_dual](../mkp_env.py#L109)。

μ 的单位是“归一化收益 / 容量比例”；若要换成原始资源单位价格，正容量下对应 `s*μ_i/c_i`，不能把 μ 本身解释成每一原始重量单位的价格。

### 3.2 平滑投影 Adam 的精确结构

默认 `iters=300,lr=0.05`。初始化：

\[
\mu_i=\max\left(\frac{\overline v}{M\max(\operatorname{mean}_ja_{ij},\varepsilon)},\varepsilon\right).
\]

每轮 `t=1,...,iters`，`f=t/iters`：

\[
\tau_t=0.05(1-f)+0.001f,\quad
g_i=1-\sum_ja_{ij}\,\sigma\left(\frac{v_j-a_j^T\mu}{\tau_t}\right).
\]

对 g 使用 Adam 一二阶矩（`β1=0.9,β2=0.999`、偏差校正、分母 `sqrt(m2_hat)+1e-8`），学习率为
`lr * (0.5*(1+cos(πf))+0.01)`，再乘本行 `max(mean_i μ_i,1e-3)` 作为价格尺度，更新后投影至 `μ>=0`。每次更新后按**未平滑的 D(μ)** 比较，逐实例保存历史最好值及对应 μ。返回 `best_mu` 和 `best_val*s`。依据：[mkp_env.py:125](../mkp_env.py#L125)。

### 3.3 LP 在系统中的四个用途

| 用途 | 调用点 | 是否进入梯度 |
|---|---|---|
| 静态物品/资源/边特征 | `static_features` 的 dual cost、reduced cost、价格 share | 否 |
| 动态价格 λ 的基准 | `WeakDecoder.__call__` 中的 `base(mu)` | μ 本身无梯度；后续策略调整有梯度 |
| 回报与验证尺度 | `profit/UB` 和 gap-to-LP | 否 |
| 启发式参考与局部搜索排序 | `greedy_dual_ratio`、`mkp_refine._ratio` | 否 |

依据：[precompute](../mkp_env.py#L206)、[static_features](../mkp_env.py#L269)、[decoder](../mkp_Model.py#L298)、[refine._ratio](../mkp_refine.py#L27)。LP 每实例预计算一次，可经 `pre=(mu,ub)` 复用；决策过程中不对剩余背包重求 LP。没有 LP 原始解、最优动作标签、MILP 认证或 Oracle 搜索。迭代次数必须为正；当前 `iters=0` 会留下 `inf` 上界，配置层未显式校验。

README 关于“接近 scipy LP 最优值约 0.05%”的说法，在当前工作区没有配套精度验证脚本或结果，不能当成这里已验证的误差保证。

## 4. 状态特征及编码器的完整输入契约

### 4.1 共同记号与维度

以下表格的编号均从 0 开始，**表中顺序就是 `torch.stack(...,dim=-1)` 的最后一维顺序**。

静态辅助量：

\[
a_{ij}=r_{ij}/\max(c_i,\varepsilon),\quad
b_{ij}=r_{ij}/\max(\operatorname{mean}_jr_{ij},\varepsilon),\quad
v_j=p_j/\operatorname{mean}_jp_j,
\]
\[
d_j=\sum_i\mu_ia_{ij},\quad
q_j=\log\max(v_j,\varepsilon)-\log\max(d_j,\varepsilon),\quad
e_j=\log\max(v_j,\varepsilon)-\log\max(\sum_ia_{ij},\varepsilon),
\]
\[
h_j=(v_j-d_j)/(|v_j|+\varepsilon),\qquad
u_i=\mu_i/(\sum_i\mu_i+\varepsilon).
\]

本节 `Z` 是环境 `_zscore()`：在指定集合上用总体标准差（`unbiased=False`），计算 `(x-mean)/(std+1e-6)` 后截断到 `[-5,5]`；不传 mask 时集合是全部物品。动态 `Z_A` 只在 AVAILABLE 物品上统计。这个 Z 与第 6 节 decoder 的方差加 epsilon 版本稍有不同。依据：[MKPEnv._zscore](../mkp_env.py#L257)。

| 特征组 | shape | 常量 |
|---|---|---|
| 静态 item/resource/edge | `(B,N,11)` / `(B,M,6)` / `(B,M,N,2)` | `F_ITEM_STATIC=11,F_RES_STATIC=6,F_EDGE_STATIC=2` |
| 动态 item/resource/edge | `(B,N,8)` / `(B,M,5)` / `(B,M,N,2)` | `F_ITEM_DYN=8,F_RES_DYN=5,F_EDGE_DYN=2` |
| global | `(B,6)` | `F_GLOBAL=6` |

依据：[feature sizes](../mkp_env.py#L357)。所有固定特征提取均 `@torch.no_grad()`；这些是输入量，不是可训练的特征生成器。

### 4.2 静态特征

| item 索引 | 实际特征 | 含义 |
|---:|---|---|
| 0 | `p_j / max_j p_j` | 相对最高收益。 |
| 1 | `Z(v_j)` | 标准化收益。 |
| 2 | `mean_i b_ij` | 相对资源消耗均值。 |
| 3 | `max_i b_ij` | 最大相对消耗。 |
| 4 | `min_i b_ij` | 最小相对消耗。 |
| 5 | `std_i(b_ij, unbiased=False)` | 跨资源消耗差异。 |
| 6 | `Z(q_j)` | 静态 LP 价格比值。 |
| 7 | `Z(e_j)` | 无价格的容量效率。 |
| 8 | `clip(h_j,-3,1)` | 相对 reduced cost。 |
| 9 | `sigmoid(10*h_j)` | 平滑的 reduced-cost 指示。 |
| 10 | `argsort(argsort(q_j))/max(N-1,1)` | q 的升序相对排名；数值越大排名越靠前。 |

| resource 索引 | 实际特征 | 含义 |
|---:|---|---|
| 0 | `c_i / max(sum_j r_ij,ε)` | 各资源紧度。 |
| 1 | `u_i` | 对偶价格占比。 |
| 2 | `M*u_i` | 按 M 缩放的价格占比。 |
| 3 | `std_j(b_ij, unbiased=False)` | 本资源跨物品消耗差异。 |
| 4 | `sum_j[(r_ij-mean_j r_ij)*(p_j-mean_j p_j)] / (norm(r_i-mean(r_i))*norm(p-mean(p))+ε)` | 该资源消耗与收益的相关系数。 |
| 5 | `log(M)/log(30)` | 资源数信息；30 是归一化常数，不是维数硬限制。 |

| edge 索引 | 实际特征 | 含义 |
|---:|---|---|
| 0 | `clip(b_ij,0,5)/2` | 相对资源消耗；最大值是 2.5。 |
| 1 | `clip(mu_i*a_ij/(v_j+ε),0,5)` | 本资源价格成本占物品归一化收益的比例。 |

三张表的直接实现：[MKPEnv.static_features](../mkp_env.py#L269)。参数 `ub` 虽传入该函数，当前静态特征没有使用它。

### 4.3 动态特征

定义 `A_j=1[AVAILABLE]`、`C_i=max(c_rem_i,0)`：

\[
L_{ij}=r_{ij}/(C_i+1),\quad O_{ij}=1[r_{ij}>C_i+\varepsilon],\quad
F_i=C_i/\max(c_i,\varepsilon),\quad n_A=\max(\sum_jA_j,1),
\]
\[
\bar r_i^A=\sum_jr_{ij}A_j/n_A,\quad
D_i^A=\sum_jr_{ij}A_j/(C_i+1),\quad
f_i^A=\sum_j(1-O_{ij})A_j/n_A,
\]
\[
q_j^{dyn}=\log\max(p_j,\varepsilon)-\log\max(\sum_iL_{ij},\varepsilon).
\]

| item 索引 | 实际特征 |
|---:|---|
| 0 | `A_j`。 |
| 1 | `1[status_j == SELECTED]`。 |
| 2 | `1[status_j == REJECTED]`。 |
| 3 | `1[sum_i O_ij == 0] * A_j`，即当前可放且未处理。 |
| 4 | `clip(max_i L_ij,0,3)`。 |
| 5 | `clip(min_i[(C_i-r_ij)/max(c_i,ε)],-1,1)`。 |
| 6 | `mean_i O_ij`，溢出的资源维度比例。 |
| 7 | `Z_A(q_j_dyn) * A_j`。 |

| resource 索引 | 实际特征 |
|---:|---|
| 0 | `F_i`，剩余容量比例。 |
| 1 | `log1p(C_i/(mean_r_i_A+1))/log(N+1)`，剩余容量相对平均候选消耗。 |
| 2 | `log1p(D_i_A)/log(N+1)`，剩余候选的总需求压力。 |
| 3 | `f_i_A`，仅就本资源维度而言可放的候选比例。 |
| 4 | `F_i-mean_i F_i`，相对资源平均剩余比例。 |

| edge 索引 | 实际特征 |
|---:|---|
| 0 | `clip(L_ij,0,3)/3`。 |
| 1 | `O_ij`，本资源上的溢出指示。 |

| global 索引 | 实际特征 |
|---:|---|
| 0 | `1-n_A/N`，已处理进度。 |
| 1 | `selected_count/N`。 |
| 2 | `profit/max(ub,ε)`。 |
| 3 | `mean_i F_i`。 |
| 4 | `min_i F_i`。 |
| 5 | `sum_j st.fits_j/n_A`，同时满足全部资源约束的候选比例。 |

四张表的直接实现：[MKPEnv.dynamic_features](../mkp_env.py#L307)。`n_A` 被截到至少 1，所以完全没有 AVAILABLE 时 global[0] 实际为 `1-1/N`；这个状态行可能在同批其他轨迹未结束时继续作为占位行参与前向。

这些输入以比值和归一化为主，但有原始单位上的 `+1` 等稳定项，不保证任意缩放原始收益/资源单位后严格不变。`perceive` 的输入直接包含 fits/overflow，不能描述成“只由原始 p/r/c 隐式猜测可行性”。

### 4.4 双部编码与每步动态融合

`StateEncoder.encode_static()`：item/resource 分别经 `Linear→GELU→Linear` 投影到 D，运行默认 3 个 `DualAxisBlock`，最后 LayerNorm。静态 edge 输入为 2 维。输出 `x_s[B,N,D]`、`y_s[B,M,D]`。同一次 forward 中先完成静态编码，再复制到 K 条轨迹，不为每条轨迹重复做静态编码。依据：[StateEncoder](../mkp_Model.py#L165)、[policy.forward](../mkp_Model.py#L347)。

每一步重新计算：

```text
g = MLP(global)                             # (BK,D)
x = x_s + MLP(item_dynamic) + g[:,None,:]  # (BK,N,D)
y = y_s + MLP(res_dynamic)  + g[:,None,:]  # (BK,M,D)
edge = concat(edge_static, edge_dynamic)   # (BK,M,N,4)
```

控制器使用默认 2 个 `DualAxisBlock`。每层为 pre-LN + residual，按顺序进行：item self-attention → item 读取 resource → resource 读取更新后的 item → resource self-attention → 两类 token 各自 FFN。edge MLP `F_edge→32→2H` 产生双向 cross-attention 的逐头加性 bias；attention 使用 PyTorch `scaled_dot_product_attention`。依据：[EdgeBias](../mkp_Model.py#L105)、[DualAxisBlock](../mkp_Model.py#L121)、[_step_core](../mkp_Model.py#L330)。

动态控制器把已处理 item 屏蔽为 item self-attention 和 resource←item attention 的 key；没有删除这些 item query。无 AVAILABLE 的行临时保留所有 key，避免全 mask。没有上一时刻控制器 hidden state 的递归传递。自注意力仍含 N² 项，支持可变 N 不等于计算量与 N 无关。依据：[DualAxisBlock.forward](../mkp_Model.py#L144)、[StrategyController.forward](../mkp_Model.py#L241)。

## 5. 高层策略 θ 的含义与初始化

### 5.1 上下文聚合与策略头

控制器将更新后的 item/resource token 分别 LayerNorm，再拼接：

```text
ctx_input = [mean_available(x), max_available(x), mean_resources(y), g]  # 4D
ctx = MLP(ctx_input)                                                     # D
```

item 均值只统计 AVAILABLE，max 对已处理 token 填 `-1e4` 后取最大值；没有 AVAILABLE 的行使用占位逻辑。四种头输出：

| 输出键 | shape | 实际变换 | 默认范围/含义 |
|---|---|---|---|
| `log_price` | `(BK,M)` | `L*tanh(price_head([y_i,ctx]))` | `L=3`，每资源价格的 log 调整。 |
| `prim_w` | `(BK,P)` | `softmax(prim_head(ctx))` | 非负、和为 1，基元顺序按配置。 |
| `kappa` | `(BK,)` | `kappa_min + softplus(kappa_head(ctx))` | 下界 0.1，代码没有上界。 |
| `residual` | `(BK,N)` | `residual_max*tanh(res_head([x_j,ctx]))` | 默认幅度 ≤3，逐物品学习残差。 |

本文定义 `θ_t=(log_price_t,w_t,κ_t,δ_t)`；实际正价格 λ 在 decoder 中生成，不是头部直接输出。θ 的长度随 M、N 变化，含 N 维逐物品残差，不能称为只输出少量全局标量。依据：[StrategyController.__init__ / forward](../mkp_Model.py#L202)。

### 5.2 实际价格与初始化

\[
base_i=\mu_i+price\_floor\cdot\operatorname{mean}_i\mu_i+10^{-6},
\qquad \lambda_{ti}=base_i\exp(log\_price_{ti}).
\]

默认 `price_floor=0.05`。它是加到每个资源价格的公共均值项，不是 `max(mu_i,floor)`。λ 严格为正；范围约为 `base*exp(±3)`，动态调整的是相对静态 μ 的倍率。依据：[WeakDecoder.__call__](../mkp_Model.py#L298)。

`_init_heads()` 的实际初值：

- 价格头和残差头的最后线性层权重、bias 为 0，所以 `log_price=0,δ=0`。
- primitive 头权重为 0；默认 `price_ratio` 的 bias=2，其他 bias=0。
- κ 头权重为 0，bias 设为 `softplus_inverse(kappa_init-kappa_min)`，默认 `κ=3`。

默认 P=4 时，`w_price_ratio=exp(2)/(exp(2)+3)≈0.711235`，其余三个权重各约 0.096255。这是偏向价格比值的**混合启发式初值**；并非只使用 price_ratio，也不与独立的静态 dual-greedy 完全相同。价格 floor、动态剩余容量分母、基元混合与随机采样都会区分两者。依据：[_init_heads](../mkp_Model.py#L225)、[greedy_dual_ratio](../mkp_env.py#L363)。

## 6. 固定弱评分器：基元、标准化与 mask

### 6.1 四种可配置基元

记 `C_i=max(c_rem_i,0)`。默认顺序为 `profit,efficiency,price_ratio,slack`；配置可选择子集和顺序，控制器检查名称是否合法。

\[
\phi_{profit}(j)=\log\max(p_j,10^{-6}),
\]
\[
\phi_{efficiency}(j)=\log\max(p_j,10^{-6})
-\log\max\left(\sum_i\frac{r_{ij}}{c_i},10^{-6}\right),
\]
\[
\phi_{price\_ratio}(j)=\log\max(p_j,10^{-6})
-\log\max\left(\sum_i\lambda_{ti}\frac{r_{ij}}{C_i+1},10^{-9}\right),
\]
\[
\phi_{slack}(j)=\operatorname{clip}\left(
\min_i\frac{C_i-r_{ij}}{c_i},-1,1\right).
\]

依据：[WeakDecoder.primitives](../mkp_Model.py#L282)。`price_ratio` 分母是 **`C_i+1`**，不是 c_i 或未平滑的 c_rem_i。efficiency/slack 直接除原始 c，当前实现未作 `clamp_min`；调用者应提供正容量，零容量数据需要另行处理。

### 6.2 标准化与最终 logits

每个基元只在当前 AVAILABLE 集合 `A` 上分别统计：

\[
\bar\phi_k=\frac{\sum_jA_j\phi_k(j)}{\max(\sum_jA_j,1)},\quad
Var_k=\frac{\sum_jA_j(\phi_k(j)-\bar\phi_k)^2}{\max(\sum_jA_j,1)},
\]
\[
z_k(j)=\operatorname{clip}\left(
\frac{\phi_k(j)-\bar\phi_k}{\sqrt{Var_k+10^{-8}}+10^{-6}},-5,5\right),
\qquad s_j=\kappa_t\sum_kw_{tk}z_k(j)+\delta_{tj}.
\]

依据：[WeakDecoder._z / __call__](../mkp_Model.py#L273)。标准化集合不是 fits 集合；即便在 hard_mask 模式，不可放但未处理物品也参与标准化。δ 加在 κ 乘积外侧，κ 并不缩放 δ。

`WeakDecoder` 是普通 Python 类，没有自己的可训练权重；θ 由神经网络输出，且固定公式对价格调整、w、κ、δ 可微，训练会通过它向控制器和编码器反传。因存在 `res_head`，不能将整个模型描述成没有可学习的物品评分分量。

### 6.3 三种可行性模式与结构 mask

| 模式 | logits 上的实际操作 | 语义 |
|---|---|---|
| `perceive`（默认） | 不额外施加可行性处罚或屏蔽 | 允许选择任意未处理物品，环境决定接受/拒绝。 |
| `soft_mask` | `s_j -= beta*(1[viol_j>0]+viol_j)` | 默认 beta=5；不仅是 `-beta*viol`。 |
| `hard_mask` | 不满足 `st.fits` 的 score 置为 `NEG=-1e4` | 使用显式可行性过滤的数值近似。 |

其中

\[
viol_j=\max_i\frac{\max(r_{ij}-c^{rem}_i,0)}{r_{ij}+1}.
\]

所有模式最后都把 `~available` 的 score 设为 `NEG=-1e4`；已 done 行的全部 logits 覆写为 0，作为同批前向占位。`NEG` 是有限数，并非数学上的 `-∞`；正常配置下用于数值屏蔽。`hard_mask` 的 fits 使用环境 ε 容差，soft violation 则直接判断 `r-c_rem>0`。依据：[NEG](../mkp_Model.py#L66)、[WeakDecoder.__call__](../mkp_Model.py#L298)。

## 7. Rollout 接口、动作分布与梯度路径

`ConstrainedMKPPolicy.forward(env,inst,num_samples=1,decode="sample",temperature=1.0,amp=False,pre=None,return_trace=False)` 执行一条或多条完整轨迹：

1. 使用传入 `pre=(mu,ub)`，否则执行 `env.precompute(inst)`。
2. 计算静态特征和静态编码，然后把同实例复制 K 次。
3. reset 环境；每步重新提取动态特征、计算 θ 和 logits。
4. `decode="greedy"`：选 `argmax(logits)`，temperature 不生效；其他 decode 字符串走采样分支，按 `softmax(logits/temperature)` 做 multinomial。
5. 只对 `active=~done` 的行累计动作 log-prob 和熵，执行环境转移；直到全批 done。

done 行虽然仍有占位动作，但环境忽略，log-likelihood 和熵均乘 active=0。`model.eval()` 不自动等于 greedy，动作方式由 decode 参数决定。源码：[ConstrainedMKPPolicy.forward](../mkp_Model.py#L336)。

| 返回键 | shape / 类型 | 精确定义 |
|---|---|---|
| `reward` | `(BK,)` | 第 2 节终局回报。 |
| `profit`,`n_reject`,`ub` | 各 `(BK,)` | 各轨迹最终收益、拒绝数、所复用的静态上界。 |
| `log_likelihood` | `(BK,)` | 活跃决策所选动作 log probability 的**总和**。 |
| `entropy` | `(BK,)` | 每条轨迹有效决策分布熵的**平均值**，分母至少为 1。 |
| `solution` | `(BK,N)` | 已接受物品的 0/1 向量。 |
| `steps` | Python int | 批量循环长度，即本批最长轨迹长度。 |
| `kappa`,`prim_w` | float / 长 P 列表 | 每步先对 active 行平均，再对全批时间步平均。 |
| `abs_residual` | float | 每步在全部 AVAILABLE 物品上求残差绝对值均值，再跨步平均；可能混入 done 行。 |
| `log_price_abs` | float | 每步对全部行和资源求绝对 log 调整均值，再跨步平均；包含 done 占位行。 |
| `primitives` | 长 P 字符串列表 | 与 `prim_w` 对齐的基元顺序。 |
| `trace`（可选） | 每步一个 dict 的列表 | CPU detach 的 `action,kappa,prim_w,lambda,fits_before`；没有 active mask。 |

以上统计并非统一的“所有活跃决策等权均值”；分析时不能混为一谈。trace 的 done 占位动作也会被记录，如需精确逐轨迹分析应另行保存 active/长度。依据：[policy.forward:384](../mkp_Model.py#L384)。

梯度路径为 `loss → 轨迹 log-prob/熵 → logits → λ,w,κ,δ → controller/encoder`。环境 step、固定状态特征、LP 预计算无梯度；离散 action 和状态转移不做可微优化。训练时满足 `model.training && grad_checkpoint && grad_enabled` 才对 `_step_core` 使用梯度 checkpoint；CUDA AMP 使用 bf16 autocast 包围 encoder/controller，decoder/logits 与环境数据使用 fp32。依据：[env no_grad](../mkp_env.py#L211)、[policy.forward:344](../mkp_Model.py#L344)。

## 8. 当前训练是 REINFORCE；PPO 的实现状态

### 8.1 终局共享基线与真实 loss

每个训练 step 随机选 N、M，生成 B 个新实例，LP 预处理后，每实例采样 K 条轨迹。记回报为 `R_bk`，轨迹 log-likelihood 为 `ell_bk`：

\[
b_b=\frac1K\sum_{k=1}^KR_{bk},\qquad A_{bk}=R_{bk}-b_b.
\]

基线包含当前轨迹本身，不是 leave-one-out，也不是价值网络。默认启用 advantage normalization：

\[
\widehat A=\frac{A}{std(A_{\text{当前 chunk}})+10^{-8}}.
\]

`std()` 使用 PyTorch 默认样本标准差，对当前 chunk 的全部 advantage 元素统计。不是对整批 B 的统一标准差，也不是每个实例单独归一化。

\[
\ell_{bk}=\sum_{t\text{ active}}\log\pi(a_{bkt}\mid s_{bkt}),\qquad
\bar H_{bk}=\frac{\sum_{t\text{ active}}H(\pi_{bkt})}{\max(T_{bk},1)},
\]
\[
\mathcal L=-\operatorname{mean}_{b,k}
  [\operatorname{stopgrad}(\widehat A_{bk})\ell_{bk}]
  -\beta_H\operatorname{mean}_{b,k}\bar H_{bk},\qquad\beta_H=0.01.
\]

未启用归一化时直接用 A。没有折扣、bootstrap、逐步 value target 或 GAE；训练器只使用终局 reward 和完整轨迹概率。依据：[train_step](../mkp_Trainer.py#L157)、[policy.forward](../mkp_Model.py#L384)。

### 8.2 内存分块与优化器更新

```text
per_chunk = max(1, train.max_rollouts_per_chunk // K)
for each chunk of complete instances:
    sample all K trajectories per instance
    compute chunk loss
    backward(loss * chunk_instance_count / B)
clip_grad_norm(all model parameters, 1.0)
if gradient norm is finite:
    Adam.step()
scheduler.step()
global_step += 1
```

一次训练 step 中每个实例的 K 条轨迹不会被拆散；所有 chunk 只对应一次 optimizer 更新。默认 B=64,K=16、上限 1024，刚好一个 chunk。若 K 超过上限，至少仍处理一个完整实例，故此配置不是严格的 rollout 数上限。由于 advantage std 在 chunk 内计算，改变分块也可能改变更新，而不只是内存占用。依据：[train_step:167](../mkp_Trainer.py#L167)。

`K=1` 时同实例共享基线使优势恒为 0，收益策略梯度消失，只剩熵项；仅一个 advantage 元素时默认 std 还会出现自由度不足问题。当前配置未对 K>1 作显式校验。

优化器为 Adam（默认 lr=1e-4，weight_decay=0），warmup 500 步，随后按总 `epochs*steps_per_epoch` 余弦降到 `lr_min=1e-6`。梯度范数非有限时跳过 optimizer 更新，但调度器和 global_step 仍前进。`set_seed()` 设置 Python、NumPy、Torch CPU/CUDA 种子，未启用 deterministic algorithms。依据：[MKPTrainer.__init__ / _lr_lambda](../mkp_Trainer.py#L62)、[set_seed](../mkp_Trainer.py#L37)、[train_step:196](../mkp_Trainer.py#L196)。

### 8.3 PPO 相关概念的当前状态

| 项目 | 当前实现 |
|---|---|
| Actor/策略网络 | 有：编码器和控制器共同训练，经固定 scorer 产生动作分布。 |
| Critic / value head / value loss | 无。 |
| old-policy log-prob、重要性 ratio | 无。 |
| PPO clipped surrogate objective | 无。 |
| GAE、折扣、bootstrap | 无。 |
| 每批数据的多轮 PPO epoch/minibatch 更新 | 无；每个训练 step 采样后更新一次。 |
| 逐步 rollout buffer / replay buffer | 无；只有前向过程累计的概率、熵和终局输出。 |
| teacher、最优动作标签、Oracle θ、反事实分支监督 | 当前训练器均无。 |
| LP 是否充当 Critic | 否；静态上界不是状态价值预测器。 |

依据：完整的 [MKPTrainer.train_step](../mkp_Trainer.py#L157) 和 [ConstrainedMKPPolicy.forward](../mkp_Model.py#L336)。若未来改 PPO，至少需要保存逐步 state/action/active/old log-prob、实现相应价值与 advantage 方案、重算同一动作概率并实现 ratio/clipping 和更新批次控制。当前 `return_trace` 缺少这些训练信息，不能直接当 PPO buffer。此段是实现缺口说明，不代表仓库已经采用某种 PPO 方案。

### 8.4 验证与训练日志的不同口径

固定合成验证集用种子 `20240607`，默认每个 `(N=100,M=5/10)` 组合 64 个实例。每次验证运行一次 greedy，计算 `100*(1-profit/UB)`，先组内平均，再将各组均值等权平均为 `gap_mean`；更小者更新 `best.pt`。验证不使用局部搜索、基准 best-known 或拒绝惩罚。依据：[_build_val_set](../mkp_Trainer.py#L113)、[validate / fit](../mkp_Trainer.py#L210)。

训练 CSV 的 `gap_ub` 则是每个实例 K 条轨迹的最大 profit 相对 UB 的 gap 后求均值；`best_reward` 是最大 reward 的均值，两者不等价。`reject_rate` 为 `mean(n_reject / batch_steps)`，batch_steps 是各 chunk 最长轨迹长度，并非精确逐轨迹尝试率。依据：[train_step:184](../mkp_Trainer.py#L184)。

## 9. 推理、评测与后处理边界

### 9.1 默认模型评测路径

`Benchmark_zeroShot.py` 从 checkpoint 恢复模型并置 eval，整个 `evaluate_instance()` 使用 `@torch.no_grad()`。每次只评测一个实例：

1. 计算一次 μ、UB 以及单次静态 `greedy_dual_ratio` 参考。
2. 模型做 1 次 greedy，再做配置数量的采样，默认 256 条、temperature=1；按 `eval.max_rollouts_per_chunk` 分块。
3. `policy_obj` 是 greedy 与全部 sampled solutions 的最大收益；`policy_mean_obj` 也包含 greedy。dual-greedy 参考不加入策略候选池。
4. 可选按 0/1 选择向量去重，取收益最高的最多 `ls_pool=16` 个起点，再分 `ls_batch=16` 执行搜索。
5. `final_obj=max(policy_obj,全部搜索结果)`；关闭搜索则与 policy_obj 相同。

LP 结果在 greedy/sampling 之间复用；每次 policy forward 仍重新算静态特征和神经静态编码，默认未跨多次 forward 缓存这些 embedding。依据：[evaluate_instance](../Benchmark_zeroShot.py#L59)、[_top_distinct](../Benchmark_zeroShot.py#L50)。

### 9.2 可选后处理的确切行为

排序使用静态 `ratio_j=p_j/max(Σ_i μ_i*r_ij/c_i,ε)`，不使用控制器当前 λ 或 δ。

| 算子 | 实际动作 | 代码 |
|---|---|---|
| ADD / fill | 显式筛选仍可放的未选物品，按 score 最大者逐个补入。 | [greedy_fill](../mkp_refine.py#L34) |
| 1-1 SWAP | 穷举一个已选与一个未选的交换，检查全部资源，取收益增量 >0.5 的最佳交换。构造 `(B,N,N)` 候选矩阵。 | [swap_11](../mkp_refine.py#L51) |
| DROP-REFILL | 只选最低 ratio 的最多 `drop_k=32` 个已选物品，分别尝试删除一个再 refill，禁止该物品立即重加，保留增益 >0.5 的最佳分支。 | [drop_refill](../mkp_refine.py#L75) |
| local search | 先 fill，再重复 `swap→fill→drop_refill`；每行无改善或达到最多 200 轮时结束。 | [local_search](../mkp_refine.py#L123) |
| ILS | 先局搜；每轮拟删除 2..4 个物品，受批内最少已选数量限制；Gumbel 抽样偏向低 ratio 物品，按 `logratio+0.3*Uniform` refill 后再局搜；当前解只接受不劣解，另存严格更好解。默认 100 轮。 | [iterated_local_search](../mkp_refine.py#L141) |

这些操作是真实的推理后处理搜索，使用可行性筛选，和默认策略的 `perceive` 动作机制不同。所有算子 no_grad，未接入训练或合成验证；有限邻域/有限轮数不提供全局最优性证明。输入正常由 0/1 操作构造；`is_feasible()` 本身只检查资源不超额，不单独认证二进制性。依据：[mkp_refine.py](../mkp_refine.py)、[is_feasible](../mkp_refine.py#L179)。

`--baseline_only` 不加载模型，而是用 `max(ls_pool,1)` 个静态 dual-ratio 起点，首个不加噪声，其余给 logratio 加 `0.3*Uniform`，然后也可运行搜索；此模式下仍名为 `policy_obj/gap_policy` 的字段实际记录启发式多起点结果。依据：[evaluate_instance:90](../Benchmark_zeroShot.py#L90)。

### 9.3 输出指标与计时的精确定义

| 字段 | 定义/解释 |
|---|---|
| `gap_dual_greedy` | 静态参考收益相对 best-known 的 gap。 |
| `gap_greedy` | 已训练策略单次 greedy 的 gap。 |
| `gap_policy` | greedy+samples 最好收益的 gap；模型路径中无局部搜索。 |
| `gap_final` | 策略候选加可选搜索后最好收益的 gap。 |
| `gap_lp_bound_to_best` | `100*(UB-best_known)/best_known`，不是 LP 对偶求解误差。 |
| `greedy_rejects` | greedy 轨迹拒绝数。 |
| `greedy_steps` | 单条 greedy 轨迹步数；此处 B=K=1。 |
| `sample_mean_rejects` | sampled trajectories 的平均拒绝数；不含 greedy。 |
| `sample_feasible_pick_rate` | 实际为 `1-mean(sampled_rejects/N)`；分母是物品数 N，**不是实际尝试数**。 |
| `n_optimal_final` | 汇总时统计 `gap_final <= 1e-9` 的条数，表示达到/超过文件 best-known；字段名不构成最优性证明。 |

各收益 gap 统一为 `100*(best_known-objective)/best_known`，可能为负。分组 α 是 `round(sum_i c_i / sum_ij r_ij,2)`；summary 同时生成 `(M,N,α)`、`(M,N,all)` 和 overall，均为逐实例算术平均。依据：[benchmark.main:169](../Benchmark_zeroShot.py#L169)、[summary](../Benchmark_zeroShot.py#L195)。

`time_policy_s` 包含 LP、参考启发式、greedy+sampling、拼接和可行性断言；`time_refine_s` 包含候选去重/筛选与搜索；`time_total_s` 覆盖整个单实例评测，排除模型加载、MAT 读取、实例构建及其设备搬移。CUDA 在边界同步。不能把 policy 时间当纯网络前向延迟。依据：[evaluate_instance:62](../Benchmark_zeroShot.py#L62)。

### 9.4 数据与训练/推理边界表

| 分量/数据 | 训练 | 固定合成验证 | 零样本基准 |
|---|---|---|---|
| 在线合成实例 | 每步新生成 | 初始化生成后固定 | 不用 |
| `.mat` / best-known | 不读取 | 不读取 | 用于实例构建和 gap，未作为模型特征 |
| LP μ/UB | 有，no_grad | 有，预缓存 | 有，每实例一次 |
| 神经编码与 θ | 有梯度 | no_grad | no_grad |
| 固定 scorer / 环境接受拒绝 | 有；梯度只经过 scorer 的策略参数输入 | 有 | 有 |
| 动作方式 | K 次 sample | 1 次 greedy | greedy + samples |
| 参数更新 | REINFORCE | 无 | 无 |
| 局部搜索/ILS | 无 | 无 | 可选，默认有 |
| 整数最优标签/Oracle θ | 无 | 无 | 无 |

依据：[train_step / validate](../mkp_Trainer.py#L157)、[evaluate_instance](../Benchmark_zeroShot.py#L59)、[static_features / dynamic_features](../mkp_env.py#L269)。基准 `best` 虽存在于 `MKPInstances` 并随 rollout 复制，当前特征、controller 和 scorer 均不使用它。

基准主入口固定使用 `read_mat()` 的 Chu 风格位置字段布局，再按 M/N 过滤并按文件顺序截取 `max_instances`。`read_GK()` 单独实现了另一布局，但没有接到 benchmark 主入口；不能只换 MAT 路径就默认支持 GK 或其他所有家族。依据：[read_mat.py](../read_mat.py#L7)、[read_GK](../read_mat.py#L40)、[benchmark.main:154](../Benchmark_zeroShot.py#L154)。

## 10. 配置、checkpoint 与恢复语义

### 10.1 配置入口和输出文件

全部默认值在 `mkp_config.py` 的 `DeviceConfig/EnvConfig/ModelConfig/TrainConfig/EvalConfig`。CLI 支持 `--section.field value` 或 `--section.field=value`，列表以逗号分隔；未知字段抛错。`--print_config` 在创建训练器/载入模型前打印并返回，但运行入口仍需已安装其 import 依赖。相对资源路径通过 `resolve_path()` 相对项目根目录解析。依据：[apply_overrides](../mkp_config.py#L224)、[resolve_path](../mkp_config.py#L255)、[train.main](../train_ConstrainedMKP.py#L18)。

训练输出：

```text
runs/<run_name>/config.json, log.txt, train_log.csv, val_log.csv
runs/<run_name>/TensorBoard events  # 可成功导入 TensorBoard 时
checkpoints/<run_name>/last.pt, best.pt
checkpoints/last.pt, best.pt        # 当前保存的快捷副本，会被后续运行覆盖
```

默认 run_name 为时间戳。每 epoch 结束保存 last；验证更优才保存 best。checkpoint 键为 `model,optimizer,scheduler,epoch,global_step,best_val,config,run_name`。没有 RNG 状态、epoch 内位置、源码/数据/checkpoint hash 或版本 schema。依据：[MKPTrainer.__init__](../mkp_Trainer.py#L62)、[_state / save](../mkp_Trainer.py#L129)。

### 10.2 三种加载行为不可混淆

| 模式 | 结构/配置来源 | 恢复内容 | 进度行为 |
|---|---|---|---|
| `train.resume` | 本次 CLI/default cfg 先建模型 | model、optimizer、scheduler、epoch、global_step、best_val | 从保存的下个 epoch 继续；不自动套用 checkpoint config。 |
| `train.init_from` | 本次 CLI/default cfg 先建模型 | 仅 model | 新 optimizer/scheduler，训练进度从头。 |
| 评测 `load_policy` | 从 checkpoint config 重建 | model；仅 eval/device 用调用方 cfg 覆盖 | 置 eval，不恢复训练过程。 |

训练同时提供 resume/init_from 时 resume 优先。三条路径均要求权重结构兼容；没有自动迁移 schema。resume 默认产生新时间戳目录，不自动接着原日志目录写；若手动复用 run_name，CSV logger 会再次写表头，config 也会重写。依据：[MKPTrainer.__init__](../mkp_Trainer.py#L62)、[_load](../mkp_Trainer.py#L145)、[_CSVLogger](../mkp_Trainer.py#L45)、[load_policy](../mkp_Trainer.py#L268)。

`fit()` 在保存前设置 `epoch=ep+1`。因此日志 epoch 从 0 开始，checkpoint epoch 表示已经完成的轮数。`--train.epochs` 是目标总轮数，不是追加轮数。resume 不恢复 RNG，epoch 中断后只能从最后保存的边界重做；调度器的 lambda 又依赖本次配置的 `epochs*steps_per_epoch`，改变总预算会影响后续学习率，不能宣称精确复现原轨迹。依据：[fit](../mkp_Trainer.py#L223)、[_lr_lambda](../mkp_Trainer.py#L99)。

评测的另一个限制：实际 model/env 采用 checkpoint 配置，但结果 `config.json` 保存的是 CLI 构造的 cfg。当 CLI 的 env/model 与 checkpoint 不同时，输出 JSON 不完整反映实际生效配置。当前结果目录名包含 run_name/epoch，但没有 checkpoint SHA。严谨复现应另外保存有效配置、源代码版本、数据/checkpoint SHA 和环境版本。依据：[benchmark.main:140](../Benchmark_zeroShot.py#L140)、[load_policy](../mkp_Trainer.py#L268)。

## 11. 改动检查与当前文档验证范围

接手实现时，可按以下依赖定位改动范围：

| 想改什么 | 需要一起核对的契约 |
|---|---|
| 状态特征 | `static_features/dynamic_features` 的顺序和维度、`F_*` 常量、StateEncoder 输入层、已有 checkpoint。 |
| θ 或基元 | `ModelConfig`、策略头与初始化、`_step_core` 输出键、WeakDecoder、strategy stats/trace、CSV 字段。 |
| 可行性机制 | 环境接受/拒绝与终止、fits 容差、结构 mask、三种 decoder 模式、评测可行性断言。 |
| LP | normalized/raw 单位、上界性质、no_grad、reward/特征/参考启发式/搜索的共享依赖。 |
| 训练算法 | 第 8 节真实目标、chunk 归一化、参数更新次数、验证选模、checkpoint 与调度恢复。 |
| 推理或搜索预算 | greedy/sample/final 分开记录、候选去重、计时范围、baseline_only 的不同含义。 |
| 新数据家族 | parser 布局、r 的轴顺序、正容量、best-known 语义、过滤实例 ID。 |

本次文档工作核对了当前源码、配置及已有 CSV/log，并重新汇总了历史指标；没有运行训练、重跑 Chu benchmark、验证 LP 与 scipy 的精度或认证整数最优性。当前默认 Python 缺少 torch/numpy/scipy，因此本次检查限于静态契约和标准库可完成的数据/文档一致性检查。当前仓库未附正式测试套件，不能将本文作为全部运行路径已通过测试的证明。
