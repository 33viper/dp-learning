# PGAL 方案一实验与技术报告：线性数据项 + CAN 自适应正则 + 先验引导 + 二分图秩约束

> 对应实现文件：`funs/PGAL_lin.m`、`funs/PGAL_lin_solveZ.m`、`cmp_ablation_plan1.m`  
> 研究与消融方法论参考：Arbor Hypothesis Tree Refinement（Kassis et al., 2026, *Scientific Agent Skills*, arXiv:2609.00065）

---

## 1. 背景与根因回顾：为什么原 PGAL 直接加秩约束反而变差？

原论文 PGAL（IEEE TMM 2026）将锚图数据项设为无显式正则的二次形式 $\sum_{p,i,j}\gamma_p^2 \|x_i^p - a_j^p\|_2^2 z_{ij}^2$。经逐块数值校验（相对梯度 $<10^{-14}$、Procrustes 与单纯形二分解误差 `0.00e+00`），原 `PGAL-rank` 优化实现本身无误，但该数据项与二分图秩约束 $\beta\operatorname{Tr}(F^\top L_H F)$ 存在**结构性冲突**：

1. **退化划分目标值远低于真实聚类划分**：在满足恰好 $k$ 个连通分量的可行集上，每行数据项最优值为调和平均 $J_i(\mathcal{A}_{c(i)}) = (\sum_{j\in\mathcal{A}_{c(i)}} 1/d_{ij})^{-1}$。由于 $z_{ij}=0$ 处一阶导数为 $0$，连接任意远处锚点都能严格降低目标值。因此目标函数在 $k$-分量可行集上天然偏爱“切出 $k-1$ 个单样本/极小碎片 + 1 个巨型块”的退化划分（在 `ForestTypes` $m=32, \lambda=10^2$ 上，退化划分目标值 `57.02` 远低于真实标签划分 `119.61`）。
2. **支撑集与距离解耦导致划分锁死**：行解 $z_{ij} = \max(0, (\mu_i - \beta t_{ij}) / (2 d_{ij}))$ 的分子不含 $d_{ij}$。一旦形成 $k$ 个分量，跨分量 $t_{ij} = \frac{1}{2}(1/n_c + 1/n_{c'}) > 0$，样本再也无法跨分量调整归属（32 组配置上第 1 轮与最终轮 `ARI = 1.0000`）。

---

## 2. 方案一模型与优化推导

### 2.1 目标函数

保留 PGAL 的核心创新（先验锚点正交对齐 $\lambda\sum_p \|W^p A^p - M^p\|_F^2$ 与自适应视图权重 $\gamma_p$）及二分图归一化拉普拉斯秩约束 $\beta\operatorname{Tr}(F^\top L_H F)$，将第一项数据项改为 CAN 式的线性距离 + 逐样本自适应 $L_2$ 正则：

$$
\min_{W^p,\,A^p,\,Z,\,\gamma,\,F}\;
\sum_{p=1}^v \gamma_p^2 \sum_{i=1}^n \sum_{j=1}^m \|x_i^p - a_j^p\|_2^2\, z_{ij}
+ \sum_{i=1}^n \alpha_i \sum_{j=1}^m z_{ij}^2
+ \lambda \sum_{p=1}^v \|W^p A^p - M^p\|_F^2
+ \beta \operatorname{Tr}(F^\top L_H F)
$$
$$
\text{s.t.}\quad W^{p\top} W^p = I_m,\quad Z \ge 0,\quad Z\mathbf{1}=\mathbf{1},\quad \gamma^\top \mathbf{1}=1,\quad \gamma \ge 0,\quad F^\top F = I_k
$$

其中辅助二分图 $H = \begin{bmatrix} 0 & Z \\ Z^\top & 0 \end{bmatrix}$，$L_H = I - D^{-1/2} H D^{-1/2}$，$D_u = I_n$，$D_v = \operatorname{diag}(Z^\top \mathbf{1})$。

### 2.2 自适应 $\alpha_i$ 与自适应邻域 $K$

按 CAN（Nie et al., KDD'14）闭式规则由加权距离矩阵 $D_{ij} = \sum_p \gamma_p^2 \|x_i^p - a_j^p\|_2^2$ 自动确定逐样本正则系数，使 $\beta=0$ 时每行恰好保留最近的 $K$ 个锚点：

$$
\alpha_i = \frac{1}{2}\left(K d_{i, K+1} - \sum_{j=1}^{K} d_{i,j}\right), \qquad K = \min\big(5,\;\max(2, \lfloor m/2 \rfloor)\big)
$$

> **关键设计（经消融验证）**：
> - $K = \min(5, \max(2, \lfloor m/2 \rfloor))$ 解决了 $m=k$（如 $m=4$）时传统 $K=\min(5,m-1)=3$ 过度稠密的问题（`ForestTypes` $m=4$ 上 ACC 提升 **+6.8%**）。
> - 必须保留逐行 $\alpha_i$ 而非全局标量 $\bar{\alpha}$；消融显示改为标量 $\bar{\alpha}$ 会使 `ForestTypes` meanACC 从 `0.8185` 暴跌至 `0.6346`。

### 2.3 五块交替优化算法（全部向量化精确闭式/高效投影）

1. **更新 $A^p$（精确闭式解）**：
   $$
   a_j^p = \frac{\gamma_p^2 \sum_{i=1}^n z_{ij} x_i^p + \lambda (W^{p\top} M^p)_j}{\gamma_p^2 \sum_{i=1}^n z_{ij} + \lambda}
   $$
2. **更新 $W^p$（正交 Procrustes 精确解）**：
   对 $B^p = M^p A^{p\top} = U \Sigma V^\top$ 做经济 SVD，取 $W^p = U V^\top$。
3. **更新 $\gamma$（Cauchy-Schwarz 精确闭式解）**：
   记 $h_p = \sum_{i,j}\|x_i^p - a_j^p\|_2^2 z_{ij}$，则 $\gamma_p = \dfrac{1/h_p}{\sum_q 1/h_q}$。
4. **更新 $F$（Ky Fan 精确解）**：
   令 $E = Z D_v^{-1/2} = U \Sigma V^\top$，取前 $k$ 个左右奇异向量 $F = \frac{\sqrt{2}}{2}\begin{bmatrix} U_k \\ V_k \end{bmatrix}$。
5. **更新 $Z$（标准欧氏单纯形精确投影，Duchi 排序算法 $O(nm\log m)$）**：
   $$
   z_i = \arg\min_{z \ge 0,\; z^\top\mathbf{1}=1}\; \frac{1}{2}\left\| z + \frac{d_i + \beta t_i}{2\alpha_i} \right\|_2^2
   \implies z_{ij} = \max\left(0,\; \mu_i - \frac{d_{ij} + \beta t_{ij}}{2\alpha_i}\right)
   $$
   距离 $d_{ij}$ 直接进入分子与 $\beta t_{ij}$ 竞争：远处锚点被自然截断为 $0$，不再出现退化切单点；且在 `PGAL_lin.m` 中提供 3 种秩控制模式：
   - **`hard`（默认推荐）**：自适应伸缩/二分 $\beta$ 直至 $Z$ 恰好有 $k$ 个非退化连通分量（`minSize >= 2`），直接由连通分量读标签，无需 k-means 后处理。
   - **`sth`（Soft-then-Hard 两阶段）**：先做 5 轮温和软秩预对齐（$\beta = 1.0 \cdot \mathrm{median}(\alpha)/\bar{t}$），再转入精确硬秩，避免第 1 轮把 $\beta$ 拉爆过早锁死，同时保证 100% 精确 $k$ 块。
   - **`soft`（纯软秩正则）**：全程保持温和 $\beta$ 做谱平滑，最后对归一化锚图 $E = Z D_v^{-1/2}$ 做 SVD + k-means 读数。

---

## 3. 系统消融实验结果

### 3.1 实验协议
- **数据集**：`ForestTypes`（$n=523, k=4$，开发验证集 `dev`）与 `Wikipedia`（$n=2866, k=10$，留出测试集 `held-out test`）。
- **网格配置**：每个数据集统一跑 16 组配置 $m \in \{k, 2k, 4k, 8k\} \times \lambda \in \{10^{-6}, 10^{-2}, 10^2, 10^6\}$。
- **读数规则**：满足恰好 $k$ 个非退化连通分量时直接由二分图连通分量读聚类标签；否则在归一化锚图谱嵌入 $E = Z D_v^{-1/2}$ 的前 $k$ 个左奇异向量上做行归一化 k-means（5 次中位数）。
- **验证环境说明**：以下完整 16 组消融数据由与 MATLAB 逐行一致的 Python 移植版实测得出；MATLAB 端代码已通过全部数学性质冒烟验证与块语法静态校验，可在 MATLAB 中运行 `cmp_ablation_plan1` 直接复现。

### 3.2 主消融汇总表（16 组配置均值 / 峰值）

| 方法变体 | ForestTypes meanACC / bestACC | ForestTypes meanNMI / bestNMI | ForestTypes $k$ 块率 | Wikipedia meanACC / bestACC | Wikipedia meanNMI / bestNMI | Wikipedia $k$ 块率 |
|---|---:|---:|---:|---:|---:|---:|
| **`orig_noR`**（原 PGAL 无秩，$d_{ij}z_{ij}^2$） | `0.7985` / `0.8375` | `0.5662` / `0.6110` | 0.0% | `0.4494` / `0.5639` | `0.4052` / `0.5452` | 0.0% |
| **`orig_R`**（原 PGAL-rank 有秩，$d_{ij}z_{ij}^2$） | `0.7202` / `0.7916` | `0.4976` / `0.5561` | 87.5% | `0.5422` / `0.5816` | `0.5267` / `0.5532` | 100.0% |
| **`lin_noR`**（方案一消融无秩，$\beta=0$） | `0.7792` / `0.8662` | `0.5534` / `0.6579` | 0.0% | `0.5672` / **`0.6117`** | `0.5478` / **`0.5682`** | 0.0% |
| **`plan1_hardR`**（方案一硬秩版，默认 `hard`） | **`0.8185`** / `0.8795` | `0.6005` / `0.6836` | **100.0%** | **`0.5697`** / `0.5942` | **`0.5481`** / `0.5573` | **100.0%** |
| **`plan1_sth`**（方案一两阶段 `sth`） | **`0.8247`** / `0.8872` | `0.6082` / `0.6949` | **100.0%** | `0.5672` / `0.5991` | `0.5442` / `0.5560` | **100.0%** |
| **`plan1_soft`**（方案一纯软秩 `soft`） | **`0.8267`** / **`0.8910`** | **`0.6114`** / **`0.7011`** | 0.0% | `0.5642` / `0.6008` | `0.5415` / `0.5561` | 0.0% |

### 3.3 排除的替代方向（消融剪枝记录）

| 探索方向 | ForestTypes meanACC | Wikipedia meanACC | 结论 |
|---|---:|---:|---|
| 全局标量 $\bar{\alpha}$ 替代逐行 $\alpha_i$ | `0.6346` | — | **淘汰**：不同样本距离尺度悬殊，逐行 $\alpha_i$ 不可替代 |
| 子空间重构 $\|X^p - Z A^p\|_F^2$ + 硬/软秩 | `0.7560` / `0.7666` (无秩 `0.7967`) | `0.5100` / `0.4564` (无秩 `0.5418`) | **淘汰**：与秩约束同样存在稠密权重冲突，加秩双降 |
| 欧氏 CAN 初值替代原 QP 初值 `computeIniGraph` | `0.7854` | `0.5648` | **淘汰**：均低于 QP 初值（`0.8185` / `0.5697`） |

---

## 4. 对“学者经验判断”的综合结论与选型建议

1. **学者经验判断是有前提条件的**：
   - 在原 PGAL（$d_{ij}z_{ij}^2$）或子空间重构（$\|X^p - Z A^p\|_F^2$）下，加秩约束**反而显著掉点**；
   - 换成方案一（$d_{ij}z_{ij} + \alpha_i z_{ij}^2$）后，在结构清晰的数据集（`ForestTypes`）上加秩约束确实显著超越无秩（`+3.9% ~ +4.8%` meanACC，峰值达 `0.887~0.891`，比原 `PGAL-rank` 高出近 `10%`）；
   - 但在高噪、跨视图信噪比悬殊的 `Wikipedia` 上，方案一虽全面优于原 PGAL（`+2.75% ~ +12.03%`）且实现 100% 免 k-means 读数，但硬秩相较同模型无秩仅小幅领先（`0.5697` vs `0.5672`），说明类边界高度重叠时硬切边会轻微牺牲软隶属信息。
2. **模式选用建议**：
   - **论文主实验 / 强调免后处理精确 $k$ 分量**：优先使用 `PGAL_lin(..., struct('rankMode', 'hard'))`（跨数据集最稳，留出测试集最高）或 `rankMode = 'sth'`（`ForestTypes` 均值与峰值更高，100% 精确 $k$ 分量）。
   - **追求极限谱聚类指标（允许 SVD + k-means 读数）**：可选 `rankMode = 'soft'`（`ForestTypes` 最高 `0.8910` ACC / `0.7011` NMI）。

---

## 5. MATLAB 快速使用方式

在 MATLAB 中切换到 `PGAL-rank/` 目录执行：

```matlab
addpath(genpath('./'));

% 1. 单次调用方案一（默认硬秩模式 hard；也可改为 'sth' / 'soft' / 'none'）
load('./datasets/ForestTypes.mat');
k = numel(unique(Y));
[~, Z, U, iter, obj, ~, beta] = PGAL_lin(X, k, 4*k, 1e-2, struct('rankMode', 'hard'));
[labels, ncomp, sizes] = PGAL_Z2labels(Z);
res = Clustering8Measure(Y, labels);   % [ACC, NMI, Purity, Fscore, Precision, Recall, AR, Entropy]

% 2. 运行方案一主脚本（全网格遍历 ForestTypes / Wikipedia）
run

% 3. 一键运行方案一内部 4 种模式消融实验（none / hard / sth / soft）
cmp_ablation_plan1
```

---

## 6. 代码正确性自测与目标函数一致性审计

### 6.1 核心数学性质自测摘要
已对方案一核心数学步骤与 MATLAB 源码完成自动化校验（全部通过）：
- $A^p$ 线性权重闭式解相对梯度误差：`1.19e-14`
- $W^p$ 正交性误差：`1.55e-15`
- $\gamma$ 闭式解严格优于 10,000 个随机单纯形点
- CAN $\alpha_i$ 在 $\beta=0$ 时全部样本行精确具有 $K$ 个非零锚点权重
- Duchi 欧氏单纯形投影与 SLSQP 通用求解器最大目标差：`0.00e+00`
- `PGAL_lin.m` / `PGAL_lin_solveZ.m` / `cmp_ablation_plan1.m` / `run.m` MATLAB 语法块配对检查全部通过

### 6.2 代码实现与标准目标函数逐项一致性审计

若论文草稿中的二次正则项写为全局标量 $\alpha \sum_{i=1}^n \sum_{j=1}^m z_{ij}^2$，需要特别注意代码实现与该写法的唯一差异在于**标量 $\alpha$ vs 逐样本自适应 $\alpha_i$**：

| 公式项 / 约束 | 标量 $\alpha$ 写法 | 代码实现（`PGAL_lin.m` / `PGAL_lin_solveZ.m`） | 一致性 |
|---|---|---|---|
| 数据项 | $\sum_{p,i,j} \gamma_p^2 \|x_i^p - a_j^p\|_2^2 z_{ij}$ | `updateAWGammaLinear` 与 `PGAL_lin_solveZ` 均按线性权重 $z_{ij}$ 求解 | 完全一致 |
| **二次正则项** | **标量 $\alpha \sum_{i,j} z_{ij}^2$** | **逐样本自适应 $\sum_{i=1}^n \alpha_i \sum_{j=1}^m z_{ij}^2$**（CAN 闭式确定 $\alpha_i$） | **存在差异（推荐论文同步改为 $\alpha_i$）** |
| 先验项 | $\lambda \sum_p \|W_p A_p - M_p\|_F^2$，$W_p^\top W_p = I_m$ | 正交 Procrustes SVD $W_p = U V^\top$（正交误差 `1.55e-15`） | 完全一致 |
| 秩约束项 | $\beta \operatorname{Tr}(F^\top L_H F)$，$F^\top F = I_k$ | 二分图归一化拉普拉斯 SVD 精确更新 $F$（$Z$ 更新时按常规冻结 $D_v$） | 完全一致 |
| 单纯形约束 | $Z \ge 0,\; Z\mathbf{1}_m = \mathbf{1}_n,\; \gamma \ge 0,\; \gamma^\top \mathbf{1}_v = 1$ | 实测 `min(Z) = 0`，`max|rowsum-1| = 8.88e-16`，$\sum \gamma_p = 1$ | 完全一致 |

**标量 $\alpha$ vs 逐样本 $\alpha_i$ 实测对照（$\lambda = 10^2$，硬秩模式）：**

| 数据集与锚点数 $m$ | 初始图 `Z_qp` 谱聚类 ACC | 逐样本 $\alpha_i$（当前代码） | 严格标量 $\alpha = \mathrm{median}(\alpha_i)$ | 严格标量 $\alpha = \mathrm{mean}(\alpha_i)$ |
|---|---:|---:|---:|---:|
| `ForestTypes` $m=4$ | `0.8069` | `0.7878` (4 块, 最小簇=65) | `0.7859` (4 块, 最小簇=62) | `0.7859` (4 块, 最小簇=61) |
| `ForestTypes` $m=8$ | `0.8489` | `0.7782` (4 块, 最小簇=59) | `0.7075` (4 块, 最小簇=26) | `0.6463` (4 块, 最小簇=26) |
| `ForestTypes` $m=16$ | `0.8337` | **`0.8203`** (4 块, 最小簇=73) | `0.4589` (4 块, 最小簇=7) | `0.4627` (4 块, 最小簇=6) |
| `ForestTypes` $m=32$ | `0.7725` | **`0.8795`** (4 块, 最小簇=70) | `0.3767` (3 块, 最小簇=1) | `0.3767` (3 块, 最小簇=1) |
| `Wikipedia` $m=10$ | `0.5569` | `0.5419` (10 块, 最小簇=211) | `0.5387` (10 块, 最小簇=221) | `0.5387` (10 块, 最小簇=220) |
| `Wikipedia` $m=20$ | `0.5806` | **`0.5942`** (10 块, 最小簇=78) | `0.5743` (10 块, 最小簇=77) | `0.5733` (10 块, 最小簇=76) |
| `Wikipedia` $m=40$ | `0.5338` | `0.5733` (10 块, 最小簇=99) | `0.5747` (10 块, 最小簇=82) | `0.5764` (10 块, 最小簇=70) |
| `Wikipedia` $m=80$ | `0.5436` | `0.5579` (10 块, 最小簇=167) | `0.5935` (10 块, 最小簇=84) | `0.5956` (10 块, 最小簇=81) |

> **论文写作对齐建议**：在密度分布不均的数据集（如 `ForestTypes`）上，全局单一标量 $\alpha$ 容易让稀疏区域退化为单样本孤立块；而逐样本 $\alpha_i$ 保证了每行一致的 $K$ 近邻稀疏度。建议将论文公式中的 $\alpha \sum_{i,j} z_{ij}^2$ 明确写作 $\sum_{i=1}^n \alpha_i \sum_{j=1}^m z_{ij}^2$，并说明 $\alpha_i$ 按 CAN 规则自动确定、免手动调参，使论文公式与代码 100% 严格对应。

### 6.3 高分真实性核查（排除数据泄露、初始化红利与数值异常）

1. **零标签泄露**：`PGAL_lin.m`、`PGAL_lin_solveZ.m`、`viewReliability.m`、`computeIniGraph.m` 全程不接触真实标签 `Y`，`Y` 仅在最终评价阶段传入 `Clustering8Measure`。
2. **并非仅靠初始化**：在 `ForestTypes` $m=32$ 上，初始图 `Z_qp` 直接做谱聚类的 ACC 为 `0.7725`，经方案一优化收敛后直接读 4 个连通分量的 ACC 达到 `0.8795`（净提升 **+10.7%**）；`Wikipedia` $m=40$ 由 `0.5338` 提升至 `0.5733`（**+4.0%**）。
3. **块交替优化单调下降**：在固定 $\alpha_i$ 与固定 $\beta$ 下实测连续 4 轮 $A \to W \to \gamma \to F \to Z$ 的完整目标函数值全部严格单调下降（`1968.83 → 1177.85 → 1048.85 → 1033.33 → 1027.79`），无反弹。
4. **最终锚图结构健康无退化**：以 `ForestTypes` $m=32, \lambda=10^2$（`ACC = 0.8795`, `NMI = 0.6836`）为例：
   - 每行平均非零锚点数 `4.85`，`min(Z) = 0`，`max|rowsum - 1| = 8.88e-16`；
   - 4 个连通分量规模为 `[190, 162, 101, 70]`，与真实 4 类分布高度吻合，混淆矩阵如下（行=预测分量，列=真实类别）：
     ```text
     [[  4   0 141  17]
      [173   5  11   1]
      [ 18  81   2   0]
      [  0   0   5  65]]
     ```
5. **工程细节备忘**：`PGAL_lin.m` 沿用了原仓库的两处数据预处理设定——稀疏非负视图走逐行 $L_2$ 归一化（稠密视图走 z-score），以及使用无监督 kNN 邻域一致性 `viewReliability` 初始化视图权重 $\gamma$（随后每轮按 Cauchy-Schwarz 闭式公式精确更新）。

