# PGAL-Plan1（线性数据项 + CAN 自适应正则 + 先验引导 + 二分图秩约束）

本目录为 PGAL（IEEE TMM 2026）的**方案一**独立可运行 MATLAB 实现，通过将原始 $d_{ij} z_{ij}^2$ 数据项重构为 CAN 式的线性距离 + 行自适应 $L_2$ 正则项，从根源消除原目标函数在秩约束下的退化切单点与划分锁死问题。

详细公式推导、消融对比数据与模式选型说明请见 [`REPORT_PLAN1.md`](./REPORT_PLAN1.md)。

## 目录结构

```text
PGAL-rank/
├── README.md                  # 项目说明
├── REPORT_PLAN1.md            # 方案一完整技术与消融实验报告
├── run.m                      # 方案一主运行脚本（遍历 ForestTypes / Wikipedia 全网格）
├── cmp_ablation_plan1.m       # 方案一内部消融脚本（none / hard / sth / soft 对比）
├── funs/
│   ├── PGAL_lin.m             # 方案一主算法（支持 hard / sth / soft / none 模式）
│   ├── PGAL_lin_solveZ.m      # Z 子问题求解器（Duchi 精确欧氏单纯形投影 + 自适应 beta）
│   ├── PGAL_Z2labels.m        # 由满足秩约束的锚图 Z 直接读连通分量聚类标签
│   ├── computeIniGraph.m      # 初始锚图构造（QP 子空间重构初值）
│   ├── viewReliability.m      # 多视图无监督 kNN 邻域一致性初始权重
│   └── mySVD.m                # 加速经济奇异值分解
├── measure/                   # 聚类评测指标（ACC, NMI, Purity, Fscore, AR 等）与 litekmeans
└── datasets/                  # ForestTypes.mat, Wikipedia.mat
```

## 快速运行（MATLAB）

在 MATLAB 中切换到 `PGAL-rank/` 目录：

```matlab
addpath(genpath('./'));

% 1. 运行方案一主脚本（默认 rankMode = 'hard'，可在 run.m 顶部改为 'sth' / 'soft' / 'none'）
run

% 2. 一键运行方案一内部消融（对比 none / hard / sth / soft）
cmp_ablation_plan1
```

## 参考引用

- Li M, Liu S, Yang H, et al. *Enhancing Anchor Representativeness and Stability in Scalable Multi-View Clustering with Prior Guidance*. IEEE Transactions on Multimedia, 2026.
- Nie F, Wang X, Huang H. *Clustering and Projected Clustering with Adaptive Neighbors*. KDD 2014.
