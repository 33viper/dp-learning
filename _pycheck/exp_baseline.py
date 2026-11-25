"""对比 cmp_rank.m 的三种设置, 并加锁定诊断。
(0) 无秩约束 + QP 初值 + 谱嵌入 kmeans
(1) 无秩约束 + 块状初值 + 谱嵌入 kmeans
(2) 当前完整代码(秩约束), 连通分量直接读标签
诊断: initACC = 块状初值所用 kmeans 标签的 ACC;
      ARI(it1,end) = 第 1 次外层迭代后的分量划分与最终划分的 ARI (=1 表示之后再没变过)
"""
import sys
import time
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari
import pgal_port as P

NORANK = dict(adaptive=False, innerMax=1)        # beta=1e-12 固定, 等价于无秩项
names = sys.argv[1:] or ['ForestTypes', 'Wikipedia']
lams = [1e-6, 1e-2, 1e2, 1e6]

for name in names:
    X, Y = P.load(name)
    k = len(np.unique(Y))
    print(f'\n=== {name} n={len(Y)} k={k} ===', flush=True)
    print(f"{'m':>3} {'lam':>6} | {'ACC0':>6} {'ACC1':>6} | {'initACC':>7} {'ACC2':>6} {'NMI2':>6} "
          f"{'kmACC2':>6} {'nc':>3} {'minSz':>5} {'maxFr':>5} {'it':>3} {'beta':>9} {'ARI(it1,end)':>12} {'t':>5}")
    for m in [k, 2 * k, 4 * k, 8 * k]:
        prep = P.prepare(X, k, m)
        for lam in lams:
            t0 = time.time()
            rng = np.random.default_rng(0)
            r0 = P.pgal(X, k, m, lam, 1e-12, dict(NORANK, blockInit=False), prep=prep)
            a0, _ = P.km_read(r0['UU'], Y, k, rng)
            r1 = P.pgal(X, k, m, lam, 1e-12, dict(NORANK, blockInit=True), prep=prep)
            a1, _ = P.km_read(r1['UU'], Y, k, rng)
            r2 = P.pgal(X, k, m, lam, prep=prep, record=True)
            lab = P.comp_labels(r2['Z'])
            sizes = np.bincount(lab)
            a2, n2 = P.acc(Y, lab), P.nmi(Y, lab)
            ka2, _ = P.km_read(r2['UU'], Y, k, rng)
            lock = ari(r2['hist'][0]['labels'], lab)
            print(f"{m:>3} {lam:>6.0e} | {a0:6.4f} {a1:6.4f} | {P.acc(Y, r2['sid']):7.4f} {a2:6.4f} {n2:6.4f} "
                  f"{ka2:6.4f} {len(sizes):>3} {sizes.min():>5} {sizes.max() / len(Y):5.2f} {r2['iter']:>3} "
                  f"{r2['beta']:9.3e} {lock:12.4f} {time.time() - t0:5.1f}", flush=True)
