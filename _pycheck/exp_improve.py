"""优化过程 / 模型改动对比 (同一套 prepare() 初始化, 只改 Z 的处理方式)。
 noR      : 无秩约束 + QP 初值, 谱嵌入 kmeans 读标签             (= cmp_rank 的 ACC0)
 R        : 当前代码(自适应 beta, 连通分量读标签)                 (= cmp_rank 的 ACC2)
 fixB(c)  : 按图中公式把 beta 当固定权重 beta = c*beta_scale, 每轮 A/W/gamma/F/Z 各更新一次,
            谱嵌入 kmeans 读标签 (c=1e-3 即代码里 beta 的初值, 几乎等于无秩)
 cont     : beta 跨外层迭代延拓: 每轮只做一次 F/Z, 分量不足 beta*1.5, 超过 k 则回退并 /1.5,
            达到 k 个非退化分量后固定 beta, 连通分量读标签
 knn-noR / knn-R : 模型改动, 每个样本只允许连 K=5 个最近锚点, 其余同 noR / R
"""
import sys
import numpy as np
import pgal_port as P

NORANK = dict(adaptive=False, innerMax=1)


def make_continuation(rho=1.5, init_scale=1e-3, min_comp=2):
    st = dict(beta=None)

    def zsolver(Z, D, k, beta, opts):
        n = D.shape[0]
        Dsafe = np.maximum(D, 1e-12)
        F, Dv, _ = P.solve_f(Z, k)
        T = P.t_matrix(F, Dv, n)
        if st['beta'] is None:
            st['beta'] = init_scale * 2 * np.median(1 / (1 / Dsafe).sum(1)) / max(T.mean(), P.EPS)
        b = st['beta']
        Zn = P.proj_diag(2 * Dsafe, b * T, 1)
        sizes, nc = P.comp_sizes(Zn)
        if nc > k or (nc == k and sizes.min() < min_comp):
            st['beta'], Zn = b / rho, Z
        elif nc < k:
            st['beta'] = b * rho
        sizes, nc = P.comp_sizes(Zn)
        ok = nc == k and sizes.min() >= min_comp
        return Zn, b, dict(ncomp=nc, minSize=int(sizes.min()), rankOK=ok)
    return zsolver


def readout(r, Y, k, rng):
    lab = P.comp_labels(r['Z'])
    nc = lab.max() + 1
    if nc == k:
        a, b = P.acc(Y, lab), P.nmi(Y, lab)
        mf = np.bincount(lab).max() / len(Y)
    else:
        a, b = P.km_read(r['UU'], Y, k, rng)
        mf = np.nan
    return a, b, nc, mf


names = sys.argv[1:] or ['ForestTypes', 'Wikipedia']
cs = [1e-3, 1e-2, 1e-1, 1, 10]
for name in names:
    X, Y = P.load(name)
    k = len(np.unique(Y))
    print(f'\n=== {name} n={len(Y)} k={k} ===   (每格: ACC/NMI/分量数)', flush=True)
    head = ['noR', 'R'] + [f'fixB({c:g})' for c in cs] + ['cont', 'knn-noR', 'knn-R']
    print(f"{'m':>3} {'lam':>6} | " + ' | '.join(f'{h:^17}' for h in head))
    for m in [k, 2 * k, 4 * k, 8 * k]:
        prep = P.prepare(X, k, m)
        for lam in [1e-2, 1e2]:
            rng = np.random.default_rng(0)
            runs = [P.pgal(X, k, m, lam, 1e-12, dict(NORANK, blockInit=False), prep=prep),
                    P.pgal(X, k, m, lam, prep=prep)]
            runs += [P.pgal(X, k, m, lam, -1, dict(NORANK, initScale=c), prep=prep) for c in cs]
            runs.append(P.pgal(X, k, m, lam, prep=prep, max_iter=100, zsolver=make_continuation()))
            runs.append(P.pgal(X, k, m, lam, 1e-12, dict(NORANK, blockInit=False, knn=5), prep=prep))
            runs.append(P.pgal(X, k, m, lam, optZ=dict(knn=5), prep=prep))
            cells = []
            for r in runs:
                a, b, nc, _ = readout(r, Y, k, rng)
                cells.append(f'{a:.3f}/{b:.3f}/{nc:<3d}')
            print(f'{m:>3} {lam:>6.0e} | ' + ' | '.join(f'{c:^17}' for c in cells), flush=True)
