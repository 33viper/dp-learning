"""对照实验: 只把数据项从 sum d_ij z_ij^2 换成 CAN/SOBG 式的 sum d_ij z_ij + alpha_i z_ij^2,
其余(预处理、锚点、M^p、W^p、gamma、F、beta 自适应、连通分量读标签)全部与当前代码相同。
用来判断"秩约束不起作用"是优化问题还是数据项形式的问题。
 lin-noR : beta=0, Z 为 K 近邻自适应锚图, 谱嵌入 kmeans 读标签
 lin-R   : 自适应 beta 强制 k 个分量, 连通分量读标签 (块状初值, 同当前代码)
 lin-R/qp: 同上, 但用 QP 初值(检验对初值是否敏感)
alpha_i 取 CAN 的闭式值(第一次外层迭代时按 K=5 计算后固定), 使 beta=0 时每行恰好 K 个非零。
"""
import sys
import numpy as np
import pgal_port as P


def can_alpha(D, K):
    Ds = np.sort(D, axis=1)
    return np.maximum(0.5 * (K * Ds[:, K] - Ds[:, :K].sum(1)), 1e-12)


def solve_z_lin(Z, D, k, beta, alpha, o=P.ZDEF):
    """与 PGAL_solveZ 相同的 beta 搜索, 只把行子问题换成 min sum (d+beta*t) z + alpha z^2。"""
    n, m = D.shape
    F, Dv, _ = P.solve_f(Z, k)
    if not beta > 0:
        beta = o['initScale'] * D.mean() / max(P.t_matrix(F, Dv, n).mean(), P.EPS)
    lo = hi = best = None
    prev = np.inf
    for _ in range(o['innerMax']):
        T = P.t_matrix(F, Dv, n)
        Zold = Z
        Z = P.proj_simplex_rows(-(D + beta * T) / (2 * alpha[:, None]))
        FN, DvN, sig = P.solve_f(Z, k)
        sizes, nc = P.comp_sizes(Z)
        data = (D * Z).sum() + (alpha[:, None] * Z ** 2).sum()
        if nc == k and sizes.min() >= o['minComp']:
            if best is None or data < best[2]:
                best = (Z, beta, data)
            F, Dv = FN, DvN
            if data < prev * (1 - o['dataTol']):
                prev = data
                continue
            break
        if nc >= k:                               # 分量过多或只靠单点凑数: beta 偏大, 回滚
            hi, Z = beta, Zold
        else:
            lo, F, Dv = beta, FN, DvN
        beta = beta * 2 if hi is None else (beta / 2 if lo is None else np.sqrt(lo * hi))
        prev = np.inf
    if best is not None:
        Z, beta = best[0], best[1]
    sizes, nc = P.comp_sizes(Z)
    return Z, beta, dict(ncomp=nc, rankOK=nc == k and sizes.min() >= o['minComp'])


def pgal_lin(X, k, m, lam, prep, rank, block, K=5, max_iter=50, seed=12):
    Xv, M, gamma = prep['X'], prep['M'], prep['gamma'].copy()
    V = len(Xv)
    W = [np.eye(m) for _ in range(V)]
    Z = P.block_init(prep['XX'], prep['An'], k, None, seed)[0] if block else P.ini_graph(prep['XX'], prep['An'])
    K = min(K, m - 1)
    alpha, beta, objs = None, -1.0, []
    for it in range(1, max_iter + 1):
        A, W, XA, gamma = P.update_AW_gamma(Xv, M, W, Z, gamma, lam)     # 线性权重 z_ij
        Dm = sum(gamma[p] ** 2 * XA[p] for p in range(V))
        if alpha is None:
            alpha = can_alpha(Dm, K)
        if rank:
            Z, beta, zi = solve_z_lin(Z, Dm, k, beta, alpha)
        else:
            Z, zi = P.proj_simplex_rows(-Dm / (2 * alpha[:, None])), dict(rankOK=True)
        objs.append((Dm * Z).sum() + (alpha[:, None] * Z ** 2).sum()
                    + lam * sum(((W[p] @ A[p] - M[p]) ** 2).sum() for p in range(V)))
        if it > 1 and abs(objs[-2] - objs[-1]) / objs[-2] < 1e-4 and zi['rankOK']:
            break
    return dict(Z=Z, UU=P.anchor_embedding(Z, k))


def readout(r, Y, k, rng):
    lab = P.comp_labels(r['Z'])
    if lab.max() + 1 == k:
        return P.acc(Y, lab), P.nmi(Y, lab), k
    a, b = P.km_read(r['UU'], Y, k, rng)
    return a, b, lab.max() + 1


names = sys.argv[1:] or ['ForestTypes', 'Wikipedia']
for name in names:
    X, Y = P.load(name)
    k = len(np.unique(Y))
    print(f'\n=== {name} n={len(Y)} k={k} ===   (每格: ACC/NMI/分量数)', flush=True)
    head = ['lin-noR', 'lin-R', 'lin-R/qp']
    print(f"{'m':>3} {'lam':>6} | " + ' | '.join(f'{h:^17}' for h in head))
    for m in [k, 2 * k, 4 * k, 8 * k]:
        prep = P.prepare(X, k, m)
        for lam in [1e-2, 1e2]:
            rng = np.random.default_rng(0)
            runs = [pgal_lin(X, k, m, lam, prep, rank=False, block=False),
                    pgal_lin(X, k, m, lam, prep, rank=True, block=True),
                    pgal_lin(X, k, m, lam, prep, rank=True, block=False)]
            cells = ['{:.3f}/{:.3f}/{:<3d}'.format(*readout(r, Y, k, rng)) for r in runs]
            print(f'{m:>3} {lam:>6.0e} | ' + ' | '.join(f'{c:^17}' for c in cells), flush=True)
