"""机制诊断: 在当前代码得到的最终解上验证
 (a) 恰好 k 个分量时 t_ij 只取 {0, (1/n_c+1/n_c')/2} 两类值, 不含任何数据距离信息
 (b) z_i 的支撑集 = 所在分量的全部锚点(支撑由 t 决定, 与 d_ij 的排序无关)
 (c) 数据上最近的锚点落在别的分量里的样本比例
 (d) 固定 A/gamma(即固定 d_ij)时, 不同划分下数据项的最优值:
     单分量(无秩) / 代码给出的划分 / 真实标签划分 / "切 k-1 个单样本"的退化划分
"""
import sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import pgal_port as P


def bip_labels(Z, tol=1e-8):
    n = Z.shape[0]
    B = sp.csr_matrix(Z > tol)
    _, lab = connected_components(sp.bmat([[None, B], [B.T, None]]), directed=False)
    return lab[:n], lab[n:]


def block_cost(inv, slab, alab):
    """样本划分 slab、锚点划分 alab 下, 块内最优 z_ij ∝ 1/d_ij 时的数据项 sum_i 1/sum_{j in c(i)} 1/d_ij。"""
    s = (inv * (slab[:, None] == alab[None, :])).sum(1)
    return np.inf if (s <= 0).any() else float((1 / s).sum())


def best_anchor_assign(inv, slab, k, sweeps=3):
    """样本划分固定(如真实标签), 用坐标下降给锚点找归属, 使数据项最小。"""
    m = inv.shape[1]
    score = np.stack([inv[slab == c].sum(0) for c in range(k)], 1)
    alab = score.argmax(1)
    for c in range(k):                      # 每类至少一个锚点
        if not (alab == c).any():
            alab[score[:, c].argmax()] = c
    cur = block_cost(inv, slab, alab)
    for _ in range(sweeps):
        for j in range(m):
            for c in range(k):
                if c == alab[j]:
                    continue
                old = alab[j]
                alab[j] = c
                new = block_cost(inv, slab, alab)
                if new < cur:
                    cur = new
                else:
                    alab[j] = old
    return alab, cur


def degenerate(inv, Dm, k):
    """贪心选 k-1 个锚点, 每个只带它最近的 1 个样本自成一个分量, 其余全在一个大分量里。"""
    n, m = inv.shape
    S = inv.sum(1)
    delta = []
    for j in range(m):
        i = int(Dm[:, j].argmin())
        others = np.ones(n, bool)
        others[i] = False
        d = Dm[i, j] - 1 / S[i] + (1 / (S[others] - inv[others, j]) - 1 / S[others]).sum()
        delta.append((d, j, i))
    delta.sort()
    slab, alab = np.zeros(n, int), np.zeros(m, int)
    used, c = set(), 1
    for _, j, i in delta:
        if i in used:
            continue
        slab[i], alab[j] = c, c
        used.add(i)
        c += 1
        if c == k:
            break
    return slab, alab, block_cost(inv, slab, alab)


name = sys.argv[1] if len(sys.argv) > 1 else 'ForestTypes'
cfgs = [(int(a), float(b)) for a, b in (s.split(':') for s in sys.argv[2:])] or [(16, 1e-2)]
X, Y = P.load(name)
Y0 = np.unique(Y, return_inverse=True)[1]
k = Y0.max() + 1
for m, lam in cfgs:
    prep = P.prepare(X, k, m)
    r = P.pgal(X, k, m, lam, prep=prep, record=True)
    Z, Dm = r['Z'], r['Dm']
    n = Z.shape[0]
    raw_s, raw_a = bip_labels(Z)
    comp_ids, slab = np.unique(raw_s, return_inverse=True)      # 样本侧分量编号 0..nc-1
    lut = {c: t for t, c in enumerate(comp_ids)}
    alab = np.array([lut.get(a, -1) for a in raw_a])             # 孤立锚点记 -1
    sizes = np.bincount(slab)
    print(f'\n### {name} m={m} lam={lam:g}: ACC={P.acc(Y, slab):.4f} sizes={sizes.tolist()} iter={r["iter"]}')

    F, Dv, _ = P.solve_f(Z, k)
    T = P.t_matrix(F, Dv, n)
    same = slab[:, None] == alab[None, :]
    cross = ~same & (alab[None, :] >= 0)
    pred = 0.5 * (1 / sizes[slab][:, None] + 1 / sizes[np.maximum(alab, 0)][None, :])
    print(f'(a) 分量内 max t_ij = {T[same].max():.2e};  跨分量 |t_ij - (1/n_c+1/n_c\')/2| 最大偏差 = '
          f'{np.abs(T[cross] - pred[cross]).max():.2e}')
    supp_eq = ((Z > 1e-8) == same).all(1).mean()
    print(f'(b) 支撑集恰为"本分量全部锚点"的样本比例 = {supp_eq:.4f}')
    nearest = Dm.argmin(1)
    print(f'(c) 数据上最近锚点不在本分量的样本比例 = {(alab[nearest] != slab).mean():.4f}')

    inv = 1 / np.maximum(Dm, 1e-12)
    J_all = float((1 / inv.sum(1)).sum())
    J_code = block_cost(inv, slab, alab)
    _, J_true = best_anchor_assign(inv, Y0, k)
    ds, da, J_deg = degenerate(inv, Dm, k)
    print(f'(d) 固定 d_ij 时数据项最优值:  单分量(无秩下界) {J_all:.4e} | 退化划分 {J_deg:.4e} '
          f'(ACC={P.acc(Y, ds):.3f}) | 代码划分 {J_code:.4e} | 真实标签划分 {J_true:.4e}')
