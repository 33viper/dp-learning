"""PGAL-rank 的 Python 移植, 仅用于诊断。
逐步对应 funs/PGAL.m、funs/PGAL_solveZ.m、cmp_rank.m; 随机数与 MATLAB 不同,
数值不会逐位一致, 但每一步的更新公式、beta 搜索逻辑、停机条件都与 MATLAB 相同。
"""
import os
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import h5py
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import normalized_mutual_info_score

EPS = np.finfo(float).eps
DSDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'PGAL-rank', 'datasets')


# ---------------------------------------------------------------- 数据与评价
def load(name):
    path = os.path.join(DSDIR, name + '.mat')
    try:
        S = sio.loadmat(path)
        X = []
        for p in range(S['X'].size):
            xp = S['X'].flat[p]
            X.append(np.asarray(xp.toarray() if sp.issparse(xp) else xp, float))
        Y = S['Y'].ravel().astype(int)
    except NotImplementedError:                 # v7.3(HDF5), 列优先 -> 转置
        with h5py.File(path, 'r') as h:
            X = [np.array(h[r], float).T for r in np.array(h['X']).ravel()]
            Y = np.array(h['Y']).ravel().astype(int)
    return X, Y


def acc(y, p):
    _, yi = np.unique(y, return_inverse=True)
    _, pi = np.unique(p, return_inverse=True)
    C = np.zeros((pi.max() + 1, yi.max() + 1))
    np.add.at(C, (pi, yi), 1)
    r, c = linear_sum_assignment(-C)
    return C[r, c].sum() / len(y)


def nmi(y, p):
    return normalized_mutual_info_score(y, p, average_method='arithmetic')   # 同 compute_nmi.m


def sqdist(A, B):
    return np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T, 0)


def litekmeans(X, k, rng, maxiter=100, reps=10):
    """同 measure/litekmeans.m: 随机样本初值, 空簇用最远样本补, 按距离和选最优重复。"""
    n = X.shape[0]
    x2 = (X ** 2).sum(1)
    best = None
    for _ in range(reps):
        C = X[rng.choice(n, k, replace=False)]
        lab = None
        for _ in range(maxiter):
            Dm = (C ** 2).sum(1)[None, :] - 2 * X @ C.T
            new = Dm.argmin(1)
            miss = np.setdiff1d(np.arange(k), new)
            if miss.size:
                val = x2 + Dm[np.arange(n), new]
                new[np.argsort(-val)[:miss.size]] = miss
            if lab is not None and (new == lab).all():
                break
            lab = new
            C = np.stack([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        s = np.sqrt(sqdist(X, C))[np.arange(n), lab].sum()
        if best is None or s < best[0]:
            best = (s, lab.copy(), C.copy())
    return best[1], best[2]


def km_read(U, Y, k, rng, runs=5):
    """cmp_rank.m 的 kmRead: 行归一化后 kmeans(20 次重复), 5 次取中位数。"""
    Wn = U / np.maximum(np.sqrt((U ** 2).sum(1, keepdims=True)), EPS)
    a, b = [], []
    for _ in range(runs):
        lab, _ = litekmeans(Wn, k, rng, 100, 20)
        a.append(acc(Y, lab))
        b.append(nmi(Y, lab))
    return float(np.median(a)), float(np.median(b))


# ---------------------------------------------------------------- 预处理
def mapstd(X):
    sd = X.std(0, ddof=1)
    sd[sd == 0] = 1
    return (X - X.mean(0)) / sd


def normalize_view(X, mode):
    if mode == 'l2':
        nr = np.sqrt((X ** 2).sum(1))
        nr[nr < EPS] = 1
        return X / nr[:, None]
    return mapstd(X)


def view_reliability(X, kk=10):
    V, n = len(X), X[0].shape[0]
    kk = min(kk, n - 1)
    N = [np.argsort(sqdist(mapstd(x), mapstd(x)), axis=1, kind='stable')[:, 1:kk + 1] for x in X]
    rel = np.zeros(V)
    for p in range(V):
        ov = [(N[p][:, :, None] == N[q][:, None, :]).sum() / (n * kk) for q in range(V) if q != p]
        rel[p] = np.mean(ov)
    rel = np.maximum(rel, 0)
    return rel / rel.sum() if rel.sum() > 0 else np.ones(V) / V


def proj_simplex_rows(V):
    n, m = V.shape
    U = -np.sort(-V, axis=1)
    css = np.cumsum(U, axis=1) - 1
    ind = np.arange(1, m + 1)
    rho = (U - css / ind > 0).sum(1)
    theta = css[np.arange(n), rho - 1] / rho
    return np.maximum(V - theta[:, None], 0)


def ini_graph(XX, An, iters=500):
    """computeIniGraph.m: 逐样本 min ||x_i - An' z||^2, z 在单纯形上(FISTA 代替 quadprog)。"""
    G, B = An @ An.T, XX @ An.T
    L = 2 * np.linalg.eigvalsh(G).max()
    Z = np.full(B.shape, 1.0 / B.shape[1])
    Yv, t = Z.copy(), 1.0
    for _ in range(iters):
        Zn = proj_simplex_rows(Yv - 2 * (Yv @ G - B) / L)
        tn = (1 + np.sqrt(1 + 4 * t * t)) / 2
        Yv, Z, t = Zn + (t - 1) / tn * (Zn - Z), Zn, tn
    return Z


def block_init(XX, An, k, sid=None, seed=12):
    """PGAL.m 的 blockInitGraph。"""
    n, m = XX.shape[0], An.shape[0]
    if sid is None:
        sid, _ = litekmeans(XX, k, np.random.default_rng(seed), 100, 10)
    D2 = sqdist(An, XX)
    nbs = np.argsort(D2, axis=1, kind='stable')[:, :min(5, n)]
    vote = EPS * np.ones((m, k))
    for j in range(m):
        w = 1 / (np.sqrt(D2[j, nbs[j]]) + EPS)
        np.add.at(vote[j], sid[nbs[j]], w)
        vote[j] /= vote[j].sum()
    Z = vote[:, sid].T
    return Z / Z.sum(1, keepdims=True), sid


# ---------------------------------------------------------------- 图与谱
def comp_labels(Z, tol=1e-8):
    """二部图 H=[0 Z; Z' 0] 的样本侧连通分量标签(同 PGAL_Z2labels.m)。"""
    n = Z.shape[0]
    B = sp.csr_matrix(Z > tol)
    _, lab = connected_components(sp.bmat([[None, B], [B.T, None]]), directed=False)
    return np.unique(lab[:n], return_inverse=True)[1]


def comp_sizes(Z, tol=1e-8):
    sizes = np.sort(np.bincount(comp_labels(Z, tol)))[::-1]
    return sizes, len(sizes)


def top_svd(E, r):
    """mySVD: 丢掉 sigma^2/sigma_1^2 < 1e-10 的奇异值, 再截前 r 个。"""
    U, s, Vt = np.linalg.svd(E, full_matrices=False)
    keep = int((s ** 2 >= 1e-10 * s[0] ** 2).sum())
    r = min(r, keep)
    return U[:, :r], s[:r], Vt[:r].T


def solve_f(Z, k):
    """PGAL_solveZ.m 的 solveF: F=[Fu;Fv]=[U;V]/sqrt(2), 取 E=Z Dv^{-1/2} 的前 k 个奇异向量。"""
    Dv = np.maximum(Z.sum(0), EPS)
    U, s, V = top_svd(Z / np.sqrt(Dv)[None, :], min(k + 1, min(Z.shape)))
    kk = min(k, U.shape[1])
    c = np.sqrt(2) / 2
    return np.vstack([c * U[:, :kk], c * V[:, :kk]]), Dv, s


def t_matrix(F, Dv, n):
    """t_ij = ||f_u^i - f_v^j/sqrt(d_j)||^2 (冻结度矩阵)。"""
    Fu, Fvn = F[:n], F[n:] / np.sqrt(Dv)[:, None]
    return np.maximum((Fu ** 2).sum(1)[:, None] + (Fvn ** 2).sum(1)[None, :] - 2 * Fu @ Fvn.T, 0)


def proj_diag(H, Fm, k=1):
    """projSimplexDiagBound: 逐行 min 1/2 x'diag(H)x + F'x, x>=0, 1'x=k, 对 mu 二分。
    H_ij = inf 表示该位置被约束为 0(仅 K 近邻支撑实验用)。"""
    lo, hi = Fm.min(1), np.where(np.isfinite(H), Fm + H, -np.inf).max(1)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        below = (np.maximum(0, mid[:, None] - Fm) / H).sum(1) < k
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    Z = np.maximum(0, (0.5 * (lo + hi))[:, None] - Fm) / H
    return Z * (k / Z.sum(1, keepdims=True))


def anchor_embedding(Z, k):
    Du, Dv = np.maximum(Z.sum(1), EPS), np.maximum(Z.sum(0), EPS)
    U, _, _ = top_svd(Z / np.sqrt(Du)[:, None] / np.sqrt(Dv)[None, :], min(k, min(Z.shape)))
    return U


def spectral_info(Z, k):
    _, _, s = solve_f(Z, k)
    sk = np.concatenate([s, np.zeros(k)])
    fn1 = k - sk[:k].sum()
    fn2 = 1 - s[k] if len(s) >= k + 1 else 1.0
    return fn1, fn2


# ---------------------------------------------------------------- Z 子问题 (PGAL_solveZ.m)
ZDEF = dict(innerMax=40, rankTol=1e-8, adaptive=True, initScale=1e-3, betaRange=(1e-12, 1e12),
            rho=2.0, minComp=2, dataTol=1e-6, trace=None)


def solve_z(Z, D, k, beta, opts=None):
    o = dict(ZDEF, **(opts or {}))
    n, m = D.shape
    Dsafe = np.maximum(D, 1e-12)
    if o.get('mask') is not None:                 # K 近邻支撑(模型改动实验), 原代码无此项
        Dsafe = np.where(o['mask'], Dsafe, np.inf)
    need_init = not (beta > 0)
    F, Dv, _ = solve_f(Z, k)
    beta_used, beta_lo, beta_hi = beta, None, None
    best, last = None, None
    prev_data = np.inf
    for t in range(o['innerMax']):
        T = t_matrix(F, Dv, n)
        if need_init:
            lam0 = np.median(1 / (1 / Dsafe).sum(1))
            beta = o['initScale'] * 2 * lam0 / max(T.mean(), EPS)
            need_init = False
        Zold = Z
        Z = proj_diag(2 * Dsafe, beta * T, 1)
        FN, DvN, sig = solve_f(Z, k)
        sk = np.concatenate([sig, np.zeros(k)])
        fn1 = k - sk[:k].sum()
        sizes, ncomp = comp_sizes(Z)
        rank_ok = (ncomp == k) and (fn1 <= o['rankTol'])
        data_now = np.where(Z > 0, Dsafe * Z ** 2, 0).sum()
        deg_free = rank_ok and sizes.min() >= o['minComp']
        if o['trace'] is not None:
            o['trace'].append(dict(beta=beta, ncomp=ncomp, minsize=int(sizes.min()), data=data_now,
                                   labels=comp_labels(Z)))
        if not o['adaptive']:
            F, Dv, beta_used = FN, DvN, beta
            continue
        if deg_free:
            if best is None or data_now < best[2]:
                best = (Z, beta, data_now)
            F, Dv, beta_used = FN, DvN, beta
            if data_now < prev_data * (1 - o['dataTol']):
                prev_data = data_now
                continue
            break
        elif rank_ok:
            if last is None or data_now < last[2]:
                last = (Z, beta, data_now)
            beta_hi, Z = beta, Zold
        elif ncomp > k:
            beta_hi, Z = beta, Zold
        else:
            beta_lo = beta
            F, Dv, beta_used = FN, DvN, beta
        if beta_lo is None or beta_hi is None:
            beta_new = beta * o['rho'] if beta_hi is None else beta / o['rho']
        else:
            beta_new = np.sqrt(beta_lo * beta_hi)
        beta_new = min(max(beta_new, o['betaRange'][0]), o['betaRange'][1])
        if beta_new == beta:
            break
        beta, prev_data = beta_new, np.inf
    if best is not None:
        Z, beta = best[0], best[1]
    elif last is not None:
        if comp_sizes(Z)[1] != k:
            Z, beta = last[0], last[1]
        else:
            beta = beta_used
    else:
        beta = beta_used
    fn1, fn2 = spectral_info(Z, k)
    sizes, ncomp = comp_sizes(Z)
    info = dict(fn1End=fn1, fn2End=fn2, ncomp=ncomp, minSize=int(sizes.min()), minFrac=sizes.min() / n,
                rankOK=(ncomp == k) and fn1 <= o['rankTol'] and sizes.min() >= o['minComp'])
    return Z, beta, info


# ---------------------------------------------------------------- 主流程 (PGAL.m)
def prepare(X, k, m, seed=12):
    """PGAL.m 的初始化部分: 预处理、gamma 初值、锚点 kmeans、M^p(与 lambda 无关, 可复用)。"""
    V = len(X)
    zfrac = max(float(np.mean(x == 0)) for x in X)
    nonneg = all(x.min() >= 0 for x in X)
    mode = 'l2' if (nonneg and zfrac > 0.2) else 'zscore'
    X = [normalize_view(x, mode) for x in X]
    gamma = view_reliability(X) if V > 1 else np.ones(1)
    XX = np.hstack([np.sqrt(gamma[p]) * X[p] for p in range(V)])
    _, An = litekmeans(XX, m, np.random.default_rng(seed), 100, 10)
    aidx = sqdist(XX, An).argmin(1)
    M = []
    for p in range(V):
        Mp = np.tile(X[p].mean(0), (m, 1))
        for j in range(m):
            if (aidx == j).any():
                Mp[j] = X[p][aidx == j].mean(0)
        M.append(Mp)
    return dict(X=X, gamma=gamma, XX=XX, An=An, M=M, mode=mode)


def update_AW_gamma(X, M, W, Zs, gamma, lam):
    """A 闭式解 -> W 正交 Procrustes -> gamma 闭式解, 与 PGAL.m 顺序一致。"""
    V = len(X)
    A = []
    for p in range(V):
        Zg = gamma[p] ** 2 * Zs
        A.append((Zg.T @ X[p] + lam * (W[p].T @ M[p])) / (Zg.sum(0) + lam)[:, None])
    for p in range(V):
        U, _, Vt = np.linalg.svd((A[p] @ M[p].T).T)
        W[p] = U @ Vt
    XA = [sqdist(X[p], A[p]) for p in range(V)]
    sxa = np.array([(XA[p] * Zs).sum() for p in range(V)])
    gamma = 1 / np.maximum(sxa, EPS)
    return A, W, XA, gamma / gamma.sum()


def pgal(X, k, m, lam, beta0=-1.0, optZ=None, max_iter=50, seed=12, prep=None, Z0=None,
         record=False, zsolver=None):
    optZ = dict(optZ or {})
    block = optZ.pop('blockInit', True)
    init_labels = optZ.pop('initLabels', None)
    knn = optZ.pop('knn', None)                   # 模型改动实验: 每个样本只连 K 个最近锚点
    if beta0 > 0 and 'adaptive' not in optZ:
        optZ['adaptive'] = False
    P = prep or prepare(X, k, m, seed)
    X, M, gamma = P['X'], P['M'], P['gamma'].copy()
    V = len(X)
    W = [np.eye(m) for _ in range(V)]
    sid = None
    if Z0 is not None:
        Z = Z0.copy()
    elif block:
        Z, sid = block_init(P['XX'], P['An'], k, init_labels, seed)
    else:
        Z = ini_graph(P['XX'], P['An'])
    Zs, beta = Z ** 2, beta0
    hist, obj_part = [], []
    zsolver = zsolver or solve_z
    for it in range(1, max_iter + 1):
        A, W, XA, gamma = update_AW_gamma(X, M, W, Zs, gamma, lam)
        Dm = sum(gamma[p] ** 2 * XA[p] for p in range(V))
        if knn:
            mask = np.zeros(Dm.shape, bool)
            np.put_along_axis(mask, np.argsort(Dm, axis=1)[:, :knn], True, axis=1)
            optZ['mask'] = mask
        Z, beta, zi = zsolver(Z, Dm, k, beta, optZ)
        Zs = Z ** 2
        op = sum(gamma[p] ** 2 * (XA[p] * Zs).sum() for p in range(V)) \
            + lam * sum(((W[p] @ A[p] - M[p]) ** 2).sum() for p in range(V))
        obj_part.append(op)
        if record:
            hist.append(dict(labels=comp_labels(Z), beta=beta, ncomp=zi['ncomp'], obj=op))
        stop = it > 1 and (abs((obj_part[-2] - op) / obj_part[-2]) < 1e-4 or op < 1e-10)
        if it >= max_iter or (stop and zi['rankOK']):
            break
    return dict(Z=Z, UU=anchor_embedding(Z, k), iter=it, beta=beta, zinfo=zi, hist=hist, sid=sid,
                Dm=Dm, A=A, W=W, gamma=gamma, obj_part=obj_part)
