"""实现核对: 在 ForestTypes (m=8, lambda=1e-2) 上逐块检查各更新是否是对应子问题的最优解。
取一个"还没有 k 个分量"的中间状态(无秩约束跑 2 轮后的稠密 Z), 避免 Tr(F'LF)=0 时检查变成平凡等式。"""
import numpy as np
from scipy.optimize import minimize
import pgal_port as P

rng = np.random.default_rng(1)
X, Y = P.load('ForestTypes')
k, m, lam = 4, 8, 1e-2
prep = P.prepare(X, k, m)
Xv, M, V = prep['X'], prep['M'], len(prep['X'])
r = P.pgal(X, k, m, lam, 1e-12, dict(adaptive=False, innerMax=1, blockInit=False), prep=prep, max_iter=2)
Z, W, gamma = r['Z'], r['W'], r['gamma']
Zs = Z ** 2
n = Z.shape[0]
print(f'中间状态: Z 的连通分量数 = {P.comp_sizes(Z)[1]}, 非零比例 = {(Z > 1e-8).mean():.3f}')


def obj_AW(A, W, gamma):
    return sum(gamma[p] ** 2 * (P.sqdist(Xv[p], A[p]) * Zs).sum()
               + lam * ((W[p] @ A[p] - M[p]) ** 2).sum() for p in range(V))


# (1) A: 闭式解处梯度应为 0
A, W2, XA, g2 = P.update_AW_gamma(Xv, M, [w.copy() for w in W], Zs, gamma, lam)
Aa = [(gamma[p] ** 2 * Zs.T @ Xv[p] + lam * W[p].T @ M[p]) / (gamma[p] ** 2 * Zs.sum(0) + lam)[:, None]
      for p in range(V)]
grad = [2 * gamma[p] ** 2 * (Zs.sum(0)[:, None] * Aa[p] - Zs.T @ Xv[p]) + 2 * lam * W[p].T @ (W[p] @ Aa[p] - M[p])
        for p in range(V)]
print(f'(1) A 闭式解处 ||grad||_F / ||A||_F = {max(np.linalg.norm(g) / np.linalg.norm(a) for g, a in zip(grad, Aa)):.2e}')

# (2) W: Procrustes 解应不差于任何随机正交阵
f_star = sum(((W2[p] @ A[p] - M[p]) ** 2).sum() for p in range(V))
f_rand = min(sum(((np.linalg.qr(rng.standard_normal((m, m)))[0] @ A[p] - M[p]) ** 2).sum() for p in range(V))
             for _ in range(2000))
print(f'(2) W: 正交性误差 {max(np.abs(w.T @ w - np.eye(m)).max() for w in W2):.1e}; '
      f'Procrustes 值 {f_star:.4f} <= 2000 个随机正交阵最小值 {f_rand:.4f}')

# (3) gamma: 闭式解应不差于单纯形上的随机点
h = np.array([(XA[p] * Zs).sum() for p in range(V)])
g_rand = rng.dirichlet(np.ones(V), 20000)
print(f'(3) gamma: 闭式解值 {(g2 ** 2 * h).sum():.6f} <= 随机单纯形点最小值 {((g_rand ** 2) * h).sum(1).min():.6f}')

# (4) F: 显式构造 L_H, Tr(F'LF) 应等于 L_H 最小 k 个特征值之和, 且等于 sum z_ij t_ij
F, Dv, s = P.solve_f(Z, k)
Hm = np.block([[np.zeros((n, n)), Z], [Z.T, np.zeros((m, m))]])
d = Hm.sum(1)
L = np.eye(n + m) - Hm / np.sqrt(np.outer(d, d))
ev = np.linalg.eigvalsh(L)
T = P.t_matrix(F, Dv, n)
print(f'(4) F: Tr(F\'L_H F) = {np.trace(F.T @ L @ F):.10f}, 最小 k 个特征值之和 = {ev[:k].sum():.10f}, '
      f'sum z_ij t_ij = {(Z * T).sum():.10f}, F\'F 误差 = {np.abs(F.T @ F - np.eye(k)).max():.1e}')

# (5) Z: 二分解应是 (冻结度矩阵的) 行子问题的最优解, 与通用求解器对比
D = sum(g2[p] ** 2 * XA[p] for p in range(V))
beta = 2 * np.median(1 / (1 / D).sum(1)) / T.mean()     # 秩项与数据项同量级
Zb = P.proj_diag(2 * D, beta * T, 1)
gap = 0
for i in rng.choice(n, 20, replace=False):
    f = lambda z: (D[i] * z ** 2).sum() + beta * (T[i] * z).sum()
    res = minimize(f, np.full(m, 1 / m), method='SLSQP', bounds=[(0, 1)] * m,
                   constraints=[dict(type='eq', fun=lambda z: z.sum() - 1)], options=dict(ftol=1e-14, maxiter=500))
    gap = max(gap, f(Zb[i]) - res.fun)
print(f'(5) Z: 二分解目标值 - SLSQP 目标值 的最大值(20 行) = {gap:.2e} (<=0 表示不差于通用求解器)')

# (6) 冻结度矩阵带来的误差: 新 Z 用旧 D_v 算的秩项 vs 用新 Z 自身度矩阵算的真实秩项
FZ = np.block([[np.zeros((n, n)), Zb], [Zb.T, np.zeros((m, m))]])
dz = FZ.sum(1)
Lz = np.eye(n + m) - FZ / np.sqrt(np.outer(dz, dz))
print(f'(6) 冻结度矩阵: 线性化秩项 sum z_new*t_old = {(Zb * T).sum():.6f}, '
      f'真实 Tr(F_old\' L_H(Z_new) F_old) = {np.trace(F.T @ Lz @ F):.6f}')
