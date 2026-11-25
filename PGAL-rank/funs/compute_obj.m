function [objVal, objPartVal] = compute_obj(XA, A, W, M, Z_sqr, gamma, lambda, beta, fn1)
%COMPUTE_OBJ  计算 PGAL 的当前目标值
%
%   [objVal, objPartVal] = compute_obj(XA,A,W,M,Z_sqr,gamma,lambda,beta,fn1)
%
%   objPartVal = sum_p gamma_p^2 * sum_ij ||x_i^p - a_j^p||^2 z_ij^2
%                + lambda * sum_p ||W^p A^p - M^p||_F^2
%   objVal     = objPartVal + beta * max(fn1,0),  fn1 = sum_{t<=k} delta_t(L_H)
%
%   向量化实现: XA{p} 是当前 A 下已算好的平方欧氏距离矩阵, 直接与 Z 逐元素加权求和。
%   原实现是 n*m 次 norm(...)^2 调用(每次还要分配临时向量), m 大时每轮多花几十万次调用。
%   原实现是脚本, 会往调用者工作区写 term1/term2/ii/jj, 这里改为函数避免副作用。

term1 = 0;
term2 = 0;
for p = 1:numel(XA)
    term1 = term1 + gamma(p)^2 * sum(sum(XA{p} .* Z_sqr));
    R = W{p} * A{p} - M{p};
    term2 = term2 + sum(sum(R.^2));
end

objPartVal = term1 + lambda * term2;                % 不含秩罚项, 供停机判据使用
objVal     = objPartVal + beta * max(fn1, 0);       % 完整目标
end
