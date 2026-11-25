function [x,ft] = EProjSimplex_new_ZJP_V2(H_i,F_i, k)
%
%% Problem
%
%  min  1/2 x'diag(H)x + F'x
%  s.t. x>=0, 1'x=k
%  *** Input column vectors H_i,F_i ***
%  Call x = EProjSimplex_new_ZJP_V2(H_i,F_i)
%
%  KKT: x_j = max(0, (tau-F_j)/H_j), 由 sum(x)=k 解 tau。
%  尺度悬殊（|F| >> H，例如秩惩罚项很大时）Newton 迭代可能走不完,
%  最后 max(v1,0) 会给出一整行 0（违反 1'x=k），这里用顶点解兜底。

if nargin < 3
    k = 1;
end

ft=1;
n = length(F_i);
H_i = max(H_i(:), eps);
F_i = F_i(:);

v0 = -F_i./H_i;
f = 1;
lambda_m = 0;
while abs(f) > 10^-10 && ft <= 100
    v1 = v0 - lambda_m./H_i;
    posidx = v1>0;
    f = sum(v1(posidx)) - k;
    d = sum(1./H_i(posidx));
    if d==0
        lambda_m = lambda_m - 100;
    else
        lambda_m = lambda_m + f/d;
    end
    ft=ft+1;
end
x = max(v1,0);

% ---------- 数值兜底 ----------
% 分段二次问题的最优解一定不差于任一顶点解 e_j (代价 1/2*H_j+F_j)。
[cmin, jv] = min(0.5*H_i + F_i);
bad = ~all(isfinite(x)) || sum(x) <= 0.5*k;
if ~bad
    phi = 0.5*sum(H_i.*x.^2) + F_i'*x;
    bad = phi > cmin + 1e-12*max(1,abs(cmin));
end
if bad
    x = zeros(n,1); x(jv) = k;
else
    x = x*(k/sum(x));        % 消掉 Newton 的微小漂移
end
