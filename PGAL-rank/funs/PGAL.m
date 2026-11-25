function [A,Z,UU,iter,obj,obj_all,beta] = PGAL(X,k,numanchor,lambda,beta0,verbose,optZ)
%PGAL  Projected Graph Anchor Learning + 秩约束（Z 恰好有 k 个连通分量）
%
%   [A,Z,UU,iter,obj,obj_all,beta] = PGAL(X,k,numanchor,lambda)
%   [A,Z,UU,iter,obj,obj_all,beta] = PGAL(X,k,numanchor,lambda,beta0,verbose,optZ)
%
%   X      : 1 x numview cell, X{p} 为 n x d_p（行是样本）
%   k      : 类别数
%   numanchor : 锚点数 m
%   lambda : ||W^p*A^p - M^p||_F^2 的权重
%   beta0  : 秩约束惩罚参数 beta 的初值；
%            缺省或 <=0 时按数据尺度自动给初值（推荐）
%            给定正数时默认关闭自适应(=固定 beta)，除非 optZ.adaptive 显式指定
%   verbose: 是否打印每次迭代的信息（默认 false）
%   optZ   : 传给 PGAL_solveZ 的选项
%            （innerMax / rankTol / adaptive / initScale / betaRange）
%            .blockInit  是否用块状初值（默认 true）
%            .initLabels n x 1 初值用的样本标签（默认空 = 对拼接特征做 kmeans）
%            （gamma 用 kNN 邻域一致性给初值, 之后按式(12)每轮更新）
%
%   与原始 PGAL 的区别只在 Z 的更新：
%     原: 逐行 min sum_j b_ij z_ij^2, sum_j z_ij = 1  （b_ij = sum_p gamma_p^2*||x_i^p-a_j^p||^2）
%     新: 逐行 min sum_j d_ij z_ij^2 + beta*sum_j t_ij z_ij，并通过 Ky Fan 松弛
%         引入 F（见 PGAL_solveZ），beta 按 L_H 的零特征值个数自适应调整，
%         最终 Z 恰好有 k 个连通分量。
%         注意 t_ij 含 D_u,D_v, 实现中固定上一轮度矩阵做线性化(冻结度矩阵的
%         块交替近似), 不是 Z 子问题的精确解。

%% initialize
maxIter = 50 ; % the number of iterations

if nargin < 5 || isempty(beta0),   beta0   = -1;  end
if nargin < 6 || isempty(verbose), verbose = false; end
if nargin < 7 || isempty(optZ),    optZ    = struct(); end
if ~isfield(optZ, 'blockInit') || isempty(optZ.blockInit)
    optZ.blockInit = true;      % 默认用块状初值（可显式置 false 退回 kNN 初值）
end
if beta0 > 0 && ~isfield(optZ, 'adaptive')
    optZ.adaptive = false;      % 显式给了 beta 就按固定 beta 处理, 避免"固定值却偷偷变"
end

m = numanchor;
numclass = k;
numview = length(X);
numsample = size(X{1},1);

%Calculate Mp
% 视图度量选择: 逐列 z-score 会把"只在少数样本出现的特征"放大成离群点, 平方欧氏
% 距离被它们支配(稀疏 TF-IDF 视图尤其严重, 拼接近乎退化成单一视图的噪声)。
% 若数据含稀疏非负视图(计数/文本类), 则全数据集改用行 L2 归一, 此时欧氏距离
% 等价于余弦距离; 稠密连续数据仍用 z-score + 欧氏。
zfrac = 0;
nonneg = true;
for p = 1 : numview
    zfrac  = max(zfrac, mean(X{p}(:) == 0));
    nonneg = nonneg && (min(X{p}(:)) >= 0);
end
if nonneg && zfrac > 0.2
    preMode = 'l2';
else
    preMode = 'zscore';
end
for p = 1 : numview
    X{p} = normalizeView(X{p}, preMode);
end

% 视图权重 gamma: 用 kNN 邻域一致性给初值, 之后按式(12)每轮更新
if numview > 1
    gamma = viewReliability(X);
else
    gamma = 1;
end

% 锚点索引在"按 sqrt(gamma) 加权的拼接空间"里 kmeans:
% 不可靠视图对锚点位置的拉扯被压小, 避免锚点被噪声视图带到类间。
XX = [];
for p = 1 : numview
    XX = [XX; sqrt(gamma(p)) * X{p}'];
end
rng(12,'twister');
[~,An] = litekmeans(XX',m, 'MaxIter',100,'Replicates',10);

% 参考锚点(先验) M^p: 取该锚点所辖样本在视图 p 自身空间里的均值, 迭代中不再更新。
% 注意 M^p 是"逐视图"的 m x d_p 矩阵, 各视图之间不共享, 因此它只是 A^p 的先验参考,
% 不是跨视图的共识锚点(共识锚点指所有视图共用一个 A, 如 SwMVC-ADF 的做法)。
% 这样 M^p 与 A^p 同尺度, 避免直接反缩放加权坐标造成的尺度错配。
[~, aidx] = min(pdist2(XX', An).^2, [], 2);
for p = 1:numview
    M{p} = zeros(m, size(X{p},2));
    for j = 1:m
        sel = (aidx == j);
        if any(sel)
            M{p}(j,:) = mean(X{p}(sel,:), 1);
        else
            M{p}(j,:) = mean(X{p}, 1);
        end
    end
%Initialize Wp
    W{p} = eye(m);
end

%Initialize Z
if isfield(optZ,'blockInit') && optZ.blockInit
    % 块状初值: 先对样本聚类, 再让锚点-样本同簇相连
    % optZ.initLabels 给了样本标签时用它(例如无约束 PGAL 嵌入上的 kmeans 结果),
    % 否则对拼接特征做 kmeans
    if isfield(optZ,'initLabels') && ~isempty(optZ.initLabels)
        Z = blockInitGraph(XX,An,numclass,optZ.initLabels);
    else
        Z = blockInitGraph(XX,An,numclass);
    end
else
    Z = computeIniGraph(XX,An',numclass);
    Z = Z';
end
Z_sqr = Z.*Z;

%Initialize beta（<=0 表示自动初始化并在迭代中自适应）
beta = beta0;

% 不对 Z 的支撑集做任何额外限制: 这里求解的就是模型本身。
% 注意 m > k 时行解 z ~ 1/d 会让质量偏向远处锚点, 这是模型本身的性质, 不是实现问题。

flag = 1;
iter = 0;
obj = [];
obj_part = [];
obj_all = [];
%%
while flag
    iter = iter + 1;

    %% optimize Ap
    % 逐行闭式解向量化: a_j = (sum_i g^2 z_ij^2 x_i + lambda*(W'M)_j) / (sum_i g^2 z_ij^2 + lambda)
    parfor p=1:numview
        Zg  = gamma(p)^2 * Z_sqr;                    % n x m
        den = sum(Zg, 1)' + lambda;                  % m x 1
        num = Zg' * X{p} + lambda * (W{p}' * M{p});  % m x d
        A{p} = num ./ den;
    end
    
    %% optimize Wp
    parfor p = 1:numview
        pp = A{p}*M{p}';
        [Unew,~,Vnew] = svd(pp','econ');
        W{p} = Unew*Vnew';
    end
    
    %% 各视图加权残差 XA（gamma 固定时只用于构造 Dm）
    sxa = zeros(1, numview);
    parfor p = 1:numview
        XA{p} = pdist2(X{p}, A{p}, 'squaredeuclidean');
        sxa(p) = sum(sum(XA{p} .* Z_sqr));
    end
    gamma = 1 ./ max(sxa, eps);
    gamma = gamma ./ sum(gamma);
    
    %% optimize Z（带秩约束: F/Z 交替 + beta 自适应）
    Dm = zeros(numsample,m);
    for p = 1:numview
        Dm = Dm+gamma(p)^2*XA{p};
    end
    [Z,beta,zinfo] = PGAL_solveZ(Z,Dm,numclass,beta,optZ);
    Z_sqr = Z.*Z;

    [obj(iter), obj_part(iter)] = compute_obj(XA, A, W, M, Z_sqr, gamma, lambda, beta, zinfo.fn1End);
    obj_all(end+1) = obj(iter);
    
    if verbose
        if zinfo.rankOK, tag = '(rank OK)'; else, tag = '(rank FAIL)'; end
        fprintf(['iter %2d: obj=%.6e  beta=%.3e  sum_{t<=k}delta_t=%.2e  ' ...
                 'delta_{k+1}=%.2e  ncomp=%d  minf=%.3f %s\n'], iter, obj(iter), beta, ...
                 zinfo.fn1End, zinfo.fn2End, zinfo.ncomp, zinfo.minFrac, tag);
    end

    % 停机判据用不含罚项的部分: beta 每轮可能跳变, 放进相对变化率会误判
    objStop = (iter>1) && (abs((obj_part(iter-1)-obj_part(iter))/(obj_part(iter-1)))<1e-4 || obj_part(iter) < 1e-10);
    if iter >= maxIter || (objStop && zinfo.rankOK)
        UU = anchorSpectralEmbedding(Z, numclass);
        flag = 0;
    end
end

% 撞到迭代上限时目标可能仍在明显下降, 不能静默当作"已收敛"返回
if iter >= maxIter && ~(objStop && zinfo.rankOK)
    rchg = NaN;
    if numel(obj_part) >= 2 && obj_part(end-1) ~= 0
        rchg = abs((obj_part(end-1)-obj_part(end))/obj_part(end-1));
    end
    fprintf(2, ['[PGAL] 警告: 达到最大迭代次数 %d (目标相对变化 %.2e, 秩条件满足=%d), ' ...
                '结果可能未完全收敛。\n'], maxIter, rchg, zinfo.rankOK);
end

if ~zinfo.rankOK
    fprintf(2, ['[PGAL] 警告: 结束时秩条件未满足 ' ...
                '(sum_{t<=k}delta_t=%.2e, delta_{k+1}=%.2e, ncomp=%d, 最小分量=%d), ' ...
                'Z 未构成 k=%d 个非退化块。\n'], ...
            zinfo.fn1End, zinfo.fn2End, zinfo.ncomp, zinfo.minSize, numclass);
end

end

% ----------------------------------------------------------------------
function Z = blockInitGraph(XX, An, k, sid)
%BLOCKINITGRAPH  块状锚图初值: 样本先聚成 k 类, 锚点归到最近样本所属的类,
%                同类的锚点-样本连边, 使初始 Z 自带 k 块结构。
%   XX : D x n 拼接特征(行是特征);  An : m x D 锚点;  返回 n x m, 每行和为 1
%   sid: 可选, n x 1 的样本簇标签; 不给则对 XX 做 kmeans
    n = size(XX, 2);
    m = size(An, 1);
    if nargin < 4 || isempty(sid)
        rng(12, 'twister');
        sid = litekmeans(XX', k, 'MaxIter', 100, 'Replicates', 10);
    end
    sid = sid(:);
    D2  = sum(An.^2, 2) + sum(XX.^2, 1) - 2 * An * XX;      % m x n
    D2  = max(D2, 0);
    % 锚点 -> 类：用最近 knb 个样本的加权投票（软分配），
    % 而不是单个最近样本的硬 argmax。m > k 时同一类有多个锚点，
    % 个别锚点被噪声样本带走也不会把整块样本挂错类。
    knb = min(5, n);
    [~, ord] = sort(D2, 2);
    vote = eps * ones(m, k);
    for j = 1:m
        nb = ord(j, 1:knb);
        w  = 1 ./ (sqrt(D2(j, nb)) + eps);
        for t = 1:knb
            vote(j, sid(nb(t))) = vote(j, sid(nb(t))) + w(t);
        end
        vote(j, :) = vote(j, :) / sum(vote(j, :));
    end
    % 软边权：样本 i 连到锚点 j 的强度 = 锚点 j 属于 i 所在类的概率
    Z = zeros(n, m);
    for j = 1:m
        Z(:, j) = reshape(vote(j, sid), n, 1);
    end
    Z = Z ./ repmat(sum(Z, 2), 1, m);
end

% ----------------------------------------------------------------------
function F = anchorSpectralEmbedding(Z, k)
%ANCHORSPECTRALEMBEDDING  归一化锚图谱嵌入（读数用）
%   S = Z*Dv^{-1}*Z' 的归一化形式为 D^{-1/2} S D^{-1/2} = E*E',
%   其中 E = Du^{-1/2} Z Dv^{-1/2}。它的前 k 个特征向量就是 E 的前 k 个左奇异向量,
%   所以直接做 SVD 即可, 复杂度 O(n*m*k), 不需要构造 n x n 的 S
%   （原实现是 O(n^2) 内存 + O(n^3) 特征分解, 与锚图的可扩展性矛盾）。
%   注意: Z 有 k 个连通分量时 E*E' 有 k 个单位特征值, 因此取最大的 k 个, 不能跳过第 1 个。
    Du = max(full(sum(Z, 2)), eps);
    Dv = max(full(sum(Z, 1))', eps);
    E  = (Z ./ sqrt(Du)) ./ sqrt(Dv)';
    [F, ~, ~] = mySVD(E, min(k, min(size(E))));
end

% ----------------------------------------------------------------------
function Xp = normalizeView(Xp, mode)
%NORMALIZEVIEW  视图预处理
%   'l2'    : 逐行 L2 归一, 欧氏距离等价于余弦距离
%   'zscore': 逐列 z-score
switch lower(mode)
    case 'l2'
        nr = sqrt(sum(Xp.^2, 2));
        nr(nr < eps) = 1;
        Xp = Xp ./ nr;
    otherwise
        Xp = mapstd(Xp', 0, 1)';
end
end
