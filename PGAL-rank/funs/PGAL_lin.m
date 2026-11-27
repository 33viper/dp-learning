function [A, Z, UU, iter, obj, obj_all, beta] = PGAL_lin(X, k, numanchor, lambda, opts)
%PGAL_LIN  方案 1: 线性数据项 + CAN 自适应 alpha + 先验引导锚点 + 二分图秩约束
%
%   目标函数:
%     min_{W^p, A^p, Z, gamma, F}
%         sum_{p=1}^v gamma_p^2 sum_{i,j} ||x_i^p - a_j^p||_2^2 z_{ij}
%         + sum_{i=1}^n alpha_i sum_{j=1}^m z_{ij}^2
%         + lambda * sum_{p=1}^v ||W^p A^p - M^p||_F^2
%         + beta * Tr(F' L_H F)
%     s.t.  W^{p'} W^p = I_m,  Z >= 0,  Z*1 = 1,  gamma >= 0,  gamma'*1 = 1,  F'F = I_k
%
%   其中 alpha_i 按 CAN (Nie et al., KDD'14) 闭式自适应确定:
%       alpha_i = 0.5 * (K * d_{i, K+1} - sum_{j=1}^K d_{i,j})
%   K 默认取自适应邻域数 min(5, max(2, floor(m/2)))，避免 m=k 时过平滑。
%
%   用法:
%     [A,Z,UU,iter,obj,obj_all,beta] = PGAL_lin(X, k, numanchor, lambda)
%     [A,Z,UU,iter,obj,obj_all,beta] = PGAL_lin(X, k, numanchor, lambda, opts)
%
%   opts 可选字段:
%     .rankMode  'hard' (默认，精确 k 个非退化连通分量)
%                'sth'  (Soft-then-Hard: 先 softSteps 轮温和软秩预对齐，再转精确硬秩)
%                'soft' (纯软秩正则，最后从归一化锚图 E=Z*Dv^{-1/2} 做 SVD 读数)
%                'none' (无秩约束，用于消融对照)
%     .K         CAN 邻域数 (默认 min(5, max(2, floor(m/2))))
%     .softSteps sth 模式软秩预对齐轮数 (默认 5)
%     .softScale soft/sth 模式温和 beta 缩放系数 c (默认 1.0)
%     .maxIter   最大外层迭代次数 (默认 50)
%     .verbose   是否打印迭代日志 (默认 false)
%     .optZ      透传给 PGAL_lin_solveZ 的结构体

    if nargin < 5 || isempty(opts), opts = struct(); end
    if ~isfield(opts, 'rankMode')  || isempty(opts.rankMode),  opts.rankMode  = 'hard'; end
    if ~isfield(opts, 'softSteps') || isempty(opts.softSteps), opts.softSteps = 5; end
    if ~isfield(opts, 'softScale') || isempty(opts.softScale), opts.softScale = 1.0; end
    if ~isfield(opts, 'maxIter')   || isempty(opts.maxIter),   opts.maxIter   = 50; end
    if ~isfield(opts, 'verbose')   || isempty(opts.verbose),   opts.verbose   = false; end
    if ~isfield(opts, 'optZ')      || isempty(opts.optZ),      opts.optZ      = struct(); end

    m = numanchor;
    numclass = k;
    numview = length(X);
    numsample = size(X{1}, 1);

    if isfield(opts, 'K') && ~isempty(opts.K)
        K = max(1, min(opts.K, m - 1));
    else
        K = max(1, min(5, max(2, floor(m / 2))));
        K = min(K, m - 1);
    end

    %% 1. 视图预处理（与 PGAL.m 保持严格一致，确保消融公平）
    zfrac = 0;
    nonneg = true;
    for p = 1:numview
        zfrac  = max(zfrac, mean(X{p}(:) == 0));
        nonneg = nonneg && (min(X{p}(:)) >= 0);
    end
    if nonneg && zfrac > 0.2
        preMode = 'l2';
    else
        preMode = 'zscore';
    end
    for p = 1:numview
        X{p} = normalizeView(X{p}, preMode);
    end

    %% 2. 视图初始权重 gamma 与先验锚点 M^p
    if numview > 1
        gamma = viewReliability(X);
    else
        gamma = 1;
    end

    XX = [];
    for p = 1:numview
        XX = [XX; sqrt(gamma(p)) * X{p}']; %#ok<AGROW>
    end
    rng(12, 'twister');
    [~, An] = litekmeans(XX', m, 'MaxIter', 100, 'Replicates', 10);

    [~, aidx] = min(pdist2(XX', An).^2, [], 2);
    M = cell(1, numview);
    W = cell(1, numview);
    A = cell(1, numview);
    XA = cell(1, numview);
    for p = 1:numview
        M{p} = zeros(m, size(X{p}, 2));
        for j = 1:m
            sel = (aidx == j);
            if any(sel)
                M{p}(j, :) = mean(X{p}(sel, :), 1);
            else
                M{p}(j, :) = mean(X{p}, 1);
            end
        end
        W{p} = eye(m);
    end

    %% 3. 锚图初始化（使用原 PGAL 的子空间重构初值，保留局部流形几何）
    Z = computeIniGraph(XX, An', numclass);
    Z = Z';

    alpha = [];
    beta = -1;
    obj = [];
    obj_part = [];
    obj_all = [];

    %% 4. 可选阶段 A: Soft-then-Hard 的软秩预对齐阶段
    iter = 0;
    if strcmpi(opts.rankMode, 'sth')
        optSoft = opts.optZ;
        optSoft.rankMode  = 'soft';
        optSoft.softScale = opts.softScale;
        for s = 1:opts.softSteps
            iter = iter + 1;
            [A, W, XA, gamma, Dm] = updateAWGammaLinear(X, M, W, Z, gamma, lambda);
            if isempty(alpha)
                alpha = computeCanAlpha(Dm, K);
            end
            [Z, beta, zinfo] = PGAL_lin_solveZ(Z, Dm, numclass, beta, alpha, optSoft);
            [obj(iter), obj_part(iter)] = computeObjLinear(XA, A, W, M, Z, alpha, gamma, lambda, beta, zinfo.fn1End); %#ok<AGROW>
            obj_all(end + 1) = obj(iter); %#ok<AGROW>
        end
    end

    %% 5. 主交替优化循环
    optMain = opts.optZ;
    if strcmpi(opts.rankMode, 'sth')
        optMain.rankMode = 'hard';
    else
        optMain.rankMode = opts.rankMode;
    end
    optMain.softScale = opts.softScale;

    needRankStop = strcmpi(optMain.rankMode, 'hard');
    zinfo = struct('rankOK', true, 'fn1End', 0, 'fn2End', 1, 'ncomp', 1, 'minSize', numsample, 'minFrac', 1);

    for t = 1:opts.maxIter
        iter = iter + 1;

        % 5.1 线性权重下的 A^p, W^p, gamma 精确闭式更新
        [A, W, XA, gamma, Dm] = updateAWGammaLinear(X, M, W, Z, gamma, lambda);

        % 5.2 首轮按 CAN 闭式规则确定行自适应 alpha_i，之后保持固定以保证目标函数统一
        if isempty(alpha)
            alpha = computeCanAlpha(Dm, K);
        end

        % 5.3 更新 Z（以及 F, beta）
        [Z, beta, zinfo] = PGAL_lin_solveZ(Z, Dm, numclass, beta, alpha, optMain);

        % 5.4 记录目标值（不含罚项的 obj_part 用于停机判据）
        [obj(iter), obj_part(iter)] = computeObjLinear(XA, A, W, M, Z, alpha, gamma, lambda, beta, zinfo.fn1End); %#ok<AGROW>
        obj_all(end + 1) = obj(iter); %#ok<AGROW>

        if opts.verbose
            fprintf('iter %2d: obj=%.6e  part=%.6e  beta=%.3e  fn1=%.2e  ncomp=%d  minSz=%d\n', ...
                iter, obj(iter), obj_part(iter), beta, zinfo.fn1End, zinfo.ncomp, zinfo.minSize);
        end

        objStop = (iter > 1) && ...
            (abs(obj_part(iter - 1) - obj_part(iter)) / max(abs(obj_part(iter - 1)), eps) < 1e-4 ...
             || obj_part(iter) < 1e-10);
        if objStop && (~needRankStop || zinfo.rankOK)
            break;
        end
    end

    %% 6. 归一化锚图谱嵌入（供 soft / none 模式或回退读数使用）
    UU = anchorSpectralEmbedding(Z, numclass);
end

% ----------------------------------------------------------------------
function [A, W, XA, gamma, Dm] = updateAWGammaLinear(X, M, W, Z, gamma, lambda)
%UPDATEAWGAMMALINEAR  线性数据项 sum_p gamma_p^2 sum_ij ||x_i^p - a_j^p||^2 z_ij 下
%                     A^p, W^p, gamma 的精确闭式解
    numview = numel(X);
    [n, m]  = size(Z);
    A  = cell(1, numview);
    XA = cell(1, numview);

    % 1) 更新 A^p: a_j^p = (gamma_p^2 sum_i z_ij x_i^p + lambda (W^{p'} M^p)_j) / (gamma_p^2 sum_i z_ij + lambda)
    parfor p = 1:numview
        Zg  = gamma(p)^2 * Z;
        den = sum(Zg, 1)' + lambda;
        num = Zg' * X{p} + lambda * (W{p}' * M{p});
        A{p} = bsxfun(@rdivide, num, den);
    end

    % 2) 更新 W^p: 正交 Procrustes
    parfor p = 1:numview
        pp = A{p} * M{p}';
        [Unew, ~, Vnew] = svd(pp', 'econ');
        W{p} = Unew * Vnew';
    end

    % 3) 更新 gamma: Cauchy-Schwarz 精确闭式解
    sxa = zeros(1, numview);
    parfor p = 1:numview
        XA{p}  = pdist2(X{p}, A{p}, 'squaredeuclidean');
        sxa(p) = sum(sum(XA{p} .* Z));
    end
    gamma = 1 ./ max(sxa, eps);
    gamma = gamma ./ sum(gamma);

    % 4) 组装加权距离矩阵 Dm
    Dm = zeros(n, m);
    for p = 1:numview
        Dm = Dm + gamma(p)^2 * XA{p};
    end
end

% ----------------------------------------------------------------------
function alpha = computeCanAlpha(D, K)
%COMPUTECANALPHA  CAN 自适应行正则系数: beta=0 时每行恰有 K 个非零近邻锚点
    Ds = sort(D, 2, 'ascend');
    alpha = 0.5 * (K * Ds(:, K + 1) - sum(Ds(:, 1:K), 2));
    alpha = max(alpha, 1e-12);
end

% ----------------------------------------------------------------------
function [objVal, objPartVal] = computeObjLinear(XA, A, W, M, Z, alpha, gamma, lambda, beta, fn1)
    term1 = 0;
    term2 = 0;
    for p = 1:numel(XA)
        term1 = term1 + gamma(p)^2 * sum(sum(XA{p} .* Z));
        R = W{p} * A{p} - M{p};
        term2 = term2 + sum(sum(R.^2));
    end
    termReg = sum(alpha .* sum(Z.^2, 2));
    objPartVal = term1 + termReg + lambda * term2;
    objVal     = objPartVal + max(beta, 0) * max(fn1, 0);
end

% ----------------------------------------------------------------------
function F = anchorSpectralEmbedding(Z, k)
    Du = max(full(sum(Z, 2)), eps);
    Dv = max(full(sum(Z, 1))', eps);
    E  = bsxfun(@rdivide, bsxfun(@rdivide, Z, sqrt(Du)), sqrt(Dv)');
    [F, ~, ~] = mySVD(E, min(k, min(size(E))));
end

% ----------------------------------------------------------------------
function Xp = normalizeView(Xp, mode)
    switch lower(mode)
        case 'l2'
            nr = sqrt(sum(Xp.^2, 2));
            nr(nr < eps) = 1;
            Xp = bsxfun(@rdivide, Xp, nr);
        otherwise
            Xp = mapstd(Xp', 0, 1)';
    end
end
