function [Z, beta, info] = PGAL_lin_solveZ(Z, D, k, beta, alpha, opts)
%PGAL_LIN_SOLVEZ  方案 1（线性数据项 + 自适应 alpha + 秩约束）的 Z 子问题求解器
%
%   求解：
%     min_{Z>=0, Z*1=1, F'F=I_k}
%         sum_{i,j} d_{ij} z_{ij} + sum_i alpha_i sum_j z_{ij}^2 + beta * Tr(F' L_H F)
%
%   其中辅助二分图 H = [0 Z; Z' 0]，L_H = I - D^{-1/2} H D^{-1/2}。
%   与原 PGAL_solveZ 的核心区别：
%     1) 数据项由 d_ij z_ij^2 改为 CAN 式的线性距离 + 行自适应 L2 正则 (d_ij z_ij + alpha_i z_ij^2)，
%        使 d_ij 进入行闭式解的分子，与 beta*t_ij 同场竞争，从根源消除“远处锚点零边际代价 -> 退化切单点”。
%     2) 行子问题化为标准欧氏单纯形投影：
%            z_i = argmin_{z>=0, 1'z=1} 0.5 * || z + (d_i + beta*t_i) / (2*alpha_i) ||_2^2
%        采用 Duchi et al. (2008) 排序精确投影，O(n*m*log m)，无步长/数值发散问题。
%     3) 支持三种模式（通过 opts.rankMode）：
%        - 'hard' (默认): 自适应调整 beta，直至 Z 恰好具有 k 个非退化连通分量 (minSize >= minComp)
%        - 'soft'       : 固定温和 beta 做 1 步 F/Z 更新（用于软秩正则或 Soft-then-Hard 第一阶段）
%        - 'none'       : beta = 0，直接解无秩约束的 CAN-PGAL 锚图
%
%   输入：
%     Z     : n x m 当前锚图（热启动）
%     D     : n x m 加权平方距离矩阵, d_ij = sum_p gamma_p^2 * ||x_i^p - a_j^p||^2
%     k     : 类别数
%     beta  : 秩罚项系数（<=0 时自动按数据尺度初始化）
%     alpha : n x 1 行自适应正则系数（CAN 闭式确定）
%     opts  : 可选结构体
%             .rankMode   'hard' | 'soft' | 'none' (默认 'hard')
%             .innerMax   F/Z 交替最大步数         (默认 40)
%             .rankTol    Ky Fan 零特征值判据阈值   (默认 1e-8)
%             .initScale  hard 模式自动 beta 初值缩放 (默认 1e-3)
%             .softScale  soft 模式自动 beta 系数 c (默认 1.0)
%             .betaRange  beta 允许范围             (默认 [1e-12, 1e12])
%             .rho        beta 伸缩因子             (默认 2)
%             .minComp    非退化最小分量样本数       (默认 2)
%             .dataTol    可行后目标相对收敛阈值     (默认 1e-6)
%             .verbose    是否打印调试信息           (默认 false)

    if nargin < 6 || isempty(opts), opts = struct(); end
    if ~isfield(opts, 'rankMode')  || isempty(opts.rankMode),  opts.rankMode  = 'hard'; end
    if ~isfield(opts, 'innerMax')  || isempty(opts.innerMax),  opts.innerMax  = 40; end
    if ~isfield(opts, 'rankTol')   || isempty(opts.rankTol),   opts.rankTol   = 1e-8; end
    if ~isfield(opts, 'initScale') || isempty(opts.initScale), opts.initScale = 1e-3; end
    if ~isfield(opts, 'softScale') || isempty(opts.softScale), opts.softScale = 1.0; end
    if ~isfield(opts, 'betaRange') || isempty(opts.betaRange), opts.betaRange = [1e-12, 1e12]; end
    if ~isfield(opts, 'rho')       || isempty(opts.rho),       opts.rho       = 2; end
    if ~isfield(opts, 'minComp')   || isempty(opts.minComp),   opts.minComp   = 2; end
    if ~isfield(opts, 'dataTol')   || isempty(opts.dataTol),   opts.dataTol   = 1e-6; end
    if ~isfield(opts, 'verbose')   || isempty(opts.verbose),   opts.verbose   = false; end

    [n, m] = size(D);
    if k > min(n, m)
        error('PGAL_lin_solveZ:k', 'k 必须不大于 min(n,m) = %d。', min(n, m));
    end
    alpha = max(alpha(:), 1e-12);

    % ---- 模式 1: 无秩约束 (beta = 0) ----
    if strcmpi(opts.rankMode, 'none')
        Z = projSimplexRows(-D ./ (2 * alpha));
        beta = 0;
        info = summarizeZ(Z, k, opts);
        return;
    end

    % ---- 模式 2: 软秩正则 (单步温和 F/Z 更新) ----
    if strcmpi(opts.rankMode, 'soft')
        [F, ~, Dv] = solveF(Z, k);
        T = computeT(F, Dv, n);
        if ~(beta > 0)
            beta = opts.softScale * median(alpha) / max(mean(T(:)), eps);
            beta = min(max(beta, opts.betaRange(1)), opts.betaRange(2));
        end
        Z = projSimplexRows(-(D + beta * T) ./ (2 * alpha));
        info = summarizeZ(Z, k, opts);
        return;
    end

    % ---- 模式 3: 硬秩约束 (自适应 beta 直至恰好 k 个非退化连通分量) ----
    [F, ~, Dv] = solveF(Z, k);
    if ~(beta > 0)
        T0 = computeT(F, Dv, n);
        beta = opts.initScale * mean(D(:)) / max(mean(T0(:)), eps);
        beta = min(max(beta, opts.betaRange(1)), opts.betaRange(2));
    end

    betaUsed = beta;
    betaLo   = [];
    betaHi   = [];
    bestZ = [];  bestBeta = NaN;  bestData = inf;
    lastZ = [];  lastBeta = NaN;  lastData = inf;
    prevData = inf;

    for t = 1:opts.innerMax
        T = computeT(F, Dv, n);
        Zold = Z;
        Z = projSimplexRows(-(D + beta * T) ./ (2 * alpha));

        [FN, ~, DvN, sig] = solveF(Z, k);
        sigk = [sig(:); zeros(k, 1)];
        fn1  = k - sum(sigk(1:k));
        [sizes, ncomp] = compSizes(Z);
        dataNow = sum(D(:) .* Z(:)) + sum(alpha .* sum(Z.^2, 2));

        rankOK  = (ncomp == k) && (fn1 <= opts.rankTol);
        degFree = rankOK && (min(sizes) >= opts.minComp);

        if opts.verbose
            fprintf('    [Z-lin] inner %2d: beta=%.3e fn1=%.2e ncomp=%d minSz=%d data=%.4e\n', ...
                t, beta, fn1, ncomp, min(sizes), dataNow);
        end

        if degFree
            if dataNow < bestData
                bestZ = Z; bestBeta = beta; bestData = dataNow;
            end
            F = FN; Dv = DvN; betaUsed = beta;
            if dataNow < prevData * (1 - opts.dataTol)
                prevData = dataNow;
                continue;
            end
            break;
        elseif rankOK
            % 切出单样本的退化可行解: 记录兜底并将 beta 判为偏大
            if dataNow < lastData
                lastZ = Z; lastBeta = beta; lastData = dataNow;
            end
            betaHi = beta;
            Z = Zold;
        elseif ncomp > k
            betaHi = beta;
            Z = Zold;
        else
            betaLo = beta;
            F = FN; Dv = DvN; betaUsed = beta;
        end

        if isempty(betaLo) || isempty(betaHi)
            if isempty(betaHi), betaNew = beta * opts.rho;
            else,               betaNew = beta / opts.rho;
            end
        else
            betaNew = sqrt(betaLo * betaHi);
        end
        betaNew = min(max(betaNew, opts.betaRange(1)), opts.betaRange(2));
        if betaNew == beta
            break;
        end
        beta = betaNew;
        prevData = inf;
    end

    if ~isempty(bestZ)
        Z = bestZ; beta = bestBeta;
    elseif ~isempty(lastZ)
        [~, ncNow] = compSizes(Z);
        if ncNow ~= k
            Z = lastZ; beta = lastBeta;
        else
            beta = betaUsed;
        end
    else
        beta = betaUsed;
    end

    info = summarizeZ(Z, k, opts);
end

% ======================================================================
function info = summarizeZ(Z, k, opts)
    n = size(Z, 1);
    [~, ~, ~, sigF] = solveF(Z, k);
    sigk = [sigF(:); zeros(k, 1)];
    info.fn1End = k - sum(sigk(1:k));
    if numel(sigF) >= k + 1
        info.fn2End = 1 - sigF(k + 1);
    else
        info.fn2End = 1;
    end
    [sizes, ncomp] = compSizes(Z);
    info.ncomp   = ncomp;
    info.minSize = min(sizes);
    info.minFrac = info.minSize / n;
    info.rankOK  = (ncomp == k) && (info.fn1End <= opts.rankTol) && (info.minSize >= opts.minComp);
end

% ======================================================================
function Z = projSimplexRows(V)
%PROJSIMPLEXROWS  逐行精确欧氏单纯形投影 (Duchi et al., ICML 2008)
%   对每行 v 求解 argmin_{z>=0, 1'z=1} 0.5*||z - v||_2^2
    [n, m] = size(V);
    U = sort(V, 2, 'descend');
    cssv = cumsum(U, 2) - 1;
    ind = 1:m;
    cond = U - bsxfun(@rdivide, cssv, ind) > 0;
    rho = sum(cond, 2);
    rho = max(rho, 1);
    idx = sub2ind([n, m], (1:n)', rho);
    theta = cssv(idx) ./ rho;
    Z = max(bsxfun(@minus, V, theta), 0);
end

% ======================================================================
function T = computeT(F, Dv, n)
    Fu  = F(1:n, :);
    Fv  = F(n+1:end, :);
    Fvn = bsxfun(@rdivide, Fv, sqrt(Dv));
    T   = bsxfun(@plus, sum(Fu.^2, 2), sum(Fvn.^2, 2)') - 2 * (Fu * Fvn');
    T(T < 0) = 0;
end

% ======================================================================
function [F, Du, Dv, sig] = solveF(Z, k)
    n  = size(Z, 1);
    Dv = max(sum(Z, 1), eps)';
    Du = ones(n, 1);
    E  = bsxfun(@rdivide, Z, sqrt(Dv)');
    [U, S, V] = mySVD(E, min(k + 1, min(size(E))));
    sig = full(diag(S));
    c   = sqrt(2) / 2;
    kk  = min(k, size(U, 2));
    F   = [c * U(:, 1:kk); c * V(:, 1:kk)];
end

% ======================================================================
function [sizes, ncomp] = compSizes(Z, tol)
    if nargin < 2, tol = 1e-8; end
    [n, m] = size(Z);
    par = 1:(n + m);
    sz  = ones(1, n + m);
    [ii, jj] = find(Z > tol);
    for t = 1:numel(ii)
        a = ufFind(par, ii(t));
        b = ufFind(par, n + jj(t));
        if a ~= b
            if sz(a) < sz(b), tmp = a; a = b; b = tmp; end
            par(b) = a; sz(a) = sz(a) + sz(b);
        end
    end
    root = zeros(n, 1);
    for i = 1:n, root(i) = ufFind(par, i); end
    [~, ~, ic] = unique(root);
    sizes = sort(accumarray(ic, 1), 'descend')';
    ncomp = numel(sizes);
end

function r = ufFind(par, x)
    r = x;
    while par(r) ~= r, r = par(r); end
end
