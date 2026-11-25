function [Z, beta, info] = PGAL_solveZ(Z, D, k, beta, opts)
%PGAL_SOLVEZ  PGAL 中带秩约束的锚图 Z 子问题（替代原来逐行解 min sum d_ij z_ij^2）
%
%   求解
%       min_{Z>=0, Z*1=1}  sum_{i,j} d_ij z_ij^2 + beta * sum_{t=1}^{k} delta_t(L_H)
%   辅助图（与论文一致, 样本在前）:
%       H = [ 0_n   Z  ]  in R^{(n+m)x(n+m)},   D = diag(D_u, D_v)
%           [ Z'    0_m ]
%       L_H = I - D^{-1/2} H D^{-1/2},  delta_t 为第 t 小特征值。
%
%   由 Ky Fan 定理改写为交替问题
%       min_{Z,F} sum_{i,j} d_ij z_ij^2 + beta*Tr(F'*L_H*F),  s.t. Z>=0, Z*1=1, F'F=I_k
%
%   1) 固定 Z 更新 F = [F_u; F_v]（按 H 的块顺序: 样本块 F_u 在上, 锚点块 F_v 在下）:
%        E = D_u^{-1/2} Z D_v^{-1/2}, 取 E 的前 k 个奇异向量 U,V
%        F_u = sqrt(2)/2*U,  F_v = sqrt(2)/2*V     (满足 F_u'F_u+F_v'F_v = I_k)
%   2) 固定 F 逐行更新 Z:
%        t_ij = ||f_u^i/sqrt(D_u(i,i)) - f_v^j/sqrt(D_v(j,j))||_2^2
%        z_i  = argmin_{z>=0, sum_j z_j = 1} sum_j d_ij z_j^2 + beta*sum_j t_ij z_j
%             = 加权单纯形投影, 用式 (21)(22) 的阈值解(mu 二分)
%
%       精确性说明: t_ij 里的 D_u,D_v 本身也是 Z 的函数, 且 D_v(j,j)=sum_i z_ij,
%       所以行与行通过 D_v 耦合, 严格来说不能逐行独立求解。这里的做法是固定
%       上一轮 Z 的 D_u,D_v(冻结度矩阵), 把秩项在当前点线性化, 解完整轮 Z 后
%       再用新 Z 重算 F/D_u/D_v。因此本函数是"冻结度矩阵的块交替/线性化近似",
%       不是 Z 子问题的精确解; 对带罚目标不保证单调下降, 只对代理函数单调。
%   3) beta 自适应（参考 CAN, Nie et al., KDD'14）:
%        L_H 的特征值为 1 +- sigma_t(E), 故
%            sum_{t=1..k} delta_t(L_H) = k - sum_{t=1..k} sigma_t(E)
%            delta_{k+1}(L_H)          = 1 - sigma_{k+1}(E)
%        前者 > rankTol -> 零特征值不足 k 个 -> beta = 2*beta
%        后者 < rankTol -> 零特征值多于 k 个 -> beta = beta/2 并回滚本次 Z 更新
%        两者都满足     -> Z 恰好 k 个连通分量 -> 立即跳出（与 PCAN 的 break 一致）
%   4) 分量规模诊断（不参与优化）:
%        归一化割对分量体积不设下界, 于是"切出一小撮样本"几乎不付代价,
%        真实数据上会出现 [大块 大块 大块 极小碎片] 这类划分。
%        这里用并查集统计分量数与最小分量占比, 写入 info 供外层判断与打印。
%
%   Z     : n x m 锚图（输入当前值作热启动，输出更新值）
%   D     : n x m 距离矩阵, d_ij = sum_p gamma_p^2*||x_i^p - a_j^p||^2
%   k     : 连通分量数（类别数）
%   beta  : 秩约束惩罚参数; beta <= 0 时按数据尺度自动给初值
%   opts  : 可选
%           .innerMax  F/Z 交替的最大次数          (默认 40)
%           .rankTol   判断特征值是否为 0 的阈值   (默认 1e-8)
%           .adaptive  是否按秩条件自动调整 beta   (默认 true)
%           .initScale beta 自动初值的缩放系数     (默认 1e-3)
%           .betaRange beta 的允许范围             (默认 [1e-12, 1e12])
%           .verbose   是否打印 beta 调整过程      (默认 false)

    if nargin < 5 || isempty(opts), opts = struct(); end
    if ~isfield(opts, 'innerMax')  || isempty(opts.innerMax),  opts.innerMax  = 40; end
    if ~isfield(opts, 'rankTol')   || isempty(opts.rankTol),   opts.rankTol   = 1e-8; end
    if ~isfield(opts, 'adaptive')  || isempty(opts.adaptive),  opts.adaptive  = true; end
    if ~isfield(opts, 'initScale') || isempty(opts.initScale), opts.initScale = 1e-3; end
    if ~isfield(opts, 'betaRange') || isempty(opts.betaRange), opts.betaRange = [1e-12, 1e12]; end
    if ~isfield(opts, 'rho')       || isempty(opts.rho),       opts.rho       = 2; end
    if ~isfield(opts, 'minComp')   || isempty(opts.minComp),   opts.minComp   = 2; end
    if ~isfield(opts, 'dataTol')   || isempty(opts.dataTol),   opts.dataTol   = 1e-6; end
    if ~isfield(opts, 'verbose')   || isempty(opts.verbose),   opts.verbose   = false; end

    [n, m] = size(D);
    if k > min(n, m)
        error('PGAL_solveZ:k', 'k 必须不大于 min(n,m) = %d。', min(n, m));
    end
    Dsafe = max(D, 1e-12);          % 避免 d_ij = 0 时除零

    needInit = ~(beta > 0);         % 需要按数据尺度自动给 beta 初值

    info = struct('rankOK', false, 'fn1End', NaN, 'fn2End', NaN, ...
                  'minFrac', NaN, 'ncomp', NaN);

    [F, Du, Dv] = solveF(Z, k);

    betaUsed = beta;                % 产生当前 Z 的 beta（返回值必须与 Z 一致）
    betaLo   = [];                  % 已知"分量偏少"的 beta 下界
    betaHi   = [];                  % 已知"分量偏多/靠切单点凑数"的 beta 上界

    % 候选解: 可行集上秩项恒为 0, 故约束问题的目标就是数据项 sum d_ij z_ij^2。
    % 取"数据项最小"的合格可行解, 结果与 innerMax、beta 搜索顺序无关。
    bestZ = [];  bestBeta = NaN;  bestData = inf;    % 非退化可行
    lastZ = [];  lastBeta = NaN;  lastData = inf;    % 退化可行（兜底）
    prevData = inf;                                  % 固定 beta 下的收敛监测

    for t = 1:opts.innerMax
        % ---------- (1) 由 F 得到 t_ij ----------
        % t_ij = ||f_u^i - f_v^j/sqrt(d_j)||^2   (文档式 (6)(7))
        % 其中 D_u = I_n 恒成立(行和为 1), 故样本块不做度归一化
        Fu = F(1:n, :);                     % 样本块（H 的前 n 个节点）
        Fv = F(n+1:end, :);                 % 锚点块（H 的后 m 个节点）
        Fvn = Fv ./ sqrt(Dv);               % D_v^{-1/2} F_v
        T = sum(Fu.^2, 2) + sum(Fvn.^2, 2)' - 2 * (Fu * Fvn');
        T(T < 0) = 0;

        % ---------- beta 自动初值 ----------
        if needInit
            % 先给出一个偏小的初值, 再靠下面的加倍过程升到"恰好 k 个分量"的最小 beta
            lam0 = median(1 ./ sum(1 ./ Dsafe, 2));
            beta = opts.initScale * 2 * lam0 / max(mean(T(:)), eps);
            needInit = false;
        end

        % ---------- (2) 固定 F, 逐行更新 Z ----------
        % z 为 Z 的第 i 行 (1 x m), 即式 (20):
        % min sum_j b_ij z_ij^2 + beta*sum_j t_ij z_ij   (sum_j z_ij = 1, z_ij >= 0)
        %  <=> min 1/2 * z*diag(2b)*z^T + (beta*t)*z^T
        % 用式 (21)(22) 的精确阈值解(对 mu 二分), 不走 Newton。
        Zold = Z;
        Z = projSimplexDiagBound(2 * Dsafe, beta * T, 1);

        % ---------- (3) 重新计算 F, 检查秩条件 ----------
        [FN, DuN, DvN, sig] = solveF(Z, k);

        sigk = [sig(:); zeros(k, 1)];
        fn1  = k - sum(sigk(1:k));                  % sum_{t<=k} delta_t(L_H)
        if numel(sig) >= k + 1
            fn2 = 1 - sig(k + 1);                   % delta_{k+1}(L_H)
        else
            fn2 = 1;
        end
        [sizes, ncomp] = compSizes(Z);
        minFrac = min(sizes) / n;                    % 最小分量占比
        % 判据直接用并查集数出来的分量数（精确, 且不依赖 sigma_{k+1} 是否存在）;
        % m <= k 时 E 只有 m 个奇异值, fn2 恒为 1, 谱判据在那一档只剩 fn1 可用。
        info.rankOK = (ncomp == k) && (fn1 <= opts.rankTol);
        dataNow = sum(Dsafe(:) .* Z(:).^2);          % 可行时它就是约束问题的目标

        % 秩条件成立还不够: Ky Fan 罚项对任意 >=k 个分量的图都是 0,
        % 于是"切掉 k-1 个单样本"也能让 ncomp==k 且 fn1==0, 但那是退化解
        % (每个"簇"只有一个样本)。要求每个分量至少 opts.minComp 个样本。
        degFree = info.rankOK && (min(sizes) >= opts.minComp);

        if opts.verbose
            fprintf(['    [Z] inner %2d: beta=%.3e fn1=%.2e fn2=%.2e ' ...
                     'ncomp=%d minf=%.3f obj=%.4e\n'], ...
                     t, beta, fn1, fn2, ncomp, minFrac, dataNow);
        end

        if ~opts.adaptive
            F = FN; Du = DuN; Dv = DvN;
            betaUsed = beta;
            continue;
        end

        if degFree
            % 合格可行解: 记下数据项最小者（可行集上秩项为 0, 数据项即约束目标）
            if dataNow < bestData
                bestZ = Z; bestBeta = beta; bestData = dataNow;
            end
            F = FN; Du = DuN; Dv = DvN;
            betaUsed = beta;
            % 固定 beta 继续 F/Z 交替, 直到目标不再下降: 返回不动点,
            % 而不是"第一次可行"处那个依赖 innerMax/搜索路径的点。
            if dataNow < prevData * (1 - opts.dataTol)
                prevData = dataNow;
                continue;
            end
            if opts.verbose
                fprintf('    [Z] inner %2d: beta=%.3e 下已收敛, 停止\n', t, beta);
            end
            break;
        elseif info.rankOK
            % 靠切单点凑出来的可行解: 视为 beta 过大, 记录兜底并向下二分
            if dataNow < lastData
                lastZ = Z; lastBeta = beta; lastData = dataNow;
            end
            betaHi = beta;
            Z = Zold;
        elseif ncomp > k
            betaHi = beta;                           % beta 偏大, 分量被切碎
            Z = Zold;                                % 回滚, 保留旧 Z 与旧 F
        else
            betaLo = beta;                           % beta 偏小, 分量不足
            F = FN; Du = DuN; Dv = DvN;              % 接受本次 Z, F 同步
            betaUsed = beta;
        end
        if isempty(betaLo) || isempty(betaHi)
            if isempty(betaHi), betaNew = beta * opts.rho;
            else,               betaNew = beta / opts.rho;
            end
        else
            betaNew = sqrt(betaLo * betaHi);         % 上下界都已知, 直接二分
        end
        betaNew = min(max(betaNew, opts.betaRange(1)), opts.betaRange(2));
        if betaNew == beta
            break;                                   % 浮点精度已到极限
        end
        beta = betaNew;
        prevData = inf;                              % beta 变了, 收敛监测重开
    end

    % 返回: 优先"非退化可行且数据项最小"的候选, 其次退化可行候选, 最后当前 Z
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
        beta = betaUsed;                % 返回与 Z 对应的 beta
    end

    % ---------- (4) 用最终返回的 Z 复核秩条件 ----------
    % 若最后一次是回滚(或多于 k 个分量), 上面的 report 针对的是被丢弃的 Z,
    % 这里重算一次, 保证 info.rankOK 与真正返回的 Z 一致。
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
    info.minFrac = min(sizes) / n;
    info.minSize = min(sizes);
    % 非退化才算满足秩条件: 分量数 = k 但靠单样本凑数的解不能算"找到 k 个簇"
    info.rankOK  = (ncomp == k) && (info.fn1End <= opts.rankTol) && (info.minSize >= opts.minComp);
    if opts.verbose && (ncomp == k) && (info.fn1End <= opts.rankTol) && (info.minSize < opts.minComp)
        fprintf(2, ['    [Z] 警告: beta=%.3e 下秩条件只靠切出 %d 个单样本分量满足' ...
                    '(最小分量 %d 个样本), 视为退化可行解。\n'], ...
                    beta, sum(sizes < opts.minComp), info.minSize);
    end

    % 文档 3.6 第 3 条: beta 触顶仍不足 k 个分量, 说明当前解陷入局部极小
    if ncomp < k && beta >= opts.betaRange(2) * (1 - 1e-12)
        fprintf(2, ['[PGAL_solveZ] 警告: beta 已达上界 %g 而连通分量数 %d < k=%d, ' ...
                    '当前解可能陷入局部极小。\n'], opts.betaRange(2), ncomp, k);
    end
end

% ======================================================================
function Z = projSimplexDiagBound(H, F, k)
%PROJSIMPLEXDIAGBOUND  逐行解  min 1/2 x'diag(H)x + F'x,  x>=0, 1'x=k
%   由 KKT 得文档式 (21):  x_j = max(0, (mu - F_j)/H_j),
%   mu 由归一化条件 (22) 唯一确定, 其左端关于 mu 单调不减, 这里用二分求。
%   为什么不直接复用 EProjSimplex_new_ZJP_V2: 那个函数用 Newton 迭代 + 步长启发式,
%   当 |F| >> H (秩惩罚项 beta 很大时正是如此) 有效集长时间为空, 100 步内可能走不完,
%   最后用"顶点解"兜底(整行只留一个非零), 已不是 (21) 的最优解。二分不依赖初值,
%   60 次即可到机器精度, 且可对所有行同时做。
    lo = min(F, [], 2);
    hi = max(F + H, [], 2);
    for it = 1:60
        mid   = 0.5 * (lo + hi);
        s     = sum(max(0, (mid - F) ./ H), 2);
        below = s < k;                    % s(mid) < k => mu 需要更大
        lo(below)  = mid(below);
        hi(~below) = mid(~below);
    end
    Z = max(0, (0.5 * (lo + hi) - F) ./ H);
    Z = Z .* (k ./ sum(Z, 2));            % 消掉二分残差, 保证行和恰为 k
end

% ======================================================================
function [sizes, ncomp] = compSizes(Z, tol)
%COMPSIZES  Z 定义的两部图（即论文 H=[0 Z; Z' 0]）的连通分量, 返回各分量的样本数
%           此处并查集的节点编号是样本 1..n、锚点 n+1..n+m, 与 H 的块顺序无关
    if nargin < 2, tol = 1e-8; end
    [n, m] = size(Z);
    par = 1:(n + m);
    sz  = ones(1, n + m);              % 并查集按规模合并, 深度 O(log n)
    [ii, jj] = find(Z > tol);
    for t = 1:numel(ii)
        a = ufFind(par, ii(t));
        b = ufFind(par, n + jj(t));
        if a ~= b
            if sz(a) < sz(b), tmp = a; a = b; b = tmp; end
            par(b) = a; sz(a) = sz(a) + sz(b);
        end
    end
    % 只统计样本侧的连通分量: 度为 0 的孤立锚点不进入 ncomp。
    % 全图分量数 c = ncomp + 孤立锚点数, rank(L_H) = n+m-c, 所以有孤立锚点时
    % rank(L_H) < n+m-k; 但 c >= k 时 Ky Fan 罚项仍为 0, 样本侧恰好 k 块的目标不受影响。
    root = zeros(n, 1);
    for i = 1:n, root(i) = ufFind(par, i); end
    [~, ~, ic] = unique(root);
    sizes  = accumarray(ic, 1);
    sizes  = sort(sizes, 'descend')';
    ncomp  = numel(sizes);
end

function r = ufFind(par, x)
    r = x;
    while par(r) ~= r, r = par(r); end
end

% ======================================================================
function [F, Du, Dv, sig] = solveF(Z, k)
%SOLVEF  固定 Z 时的 F 子问题（节点顺序与论文辅助图 H 一致: 样本在前）
%   H = [0_n Z; Z' 0_m] in R^{(n+m)x(n+m)},  D = diag(D_u, D_v)
%   L = I - D^{-1/2} H D^{-1/2}, 其特征值为 1 +- sigma_t(D_u^{-1/2} Z D_v^{-1/2})
%   F = [F_u; F_v] 按 H 的块顺序堆叠（样本块在前, 锚点块在后）, F'F = I_kk
    n = size(Z, 1);  m = size(Z, 2);                %#ok<ASGLU>
    Dv = max(sum(Z, 1), eps)';                      % 锚点侧度（H 右下块）
    Du = ones(n, 1);                                % 行和为 1 => D_u ≡ I_n（文档 2.2 / 7.1）
    E  = Z ./ sqrt(Dv)';                            % B = D_u^{-1/2} Z D_v^{-1/2} = Z D_v^{-1/2}
    [U, S, V] = mySVD(E, min(k + 1, min(size(E))));
    sig = full(diag(S));
    c   = sqrt(2) / 2;
    % mySVD 会丢掉近零奇异值, 所以图的分量数 c < k 时 U,V 只有 c 列, kk < k,
    % F 不满足 F'F = I_k。这对 Z 的更新没有影响:
    %   丢掉的 sigma 都是 0, Ky Fan 值 k - sum_{t<=k} sigma_t 仍按补零计算(见 fn1);
    %   少掉 k-kk 个方向时线性项 sum t_ij z_ij = kk - 2*sum z_ij a_ij 与真实值只差
    %   常数 beta*(kk-k), 在 sum_j z_ij = 1 的单纯形上不改变 argmin。
    % 因此这里用"零填充"的等价形式, 而不是补随机正交列(补列会改变 t, 使结果依赖随机数)。
    kk  = min(k, size(U, 2));                       % 奇异值可能少于 k 个
    Fu  = c * U(:, 1:kk);                           % n x kk, 样本块（H 的前 n 行）
    Fv  = c * V(:, 1:kk);                           % m x kk, 锚点块（H 的后 m 行）
    F   = [Fu; Fv];                                 % (n+m) x kk
end
