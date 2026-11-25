function rel = viewReliability(X, kk)
%VIEWRELIABILITY  各视图的可靠性权重（无监督, 不需要真实标签）
%
%   rel(p) = 视图 p 的 kNN 邻域 与 其它视图 kNN 邻域 的平均重叠率。
%   与其它视图的局部结构一致 -> 权重高; 与所有视图都不一致(噪声视图) -> 权重趋近 0。
%   返回归一化权重（和为 1）。
%
%   rel = viewReliability(X)        % kk 默认 10
%   rel = viewReliability(X, kk)

    if nargin < 2 || isempty(kk), kk = 10; end
    V = numel(X);
    if V == 1, rel = 1; return; end

    n  = size(X{1}, 1);
    kk = min(kk, n - 1);

    % 每个视图的 kNN 邻域: 用 knnsearch 而不是 pdist2 + sort,
    % 后者要显式存 n x n 距离矩阵（n 大时内存 O(n^2) 直接爆掉）。
    N = cell(1, V);
    for p = 1:V
        Xp = mapstd(X{p}', 0, 1)';
        [ix, ~] = knnsearch(Xp, Xp, 'K', kk + 1);   % 最近的是样本自身
        N{p} = ix(:, 2:end);
    end

    % 两两视图的邻域重叠率, 再对其它视图取平均
    rel = zeros(1, V);
    for p = 1:V
        ov = zeros(1, V);
        for q = 1:V
            if q == p, continue; end
            c = 0;
            for i = 1:n
                c = c + numel(intersect(N{p}(i, :), N{q}(i, :)));
            end
            ov(q) = c / (n * kk);
        end
        rel(p) = sum(ov) / (V - 1);
    end

    rel = max(rel, 0);
    if sum(rel) <= 0
        rel = ones(1, V) / V;
    else
        rel = rel / sum(rel);
    end
end
