function [labels, ncomp, sizes] = PGAL_Z2labels(Z, tol)
%PGAL_Z2LABELS  由带秩约束的锚图 Z 直接读出聚类标签（不做 k-means 后处理）
%
%   秩约束保证 H = [0 Z; Z' 0] 恰好有 k 个连通分量, 因此样本所属类别
%   就是它在二部图中的连通分量编号, 不需要再对嵌入 U 做 k-means。
%   （k-means 只在 Z 没有精确 k 分量结构时才作为"量化/修补"手段。）
%
%   [labels, ncomp, sizes] = PGAL_Z2labels(Z)
%   Z     : n x m 锚图, Z >= 0, 行和为 1
%   tol   : 视为"有边"的阈值（默认 1e-8；单纯形投影产生的 0 是精确 0）
%   labels: n x 1, 取值 1..ncomp
%   ncomp : 样本侧连通分量个数（正常情况下应等于 k; 孤立锚点不计入）
%   sizes : ncomp x 1, 各分量的样本数

    if nargin < 2 || isempty(tol)
        tol = 1e-8;
    end

    [n, m] = size(Z);
    nb = n + m;                     % 二部图节点数
    parent = 1:nb;

    [rw, cl] = find(Z > tol);       % 二部图的边: 样本 rw(e) -- 锚点 n+cl(e)
    for e = 1:numel(rw)
        ra = rw(e);
        while parent(ra) ~= ra
            parent(ra) = parent(parent(ra));    % 路径减半
            ra = parent(ra);
        end
        rb = n + cl(e);
        while parent(rb) ~= rb
            parent(rb) = parent(parent(rb));
            rb = parent(rb);
        end
        if ra ~= rb
            if ra < rb, parent(rb) = ra; else, parent(ra) = rb; end
        end
    end

    root = zeros(nb, 1);
    for i = 1:nb
        r = i;
        while parent(r) ~= r
            r = parent(r);
        end
        root(i) = r;
    end

    [u, ~, ic] = unique(root(1:n));          % 只取样本节点
    labels = ic(:);
    ncomp = numel(u);
    sizes = accumarray(labels, 1, [ncomp, 1]);
end
