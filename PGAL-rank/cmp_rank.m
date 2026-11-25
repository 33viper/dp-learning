% 对比: 加秩约束 vs 不加秩约束
% 无秩约束 = beta 固定 1e-12 且 adaptive=false（罚项实际不存在）
% 无秩时 Z 是稠密的, 连通分量只有 1 个, 所以必须用谱嵌入 + kmeans 读出
function cmp_rank

files = {'./datasets/ForestTypes.mat', './datasets/SyntheticClean.mat'};
names = {'ForestTypes', 'SyntheticClean'};
ms    = {[4 8 16 32], [3 6 12]};

for d = 1:2
    S = load(files{d});
    X = S.X; Y = S.Y(:); k = numel(unique(Y));
    fprintf('\n=== %s  n=%d  k=%d ===\n', names{d}, numel(Y), k);
    fprintf('%-5s %-8s %-8s %-8s %-6s | %-8s %-6s | %-8s %-8s %-6s\n', ...
            'm', 'ACC0', 'NMI0', 'nnz/行', 'nc', 'ACC1', 'nc', 'ACC2', 'KMACC2', 'nc');
    for m = ms{d}
        % (0) 无秩约束 + 原始 kNN 初值
        [~,Z0,U0] = PGAL(X,k,m,1e-2,1e-12,false,struct('adaptive',false,'blockInit',false));
        [a0,n0] = kmRead(U0,Y,k);
        [~,nc0]  = PGAL_Z2labels(Z0);
        dens = mean(sum(Z0 > 1e-8, 2));
        % (1) 无秩约束 + 当前块初值（隔离初始化的贡献）
        [~,Z1,U1] = PGAL(X,k,m,1e-2,1e-12,false,struct('adaptive',false,'blockInit',true));
        [a1,~]   = kmRead(U1,Y,k);
        [~,nc1]  = PGAL_Z2labels(Z1);
        % (2) 当前完整代码（秩约束）
        [~,Z2,U2] = PGAL(X,k,m,1e-2);
        lab = PGAL_Z2labels(Z2);
        r   = Clustering8Measure(Y, lab);
        [a2,~]   = kmRead(U2,Y,k);
        [~,nc2]  = PGAL_Z2labels(Z2);
        fprintf('%-5d %-8.4f %-8.4f %-8.1f %-6d | %-8.4f %-6d | %-8.4f %-8.4f %-6d\n', ...
                m, a0, n0, dens, nc0, a1, nc1, r(1), a2, nc2);
    end
end
end

function [a,n] = kmRead(U,Y,k)
% 行归一化后 kmeans, 5 次取中位数
W = U ./ max(sqrt(sum(U.^2,2)), eps);
aa = zeros(1,5); nn = zeros(1,5);
for t = 1:5
    lab = litekmeans(W,k,'MaxIter',100,'Replicates',20);
    r = Clustering8Measure(Y,lab);
    aa(t) = r(1); nn(t) = r(2);
end
a = median(aa); n = median(nn);
end
