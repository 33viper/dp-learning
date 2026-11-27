function cmp_ablation_plan1
%CMP_ABLATION_PLAN1  系统消融实验: 原 PGAL(无秩/有秩) vs 方案 1(线性数据项 + 自适应 alpha + 先验引导 + 秩约束)
%
%   对比 6 组变体（覆盖 m in [k, 2k, 4k, 8k] x lambda in [1e-6, 1e-2, 1e2, 1e6] 共 16 组配置）:
%     1) orig_noR    : 原始 PGAL (d_ij z_ij^2, 无秩约束, 谱嵌入 + k-means 读数)
%     2) orig_R      : 原 PGAL-rank (d_ij z_ij^2, 硬秩约束, 连通分量读数)
%     3) lin_noR     : 方案 1 消融无秩版 (d_ij z_ij + alpha_i z_ij^2, beta=0, 谱嵌入读数)
%     4) plan1_hardR : 方案 1 硬秩版 (自适应 K + CAN alpha_i + 硬秩约束, 连通分量直接读数)
%     5) plan1_sth   : 方案 1 两阶段 Soft-then-Hard 版 (5 轮软秩预对齐 + 硬秩, 连通分量直接读数)
%     6) plan1_soft  : 方案 1 软秩正则版 (温和固定 beta + 归一化谱嵌入读数)
%
%   运行方式（在 PGAL-rank 目录下）:
%     addpath(genpath('./')); cmp_ablation_plan1

    addpath(genpath('./'));
    dsList = {'ForestTypes', 'Wikipedia'};
    lams   = [1e-6, 1e-2, 1e2, 1e6];
    variants = {'orig_noR', 'orig_R', 'lin_noR', 'plan1_hardR', 'plan1_sth', 'plan1_soft'};

    for d = 1:numel(dsList)
        dsName = dsList{d};
        S = load(fullfile('./datasets', [dsName, '.mat']));
        X = S.X;
        Y = S.Y(:);
        k = numel(unique(Y));
        n = numel(Y);
        ms = [k, 2*k, 4*k, 8*k];

        fprintf('\n============================================================================\n');
        fprintf('Dataset: %s (n=%d, k=%d) | 16 configs: m=[%s], lambda=[1e-6,1e-2,1e2,1e6]\n', ...
            dsName, n, k, num2str(ms));
        fprintf('============================================================================\n');
        fprintf('%-14s | %8s %8s %8s %8s %7s %8s\n', ...
            'variant', 'meanACC', 'bestACC', 'meanNMI', 'bestNMI', 'rankOK', 'time(s)');
        fprintf('----------------------------------------------------------------------------\n');

        for vi = 1:numel(variants)
            vname = variants{vi};
            accs = zeros(1, numel(ms) * numel(lams));
            nmis = zeros(1, numel(ms) * numel(lams));
            oks  = zeros(1, numel(ms) * numel(lams));
            idx  = 0;
            t0   = tic;

            for m = ms
                for lam = lams
                    idx = idx + 1;
                    switch vname
                        case 'orig_noR'
                            [~, Z, U] = PGAL(X, k, m, lam, 1e-12, false, struct('adaptive', false, 'innerMax', 1, 'blockInit', false));
                            [a, b] = kmRead(U, Y, k);
                            ok = false;
                        case 'orig_R'
                            [~, Z, U] = PGAL(X, k, m, lam);
                            [a, b, ok] = readGraphOrKM(Z, U, Y, k);
                        case 'lin_noR'
                            [~, Z, U] = PGAL_lin(X, k, m, lam, struct('rankMode', 'none'));
                            [a, b] = kmRead(U, Y, k);
                            ok = false;
                        case 'plan1_hardR'
                            [~, Z, U] = PGAL_lin(X, k, m, lam, struct('rankMode', 'hard'));
                            [a, b, ok] = readGraphOrKM(Z, U, Y, k);
                        case 'plan1_sth'
                            [~, Z, U] = PGAL_lin(X, k, m, lam, struct('rankMode', 'sth', 'softSteps', 5, 'softScale', 1.0));
                            [a, b, ok] = readGraphOrKM(Z, U, Y, k);
                        case 'plan1_soft'
                            [~, Z, U] = PGAL_lin(X, k, m, lam, struct('rankMode', 'soft', 'softScale', 1.0));
                            [a, b, ok] = readGraphOrKM(Z, U, Y, k);
                    end
                    accs(idx) = a;
                    nmis(idx) = b;
                    oks(idx)  = double(ok);
                end
            end

            elapsed = toc(t0);
            fprintf('%-14s | %8.4f %8.4f %8.4f %8.4f %6.1f%% %8.2f\n', ...
                vname, mean(accs), max(accs), mean(nmis), max(nmis), 100 * mean(oks), elapsed);
        end
    end
end

function [a, b, ok] = readGraphOrKM(Z, U, Y, k)
    [lab, nc, sizes] = PGAL_Z2labels(Z);
    ok = (nc == k) && (min(sizes) >= 2);
    if ok
        r = Clustering8Measure(Y, lab);
        a = r(1);
        b = r(2);
    else
        [a, b] = kmRead(U, Y, k);
    end
end

function [a, n] = kmRead(U, Y, k)
    W = bsxfun(@rdivide, U, max(sqrt(sum(U.^2, 2)), eps));
    aa = zeros(1, 5);
    nn = zeros(1, 5);
    for t = 1:5
        lab = litekmeans(W, k, 'MaxIter', 100, 'Replicates', 20);
        r = Clustering8Measure(Y, lab);
        aa(t) = r(1);
        nn(t) = r(2);
    end
    a = median(aa);
    n = median(nn);
end
