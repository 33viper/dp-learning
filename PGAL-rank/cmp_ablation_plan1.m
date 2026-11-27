function cmp_ablation_plan1
%CMP_ABLATION_PLAN1  方案一内部消融实验（无秩 / 硬秩 / 两阶段 Soft-then-Hard / 纯软秩）
%
%   在 ForestTypes 与 Wikipedia 上统一评测 16 组配置:
%     m in [k, 2k, 4k, 8k], lambda in [1e-6, 1e-2, 1e2, 1e6]
%
%   对比 4 组方案一模式:
%     1) lin_noR     : 方案一无秩版 (rankMode='none', beta=0, 归一化谱嵌入 k-means 读数)
%     2) plan1_hardR : 方案一硬秩版 (rankMode='hard', 严格 k 个非退化连通分量直接读标签)
%     3) plan1_sth   : 方案一两阶段版 (rankMode='sth', 5 轮软秩预对齐 + 严格硬秩直接读标签)
%     4) plan1_soft  : 方案一软秩版 (rankMode='soft', 温和固定 beta + 归一化谱嵌入读数)
%
%   运行方式（在 PGAL-rank 目录下）:
%     addpath(genpath('./')); cmp_ablation_plan1

    addpath(genpath('./'));
    dsList   = {'ForestTypes', 'Wikipedia'};
    lams     = [1e-6, 1e-2, 1e2, 1e6];
    variants = {'lin_noR', 'plan1_hardR', 'plan1_sth', 'plan1_soft'};

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
