clear;
clc;
warning off;
addpath(genpath('./'));

%% 方案一（PGAL_lin）主运行脚本
%  目标函数: 线性数据项 + CAN 自适应 alpha + 先验引导锚点 + 二分图秩约束
%  支持模式:
%    'hard' (默认): 严格 k 个非退化连通分量，免 k-means 直接读标签
%    'sth'        : 5 轮温和软秩预对齐 + 严格 k 个非退化连通分量
%    'soft'       : 温和软秩正则 + 归一化谱嵌入 k-means 读数
%    'none'       : 无秩约束消融对照

rankMode = 'hard';
ds       = {'ForestTypes', 'Wikipedia'};
dsPath   = './datasets/';
resPath  = './res-lmd/';
if ~exist(resPath, 'dir'), mkdir(resPath); end
if ~exist('./totalResults', 'dir'), mkdir('./totalResults'); end

hdr = {'anchor','lambda','ncomp','beta','ACC','NMI','Purity','Fscore', ...
       'Precision','Recall','AR','Entropy','maxFrac','minFrac','iter','time'};

for dsi = 1:length(ds)
    dataName = ds{dsi};
    fprintf('\n==================== %s (mode=%s) ====================\n', dataName, rankMode);
    load(strcat(dsPath, dataName));
    k = length(unique(Y));
    n = length(Y);

    lambda = [1e-6 1e-2 10^2 10^6];
    anchor = [k 2*k 4*k 8*k];

    txtpathmean = strcat(resPath, strcat(dataName, '_plan1_mean.txt'));
    fid = fopen(txtpathmean, 'a');
    fprintf(fid, 'Dataset:%s  Mode:%s  Date:%s\n', dataName, rankMode, datestr(now));
    fprintf(fid, '%s\n', strjoin(hdr, '\t'));
    fclose(fid);

    allvalid = [];
    allres   = [];
    minComp  = 2;

    for ichor = 1:length(anchor)
        for id = 1:length(lambda)
            tic;
            opts = struct('rankMode', rankMode);
            [~, Z, U, iter, ~, ~, beta] = PGAL_lin(X, k, anchor(ichor), lambda(id), opts);
            [labels, ncomp, sizes] = PGAL_Z2labels(Z);
            if (ncomp == k) && (min(sizes) >= minComp)
                resmean = Clustering8Measure(Y, labels);
                isValid = true;
            else
                resmean = kmRead8(U, Y, k);
                isValid = false;
            end
            maxFrac = max(sizes) / n;
            minFrac = min(sizes) / n;
            elapsed = toc;

            row = [anchor(ichor) lambda(id) ncomp beta resmean maxFrac minFrac iter elapsed];
            allres = [allres; row]; %#ok<AGROW>
            if isValid || ~strcmpi(rankMode, 'hard')
                allvalid = [allvalid; row]; %#ok<AGROW>
            end

            fprintf(['Anchor:%d \t Lambda:%g\t beta:%10.2e\t Kcomp:%d\t ' ...
                     'ACC:%8.4f NMI:%8.4f Pur:%8.4f F:%8.4f\t maxF:%.2f minF:%.3f it:%d\t Time:%8.2f\n'], ...
                     anchor(ichor), lambda(id), beta, ncomp, resmean(1), resmean(2), ...
                     resmean(3), resmean(4), maxFrac, minFrac, iter, elapsed);
            dlmwrite(txtpathmean, row, '-append', 'delimiter', '\t', 'newline', 'pc');
        end
    end

    if ~isempty(allvalid)
        [~, bestIdx] = max(allvalid(:, 5));
        bestRow = allvalid(bestIdx, :);
        fprintf('  [best] %d/%d valid, m=%d lambda=%g ACC=%.4f NMI=%.4f | meanACC=%.4f meanNMI=%.4f\n', ...
            size(allvalid, 1), size(allres, 1), bestRow(1), bestRow(2), ...
            bestRow(5), bestRow(6), mean(allvalid(:, 5)), mean(allvalid(:, 6)));

        fid = fopen('./totalResults/AllDatasetResultMean.txt', 'a');
        fprintf(fid, '%s\t%s\n', dataName, strjoin(hdr, '\t'));
        fprintf(fid, '%s\t', dataName);
        fprintf(fid, '%g\t', bestRow(1:end-1));
        fprintf(fid, '%.2f\n', bestRow(end));
        fclose(fid);
    end
end

function resmean = kmRead8(U, Y, k)
    W = bsxfun(@rdivide, U, max(sqrt(sum(U.^2, 2)), eps));
    M = zeros(5, 8);
    for t = 1:5
        lab = litekmeans(W, k, 'MaxIter', 100, 'Replicates', 20);
        M(t, :) = Clustering8Measure(Y, lab);
    end
    resmean = median(M, 1);
end
