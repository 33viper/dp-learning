clear;
clc;
warning off;
addpath(genpath('./'));

%% dataset
ds = {'ForestTypes','Wikipedia'};
dsPath = './datasets/';
resPath = './res-lmd/';

% 结果文件列含义（每次运行都写入表头，便于事后审计）
hdr = {'anchor','lambda','ncomp','beta','ACC','NMI','Purity','Fscore', ...
       'Precision','Recall','AR','Entropy','maxFrac','minFrac','iter','time'};

for dsi = 1:length(ds)
    dataName = ds{dsi}; disp(dataName);
    load(strcat(dsPath,dataName));
    k = length(unique(Y));
    n = length(Y);

    lambda = [1e-6 1e-2 10^2 10^6];
    anchor = [k 2*k 4*k 8*k];   % 含 m=k: Wikipedia 这类两视图质量悬殊的数据上最优就在 m=k

    txtpathmean = strcat(resPath,strcat(dataName,'_mean.txt'));
    fid = fopen(txtpathmean,'a');
    fprintf(fid,'Dataset:%s  Date:%s\n', dataName, datestr(now));
    fprintf(fid,'%s\n', strjoin(hdr,'\t'));
    fclose(fid);

    %%
    allvalid = [];        % 只保留"恰好 k 个非退化分量"的配置（用于取最优/均值）
    allres   = [];        % 全部配置（含失败），仅用于统计个数
    nfail = 0;            % 未满足非退化秩条件的配置数
    minComp = 2;          % 每个分量至少 2 个样本: 排除"切单点凑数"的退化可行解
    for ichor = 1:length(anchor)
        for id = 1:length(lambda)
            tic;
            [~,Z,~,iter,~,~,beta] = PGAL(X,k,anchor(ichor),lambda(id));
            % 秩约束已保证 Z 恰好有 k 个连通分量, 直接读分量标签, 不做 k-means 后处理
            [labels,ncomp,sizes] = PGAL_Z2labels(Z);
            resmean = Clustering8Measure(Y,labels);
            maxFrac = max(sizes)/n;
            minFrac = min(sizes)/n;
            timer(ichor,id) = toc;
            row = [anchor(ichor) lambda(id) ncomp beta resmean maxFrac minFrac iter timer(ichor,id)];
            allres = [allres; row]; %#ok<AGROW>
            if (ncomp ~= k) || (min(sizes) < minComp)
                fprintf(2,'  [warning] anchor=%d lambda=%g: 分量数 %d (k=%d), 最小分量 %d 个样本 (minFrac=%.3f)\n', ...
                    anchor(ichor),lambda(id),ncomp,k,min(sizes),minFrac);
                nfail = nfail + 1;      % 失败/退化配置不计入最优/均值
            else
                allvalid = [allvalid; row]; %#ok<AGROW>
            end
            fprintf(['Anchor:%d \t Lambda:%g\t beta:%10.2e\t Kcomp:%d\t ' ...
                     'ACC:%8.4f NMI:%8.4f Pur:%8.4f F:%8.4f\t maxF:%.2f minF:%.3f it:%d\t Time:%8.2f\n'], ...
                     anchor(ichor), lambda(id), beta, ncomp, resmean(1), resmean(2), ...
                     resmean(3), resmean(4), maxFrac, minFrac, iter, timer(ichor,id));
            dlmwrite(txtpathmean, row, '-append','delimiter','\t','newline','pc');
        end
    end

    if isempty(allvalid)
        fprintf(2,'  [warning] %d 个配置未满足"恰好 k 个非退化分量", 不给出最优值\n', nfail);
        maxresultmean = nan(1,size(allres,2));
    else
        [~,d] = max(allvalid(:,5));            % 第 5 列是 ACC
        maxresultmean = allvalid(d,:);
        fprintf('  [best] %d/%d 个配置满足非退化秩条件, 最优 m=%d lambda=%g ACC=%.4f NMI=%.4f\n', ...
            size(allvalid,1), size(allres,1), maxresultmean(1), maxresultmean(2), ...
            maxresultmean(5), maxresultmean(6));
    end

    fid = fopen('./totalResults/AllDatasetResultMean.txt','a');
    fprintf(fid,'%s\t%s\n', dataName, strjoin(hdr,'\t'));
    fprintf(fid,'%s\t', dataName);
    fprintf(fid,'%g\t', maxresultmean(1:end-1));
    fprintf(fid,'%.2f\n', maxresultmean(end));
    fclose(fid);
end
