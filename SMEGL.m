function [y_idx, Tobj] = SMEGL(B, Q, Ss, Fs, opts, r2Temp, betaTemp)
    k = opts.k; % 簇数
    v = length(Ss);
    n = size(Ss{1}, 1);

    % 初始化变量
    Rs = Fs;
    p = ones(1, v);
    M = Ss;
    A0 = cell(1, v);
    Bs = B;
    Qs = Q;

    for idx = 1:v
        A0{idx} = M{idx} - diag(diag(M{idx}));  % 移除自连接
    end

    u = ones(n, n); % 初始化权重矩阵

    % 初始化T
    T = zeros(n, k);
    for idx = 1:v
        Rt = Rs{idx};
        St = M{idx};
        T = T + p(idx) * (St * Rt);
    end
    Y = max(T / v, 0);  % 非负化

    max_iter = 100;
    Tobj = zeros(1, max_iter);

    for iter = 1:max_iter
        % 迭代更新 H
        T = zeros(n, k);
        for idx = 1:v
            Rt = Rs{idx};
            St = M{idx};
            T = T + p(idx) * (St * Rt);
        end
        Y = max(T / v, 0);

        % 迭代更新F
        for idx = 1:v
            St = M{idx};
            temp = St' * Y;
            [Ur, ~, Vr] = svds(temp, k);
            Rs{idx} = Ur * Vr';   
        end

        % 更新Q
        for idx = 1:v
            Qs{idx} = Bs{idx} / M{idx};
        end

        % 迭代更新 M
        for idx = 1:v
            temp = Rs{idx} * Y';
            temp_new = Q{idx}' * Bs{idx};
            Q_temp = Qs{idx}' * Qs{idx};
            M{idx} = zeros(n);
            for i = 1:n
                ai = A0{idx}(i, :);
                di = temp(i, :);
                di_new = temp_new(i,:);
                bi = Q_temp(i,:);
                si = EProjSimplexdiag(u(i) .* ai + r2Temp * di + betaTemp * di_new, u(i) + (r2Temp / 2) .* ones(1, n) + betaTemp * bi);
                u(i) = 1 ./ (2 * sqrt((si - ai) .^ 2 + eps));
                M{idx}(i, :) = si;
            end
        end

        % 目标函数计算
        obj = 0;
        for idx = 1:v
            Rt = Rs{idx};
            St = M{idx};
            Qt = Qs{idx};
            Bt = Bs{idx};
            obj = obj + norm(St - Y * Rt', 'fro')^2 + norm(Ss{idx} - St, 1) + norm(Bt - Qt * St, 'fro')^2;
        end
        Tobj(iter) = obj;

        % 收敛检查
        if iter > 1 && abs(obj - Tobj(iter - 1)) / Tobj(iter - 1) < 1e-8
            break;
        end
    end

    [~, y_idx] = max(Y, [], 2);  
end
