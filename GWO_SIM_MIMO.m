clc; clear; close all;

%% ================= THAM SO =================
SearchAgents = 30;      % So luong soi
MaxIter = 100;          % So vong lap
M = 32;                 % So phan tu metasurface

lb = zeros(1,M);        % Gioi han duoi
ub = 2*pi*ones(1,M);    % Gioi han tren

%% ================= KHOI TAO =================
Positions = rand(SearchAgents,M).*(ub-lb) + lb;

Alpha_pos  = zeros(1,M);
Beta_pos   = zeros(1,M);
Delta_pos  = zeros(1,M);

Alpha_score = -inf;
Beta_score  = -inf;
Delta_score = -inf;

Convergence_GWO = zeros(1,MaxIter);

%% ================= GWO =================
for t = 1:MaxIter

    for i = 1:SearchAgents

        % Gioi han bien
        Positions(i,:) = min(max(Positions(i,:),lb),ub);

        % Danh gia fitness
        fit = fitness_SIM_MIMO(Positions(i,:));

        % Cap nhat Alpha, Beta, Delta
        if fit > Alpha_score
            Delta_score = Beta_score;  Delta_pos = Beta_pos;
            Beta_score  = Alpha_score; Beta_pos  = Alpha_pos;
            Alpha_score = fit;         Alpha_pos = Positions(i,:);

        elseif fit > Beta_score
            Delta_score = Beta_score;  Delta_pos = Beta_pos;
            Beta_score  = fit;         Beta_pos  = Positions(i,:);

        elseif fit > Delta_score
            Delta_score = fit;
            Delta_pos   = Positions(i,:);
        end
    end

    % Luu lich su hoi tu
    Convergence_GWO(t) = Alpha_score;

    % He so dieu chinh
    a = 2 - 2*t/MaxIter;

    % Cap nhat vi tri soi
    for i = 1:SearchAgents
        for j = 1:M
            X1 = update_position(Alpha_pos(j), Positions(i,j), a);
            X2 = update_position(Beta_pos(j),  Positions(i,j), a);
            X3 = update_position(Delta_pos(j), Positions(i,j), a);
            Positions(i,j) = (X1 + X2 + X3)/3;
        end
    end

    fprintf('Iter %3d | Best Capacity = %.4f\n', t, Alpha_score);
end

disp('Optimal phase vector (GWO):');
disp(Alpha_pos);

%% ================= BASELINE (RANDOM) =================
Baseline = zeros(1,MaxIter);
for k = 1:MaxIter
    theta_rand = rand(1,M)*2*pi;
    Baseline(k) = fitness_SIM_MIMO(theta_rand);
end

%% ================= VE BIEU DO =================
figure;
plot(1:MaxIter, Convergence_GWO, 'r-', 'LineWidth', 2); hold on;
plot(1:MaxIter, Baseline, 'b--', 'LineWidth', 2);
grid on;
xlabel('Iteration');
ylabel('Capacity (bits/s/Hz)');
legend('Proposed GWO','Baseline (Random Phase)','Location','best');
title('Performance Comparison: GWO vs Baseline');
