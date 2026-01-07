%% rest of the code
clc
clear 
close all

load('iddata-03.mat');

id_X = id_array(:, 2); 
id_Y = id_array(:, 3); 

val_X = val_array(:, 2);
val_Y = val_array(:, 3);

figure(1)
subplot(2,1,1); plot(id_X); title('ID input'); grid on;
subplot(2,1,2); plot(id_Y); title('ID output'); grid on;

na_range = 1:3;
m_range = 1:3;

results = []; %to store na, m, mse_min_val and mse_pred_val
cnt = 0;
total_iter = length(na_range) * length(m_range);

for na = na_range
    nb=na;
    for m = m_range
        cnt = cnt+1;
        [theta, exponents] = train_narx(id_X, id_Y, na, nb, m);

        y_sim = predict_narx(val_X, val_Y, theta, exponents, na, nb, 'simulation');
        mse_sim_val = mean((val_Y(na+1:end) - y_sim(na+1:end)).^2);
        
        y_pred = predict_narx(val_X, val_Y, theta, exponents, na, nb, 'prediction');
        mse_pred_val = mean((val_Y(na+1:end) - y_pred(na+1:end)).^2);

        y_sim_id = predict_narx(id_X, id_Y, theta, exponents, na, nb, 'simulation');
        mse_sim_id = mean((id_Y(na+1:end) - y_sim_id(na+1:end)).^2);

        y_pred_id = predict_narx(id_X, id_Y, theta, exponents, na, nb, 'prediction');
        mse_pred_id = mean((id_Y(na+1:end) - y_pred_id(na+1:end)).^2);

        results = [results; na, m, mse_sim_id, mse_pred_id, mse_sim_val, mse_pred_val];

    end
end

[min_mse, idx_best] = min(results(:, 5));
best_na = results(idx_best, 1);
best_nb = best_na;
best_m = results(idx_best, 2);

[theta_best, exponents_best] = train_narx(id_X, id_Y, best_na, best_nb, best_m);

y_sim_val_best = predict_narx(val_X, val_Y, theta_best, exponents_best, best_na, best_na, 'simulation');
y_pred_val_best = predict_narx(val_X, val_Y, theta_best, exponents_best, best_na, best_na, 'prediction');

y_sim_id_best = predict_narx(id_X, id_Y, theta_best, exponents_best, best_na, best_na, 'simulation');
y_pred_id_best = predict_narx(id_X, id_Y, theta_best, exponents_best, best_na, best_na, 'prediction');

figure(2)
plot(results(:,3), 'b--s'); hold on; 
plot(results(:,4), 'r--d'); 
plot(results(:,5), 'b-o');
plot(results(:,6), 'r-x');
legend('MSE Sim ID', 'MSE Pred ID', 'MSE Sim VAL', 'MSE Pred VAL');
title('Error Comparison');

% simulation plot
figure(3)
start_p = best_na + 1;
subplot(2,1,1);
plot(id_Y(start_p:end), 'k'); hold on; plot(y_sim_id_best(start_p:end), 'r--');
legend('Real-ID', 'Sim-ID'); title('Simulation on ID Data'); grid on;
subplot(2,1,2);
plot(val_Y(start_p:end), 'k'); hold on; plot(y_sim_val_best(start_p:end), 'r--');
legend('Real-VAL', 'Sim-VAL'); title('Simulation on VAL Data'); grid on;

% prediction plot
figure(4)
subplot(2,1,1);
plot(id_Y(start_p:end), 'k'); hold on; plot(y_pred_id_best(start_p:end), 'g--', 'LineWidth', 1);
legend('Real-ID', 'Pred-ID'); title('Prediction on ID Data'); grid on;
subplot(2,1,2);
plot(val_Y(start_p:end), 'k'); hold on; plot(y_pred_val_best(start_p:end), 'g--');
legend('Real-VAL', 'Pred-VAL'); title('Prediction on VAL Data'); grid on;

%% important functions
function [theta, exponents] = train_narx(u, y, na, nb, m)
    N = length(y);
    n_vars = na + nb;

    exponents = get_exp(n_vars, m);

    start_k = max(na, nb) + 1;
    num_samples = N - start_k + 1;
    num_regressors = size(exponents, 1);
    
    Phi = zeros(num_samples, num_regressors);
    target = y(start_k:end);

    idx = 1;
    for k = start_k:N
        past_y = y(k-1:-1:k-na)';
        past_u = u(k-1:-1:k-nb)';
        
        vars = [past_y, past_u];

        Phi(idx, :) = calculate_poly(vars, exponents);
        idx = idx + 1;
    end
    theta = Phi\target;
end

function y_out = predict_narx(u, y, theta, exponents, na, nb, sim_type)
    N = length(y);
    start_k = max(na, nb) + 1;
    y_out = zeros(N, 1);
    
    y_out(1:start_k-1) = y(1:start_k-1);
    
    for k = start_k:N
        if strcmp(sim_type, 'simulation')
            past_y = y_out(k-1:-1:k-na)';
        else
            past_y = y(k-1:-1:k-na)';
        end
        past_u = u(k-1:-1:k-nb)';
        vars = [past_y, past_u];
        
        phi_row = calculate_poly(vars, exponents);
        y_out(k) = phi_row * theta;
    end
end

function row = calculate_poly(x, exponents)
    term_matrix = x .^ exponents;
    row = prod(term_matrix, 2)';
end

function exps = get_exp(n_vars, max_deg)
    if n_vars == 1
        exps = (0:max_deg)';
    else
        sub_exps = get_exp(n_vars-1, max_deg);
        exps = [];
        for i = 1:size(sub_exps, 1)
            current_sum = sum(sub_exps(i,:));
            remaining = max_deg - current_sum;
            for k = 0:remaining
                exps = [exps; sub_exps(i,:), k];
            end
        end
    end
end

