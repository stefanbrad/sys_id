clc
clear
close all

data = load("proj_fit_10.mat");

x1_id = data.id.X{1};
x2_id = data.id.X{2};
y = data.id.Y;

figure(1)
mesh(x1_id,x2_id,y);

x1_v = data.val.X{1};
x2_v = data.val.X{2};
yv = data.val.Y;

figure(2)
mesh(x1_v,x2_v,yv);

n1_id = length(x1_id);
n2_id = length(x2_id);
x_id = zeros(n1_id*n2_id, 2);
idx = 1;
for j = 1:n2_id
    for i = 1:n1_id
        x_id(idx, :) = [x1_id(i), x2_id(j)];
        idx = idx + 1;
    end
end
y_id = y(:);

n1_val = length(x1_v);
n2_val = length(x2_v);
x_val = zeros(n1_val*n2_val, 2);
idx = 1;
for j = 1:n2_val
    for i = 1:n1_val
        x_val(idx, :) = [x1_v(i), x2_v(j)];
        idx = idx + 1;
    end
end
y_val = yv(:);

max_degree = 20;
mse_id = zeros(max_degree,1);
mse_val = zeros(max_degree,1);
degrees = 1:max_degree;

for m = degrees
Phi_id  = generate_regressors(x_id,  m);
Phi_val = generate_regressors(x_val, m);

theta = Phi_id\y_id;
y_pred_id = Phi_id*theta;
y_pred_val = Phi_val*theta;


mse_id(m) = mean((y_id - y_pred_id).^2);
mse_val(m) = mean((y_val - y_pred_val).^2);

m
mse_id(m)
mse_val(m)

end

figure(3)
plot(degrees, mse_id, 'b-o');
hold on
plot(degrees, mse_val, 'r-s');
hold off


[min_mse_val, optimal_m] = min(mse_val);
optimal_m
min_mse_val
mse_id(optimal_m)

Phi_id_best = generate_regressors(x_id, optimal_m);
Phi_val_best = generate_regressors(x_val, optimal_m);

theta_best = Phi_id_best\y_id;

y_pred_id_best = Phi_id_best*theta_best;
y_pred_val_best = Phi_val_best*theta_best;

y_pred_id = reshape(y_pred_id_best, n1_id, n2_id);
y_pred_val = reshape(y_pred_val_best, n1_val, n2_val);

figure(4)
mesh(x1_id, x2_id, y);
hold on
surf(x1_id, x2_id, y_pred_id);
hold off

figure(5)
mesh(x1_v, x2_v, yv);
hold on
surf(x1_v, x2_v, y_pred_val);
hold off
