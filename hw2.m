% for functions and parameters
A = 2;
r = 0.5;
mu = @(x, y) 1 + A * exp(-(x-0.5).^2 / (2*r^2));
f = @(x, y) -(mu(x, y)*(2*y.*(1-y))+(1-2*x).*y.*(1-y).*(mu(x, y)-1).*(-(x-0.5)/r^2)+mu(x, y)*(-2*x.*(1-x)));
u = @(x, y) x .* y .* (1 - x) .* (1 - y);
l = 0;
h = 1;
l_n = 2;
r_n = 20;

[~, ~, center_errors, L1_errors, L_inf_errors] = solve_pde(A, r, l, h, l_n, r_n, mu, f, u);


function [U, updates, center_errors, L1_errors, L_inf_errors] = solve_pde(A, r, l, h, l_n, h_n, mu, f, u_func)
    ns = l_n:h_n;
    center_errors = zeros(size(ns));
    L1_errors = zeros(size(ns));
    L_inf_errors = zeros(size(ns));
    delta_area = ((h-1e-9) - l)^2 / 1000;

    for n = ns
        [U, updates] = run(mu, f, l, h, n);
        delta_x = (h-l) / n;

        U_True = get_block_value(u_func, n);
        U_True = reshape(U_True, [n, n]);

        center_error = mean(mean(abs(U-U_True)));

        % Construct a finer mesh to evaluate L1 and L_inf error
        x_grid = linspace(l, h-1e-9, 1000);
        [x_mesh, y_mesh] = meshgrid(x_grid, x_grid);
        U_true_fine = u_func(x_mesh, y_mesh);

        coarse_grid = linspace(l, h, n+1);
        coarse_grid = coarse_grid + delta_x/2;
        coarse_grid = coarse_grid(1:end-1);

        [coarse_mesh_x, coarse_mesh_y] = meshgrid(coarse_grid, coarse_grid);
        interpolate = griddedInterpolant(coarse_mesh_x', coarse_mesh_y', U');
        U_fine = interpolate(x_grid, x_grid);

        L1_error = sum(sum(abs(U_true_fine - U_fine))) * delta_area;
        L_inf_error = max(max(abs(U_true_fine - U_fine)));

        center_errors(n-l_n+1) = center_error;
        L1_errors(n-l_n+1) = L1_error;
        L_inf_errors(n-l_n+1) = L_inf_error;

        fprintf('n=%d, center_error=%f, L1_error=%f, L_inf_error=%f\n', n, center_error, L1_error, L_inf_error);
    end
end

function [U, updates] = run(mu, f, l, h, n)
    delta_x = (h - l) / n;
    A = zeros(n^2, n^2);

    % Construct A
    for i = 1:n^2
        row = floor((i-1) / n) + 1;
        col = mod(i-1, n) + 1;

        mu_l = mu(row, col-0.5);
        mu_r = mu(row, col+0.5);
        mu_u = mu(row+0.5, col);
        mu_d = mu(row-0.5, col);

        A(i, i) = 2 * mu_l;

        if col > 1
            A(i, i-1) = -mu_l;
        end

        if col < n
            A(i, i+1) = -mu_r;
        end

        if row > 1
            A(i, i-n) = -mu_d;
        end

        if row < n
            A(i, i+n) = -mu_u;
        end
    end

    A = A / delta_x^2;

    F = get_block_value(f, n);
    F = reshape(F, [], 1);

    % Gauss Seidel method
    U = zeros(n^2, 1);
    updates = zeros(1, 1e5);
    thres = 1e-5;
    prev_U = U;

    for k = 1:1e5
        update = 0;

        for j = 1:length(F)
            R = F(j) - A(j, :) * U + A(j, j) * U(j);
            new_val = R / A(j, j);
            update = update + abs(new_val - U(j));
            U(j) = new_val;
        end

        updates(k) = update / length(F);

        % Check convergence based on relative change
        if norm(U - prev_U) / norm(U) < thres
            break;
        end

        prev_U = U;
    end

    U = reshape(U, [n, n]);
    updates = updates(1:k);
end

function block = get_block_value(func, n)
    block = zeros(n, n);

    for i = 1:n
        for j = 1:n
            block(i, j) = func(i, j);
        end
    end
end