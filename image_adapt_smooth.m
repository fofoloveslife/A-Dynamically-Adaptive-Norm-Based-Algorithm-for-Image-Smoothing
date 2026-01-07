function S = image_adapt_smooth(X, lambda, p, tau, sigma)

% default parameters
if ~exist('lambda', 'var') || isempty(lambda), lambda = 0.2; end
if ~exist('p', 'var') || isempty(p), p = 0.8; end
if ~exist('tau', 'var') || isempty(tau), tau = 1; end
if ~exist('sigma', 'var') || isempty(sigma), sigma = 0.1; end

p_final = p;         
p_init = 0.99;       
sigma_init = 5;

MAX_ITER = 50;      
rho = 1;            
epsilon = 1e-5;      

X = im2double(X);
[row, col, cha] = size(X);

S = X; 
Lx = zeros(row, col, cha);
Ly = zeros(row, col, cha);
dx = zeros(row, col, cha);
dy = zeros(row, col, cha);

fx = [1, -1]; 
fy = [1; -1];
sizeI2D = [row, col];

otfFx = psf2otf(fx, sizeI2D);
otfFy = psf2otf(fy, sizeI2D);

Denormin = 1 + rho * (abs(otfFx).^2 + abs(otfFy).^2);
if cha > 1 
    Denormin = repmat(Denormin, [1, 1, cha]); 
end

Normin_X = fft2(X);

for k = 1:MAX_ITER
    % --- Step 1: S-update
    term_x = Lx - dx;
    term_y = Ly - dy;
    
    Normin_grad = [term_x(:,end,:) - term_x(:,1,:), -diff(term_x, 1, 2)];
    Normin_grad = Normin_grad + [term_y(end,:,:) - term_y(1,:,:); -diff(term_y, 1, 1)];
    
    FS = (Normin_X + rho * fft2(Normin_grad)) ./ Denormin;
    S = real(ifft2(FS));

    % --- Step 2: L-update 
    h = [diff(S, 1, 2), S(:,1,:) - S(:,end,:)];
    v = [diff(S, 1, 1); S(1,:,:) - S(end,:,:)];
    
    progress = k / MAX_ITER;
    
    sigma_current = sigma + (sigma_init - sigma) * (1 - progress);
    
    p_current = p_final + (p_init - p_final) * (1 - progress);

    grad_L2_sq_x = sum(h.^2, 3);
    grad_L2_sq_y = sum(v.^2, 3);
    denominator = 2 * sigma_current^2; 
    w_spatial_x = exp(-grad_L2_sq_x / denominator);
    w_spatial_y = exp(-grad_L2_sq_y / denominator);
    if cha > 1
        w_spatial_x = repmat(w_spatial_x, [1, 1, cha]);
        w_spatial_y = repmat(w_spatial_y, [1, 1, cha]);
    end

    grad_abs_x = abs(Lx) + epsilon;
    num_w_x = (1 + tau) * grad_abs_x.^(p_current - 1); 
    den_w_x = tau + grad_abs_x.^(p_current - 1);       
    w_nonconvex_x = num_w_x ./ den_w_x;
    
    grad_abs_y = abs(Ly) + epsilon;
    num_w_y = (1 + tau) * grad_abs_y.^(p_current - 1); 
    den_w_y = tau + grad_abs_y.^(p_current - 1);      
    w_nonconvex_y = num_w_y ./ den_w_y;
   
    threshold_x = (lambda / rho) * w_spatial_x .* w_nonconvex_x;
    threshold_y = (lambda / rho) * w_spatial_y .* w_nonconvex_y;
    
    Lx = max(abs(h + dx) - threshold_x, 0) .* sign(h + dx);
    Ly = max(abs(v + dy) - threshold_y, 0) .* sign(v + dy);

    % --- Step 3: d-update
    dx = dx + h - Lx;
    dy = dy + v - Ly;
end

end