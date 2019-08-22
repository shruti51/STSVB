function Result=stsvb_demo(Phi,Y,blk_structure,learn_alpha,status,max_iter)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STSVB recovers block sparse signal using matrix variate gaussian scale mixture
% parameterized by some scalar random parameters and deterministic matrices
% to model spatio-temporal correlation.

% ==========INPUTS=============
% Phi: M X N Gaussian matrix
% Y: Observation matrix
% blk_structure: block partition of unknown solution matrix X. (For physiological signals, 
%                block partition does not need to be consistent with the true block partition.)
% learn_alpha: (1) If learn_alpha = 0, algorithm used will be STSVB-J.
%              (2) If learn_alpha = 1, algorithm used will be STSVB-L.
%              (3) If learn_alpha = 2, algorithm used will be STSVB-St.
% status: (1) If status = 0, we use noiseless setting where noise variance parameter beta is not learned.
%         (2) If status = 1, we use mildly noisy setting where SNR of signal is 25dB.
%         (3) If status = 2, we use low SNR noise update rules.
% max_iter: Number of iterations required to terminate the algorithm.

% ==========OUTPUTS==============
% Result: 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Scaling
scale=mean(std(Y));
if (scale<0.4) || (scale>1)
    Y=Y/scale*0.4;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Problem dimensions
[M,N]=size(Phi);
L=size(Y,2);

% Initial parameter values
epsilon=1e-6;
lambda = 1;

if status==0 % noiseless case
    prune_alpha=1e+10;
    beta=1e+12;
    matrix_reg=zeros(L);
elseif status==1 % high SNR case
    prune_alpha=1e+3;
    beta=1e+3;
    matrix_reg=eye(L)*0.5;
elseif status==2 % low SNR case
    prune_alpha=1e+2;
    beta=1e+3;
    matrix_reg=eye(L)*2;
end

Y_0 = Y;
Phi_0 = Phi;
blk_structure_0 = blk_structure;
g = length(blk_structure);
blk_length_list = zeros(1,g);         % Number of blocks 
k_a = 1e-6; theta_a = 1e-6; a_0 = 1; a = a_0 * ones(g,1);
k_beta = 1e-6; theta_beta = 1e-6; 
k_b = 1e-6; theta_b = 1e-6; b_0 = 1; b = b_0 * ones(g,1);

% Block Partitioning and size information
for k = 1 : g-1
    blk_length_list(k) = blk_structure(k+1) - blk_structure(k);
end
blk_length_list(g) = N - blk_structure(end) + 1;
max_blk_length = max(blk_length_list);
 
if sum(blk_length_list == max_blk_length) == g
    equal_blk_size = 1;
else
    equal_blk_size = 0;
end

% Covariance Matrix initialization 
PI_inv = cell(g);  
for k = 1 : g
    PI_inv{k} = eye(blk_length_list(k));
end
B = eye(L);

% Scalar random parameter initialization
alpha = ones(g,1); alpha_inv = ones(g,1);
list = 1:g;       % alpha's list
usable_list = length(list);
count = 0;

% Solution Matrix initialization
Mu_t = zeros(N,L);   


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% ITERATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while (1)
  
    count = count + 1;
    
    % Inducing sparsity in solution matrix by pruning out some alpha's.
    if (max(alpha) > prune_alpha)
        index = find(alpha < prune_alpha);
        usable_list = length(index);
        list = list(index);
        blk_structure = blk_structure(index);
        blk_length_list = blk_length_list(index);
        alpha = alpha(index);
        temp = PI_inv;
        PI_inv = [];
        for k = 1 : usable_list
            PI_inv{k} = temp{index(k)};
        end
        % Construct new Phi
        temp = [];
        for k = 1 : usable_list
            temp = [temp, Phi_0(:,blk_structure(k):blk_structure(k)+blk_length_list(k)-1)];
        end
        Phi = temp;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%% Spatially Whitening %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    Y = Y_0 * sqrtm(inv(B));
    
    % Phi*PI*Phi^t implementation
    Phi_PIinv_Phi = zeros(M);
    current_loc = 0;
    for i = 1 : usable_list
        current_blk_len = size(PI_inv{i},1);
        current_loc = current_loc + 1;
        segment = current_loc : current_loc + current_blk_len - 1;
        Phi_PIinv_Phi = Phi_PIinv_Phi + Phi(:,segment) * PI_inv{i} * Phi(:,segment)';
        current_loc = segment(end);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%% Signal Estimate %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    H = Phi' / (1/beta * eye(M) + Phi_PIinv_Phi);
    H_y = H * Y;
    H_Phi = H * Phi;
    M_x = zeros(size(Phi,2),L);
    Sigma_x = []; current_loc = 0;
    Tau = []; Tau_inv=[];
    r_0 = 0; r_1 = 0; Tau_inv_0 = zeros(max_blk_length);
    for i = 1 : usable_list
        current_blk_len = size(PI_inv{i},1);
        current_loc = current_loc + 1;
        segment = current_loc : current_loc + current_blk_len - 1;
        current_loc = segment(end);
        M_x(segment,:) = PI_inv{i} * H_y(segment,:);
        Sigma_x{i} = PI_inv{i} - PI_inv{i} * H_Phi(segment,segment) * PI_inv{i};
        
        % Learning Tau_inv_i's (intra-block correlation within columns)
        if equal_blk_size == 0
           if current_blk_len > 1
              temp = Sigma_x{i} * alpha(i) + M_x(segment,:) * M_x(segment,:)' * alpha(i)/L;
              r_0 = r_0 + mean(diag(temp));
              r_1 = r_1 + mean(diag(temp,1));
           end
        else
              Tau_inv_0 = Tau_inv_0 + Sigma_x{i} * alpha(i) + M_x(segment,:) * M_x(segment,:)' * alpha(i)/L;
        end        
    end
    
    % Intra-block correlation with regularization
    if equal_blk_size == 1      % If blocks have the same size
        % Constraining the blocks to have the same correlation structure
        r = mean( diag(Tau_inv_0,1))/mean(diag(Tau_inv_0));
        if abs(r) >= 0.99
            r = sign(r) * 0.99;
        end
        Tau_hat = [];
        for j = 1 : max_blk_length
            Tau_hat(j) = r^(j-1);
        end
        Tau_inv_0 = toeplitz(Tau_hat);
        for i = 1 : usable_list
            Tau_inv{i} = Tau_inv_0;
            Tau{i} = inv(Tau_inv{i});
        end
        
    elseif equal_blk_size == 0 % if blocks have different sizes
        r = r_1/r_0;
        if abs(r) >= 0.99
            r = 0.99 * sign(r);
        end
        for i = 1 : usable_list
            current_blk_len = size(Sigma_x{i},1);
            Tau_hat = [];
            for j = 1 : current_blk_len
                Tau_hat(j) = r^(j-1);
            end
            Tau_inv{i} = toeplitz(Tau_hat);
            Tau{i} = inv(Tau_inv{i});
        end
    end
    
    % To estimate alpha_i and beta parameter
    
    alpha_old = alpha;
    current_loc = 0;
    if status~=0 % Noisy Case
        beta_comp = 0;
    end   
    for i = 1 : usable_list
        current_blk_len = size(Sigma_x{i},1);
        current_loc  = current_loc + 1;
        segment = current_loc : current_loc + current_blk_len  - 1;
        current_loc = segment(end);

        if learn_alpha == 0
            
            alpha(i) = (current_blk_len * L)/(L * trace(Tau{i} * Sigma_x{i}) + sum( sum( (M_x(segment,:)' * Tau{i}) .* M_x(segment,:)', 2)));
            
        elseif learn_alpha == 1
            
            alpha(i) = sqrt(b(i)) / (sqrt(L * trace(Tau{i} * Sigma_x{i}) + sum( sum( (M_x(segment,:)' * Tau{i}) .* M_x(segment,:)', 2))));
            alpha_inv(i) = 1/alpha(i) + 1/b(i);
            b(i) = (k_b + (current_blk_len * L + 1)/2) / (theta_b + alpha_inv(i)/2);
            
        else
            
            alpha(i) = (2 * lambda + current_blk_len * L )/(a(i) + L * trace(Tau{i} * Sigma_x{i}) + sum( sum( (M_x(segment,:)' * Tau{i}) .* M_x(segment,:)', 2)));
            a(i) = (k_a + lambda) / (theta_a + alpha(i)/2);
            
        end
        
        PI_inv{i} = Tau_inv{i} / alpha(i);
        
        if status == 1
            beta_comp = beta_comp + trace(Sigma_x{i} * Tau{i}) * alpha(i);
        elseif status == 2
            beta_comp = beta_comp + trace(Phi(:,segment) * Sigma_x{i} * Phi(:,segment)');
        end
    end
    
    if status == 1
        beta=(M*L + 2*k_beta)/(2*theta_beta + norm(y-Phi*M_x,'fro')^2+ (1/beta)*(length(M_x) * L - beta_comp * L ));
    elseif status == 2
        beta=(M*L + 2*k_beta)/(2*theta_beta + norm(y-Phi*M_x,'fro')^2+ beta_comp * L);
    end
    
    % learn spatial (inter-column) covariance matrix B
    Mu_old = Mu_t;
    Mu_t = M_x*sqrtm(B);
    B = zeros(L);
    current_loc = 0;
    
    for i = 1 : usable_list
        current_loc = current_loc + 1;
        current_blk_len = size(PI_inv{i},1);
        segment = current_loc : current_loc + current_blk_len - 1;
        current_loc = segment(end);
        B = B + Mu_t(segment,:)' * Tau{i} * Mu_t(segment,:) * alpha_old(i);
    end
    
    B = B / norm(B,'fro');
    B = B + matrix_reg;
    B = B ./ norm(B);
    if size(Phi,2) <= size(Phi,1)
        B = eye(L);
    end
    
    % Stopping conditions
    if (size(M_x) == size(Mu_old))
        d_Mu = max(max(abs(Mu_t - Mu_old)));
        %  disp(['difference in Mu is : ', num2str(d_Mu)])
        if (d_Mu <= epsilon)
            break;
        end
    end
    if (count >= max_iter)
        %disp('Reached max iteration. Stop \n \n');
        break;
    end
end
% Hyperparameters 
alpha_used = sort(list);
alpha_est = zeros(g,1);
alpha_est(list,1) = alpha;

% Reconstructed signal
X_hat = zeros(N,L);
current_loc = 0;

for i = 1 : usable_list
    current_blk_len = size(PI_inv{i},1);
    current_loc = current_loc + 1;
    segment = current_loc : current_loc + current_blk_len - 1;
    
    real_locations = blk_structure_0(list(i)) : blk_structure_0(list(i)) + current_blk_len - 1;
    X_hat(real_locations,:) = Mu_t(segment,:);
    current_loc = segment(end);
end

% rescaling the signal
if (scale < 0.4) || (scale > 1)
    Result.x = X_hat * scale / 0.4;
else
    Result.x = X_hat;
end

Result.alpha_used = alpha_used;
Result.alpha_est = alpha_est;
Result.Tau = Tau_inv;
Result.B = B;
Result.count = count;
Result.beta = beta;
return;

