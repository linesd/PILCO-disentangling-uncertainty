 %% Monte Carlo Uncertainty
tic;

% get data
[~, state_len] = size(x);  
X_tr = dynmodel.inputs;
Y_tr = dynmodel.targets;
[x_len, dimx] = size(X_tr);
[y_len, dimy] = size(Y_tr);
Y_te = zeros(x_len, 1); % dummys which are not actually used
X_te = rand(y_len, dimx); % dummys which are not actually used

% hyper parameters
loghyper = rand(dimx+2,1);

% loop through the number of targets and get set of optimised hyperparams for each
opt_params = zeros(dimx+2 + nbf*dimx, dimy);
for ii = 1:dimy
    fprintf('Learning uncertainty model %i of %i ... \n', ii, dimy);
    [~, ~, ~, ~, loghyper, ~] = ssgprfixed_ui(X_tr, Y_tr(:, ii),  X_te, Y_te, nbf, -1000, loghyper);
    opt_params(:,ii) = loghyper;
end

% initialise trajectories
trajectories = nan(M_num, T_num, N_num, state_len);
trajectory_costs = nan(M_num, T_num, N_num);

% do Monte Carlo rollouts
parfor mm =1:M_num
    % draw dimy samples of the weights
    weights = zeros(2*nbf, dimy);
    for ii = 1:dimy
        % posterior mu and cov only depend on data and not test data
        % which is just a dummy here.
        [mu_p, cov_p, ~] = ssgprfixed(opt_params(:,ii), X_tr, Y_tr(:, ii), X_te, true);
        weights(:,ii)  =  mvnrnd(mu_p', cov_p, 1);
    end

    % get N starting states
    states = zeros(N_num, state_len); % should be N x 5
    for ii = 1:N_num
        states(ii, dyno)  = mvnrnd(mu0', S0 , 1);
        states(ii, 5:6) = [sin(states(ii, angi)) cos(states(ii, angi))];
    end

    %do the time rollouts
    for tt = 1:T_num

        % calculate costs and actions for the N states
        for nn = 1:N_num
            trajectory_costs(mm, tt, nn) = cost.fcn(cost, states(nn,dyno)',zeros(length(dyno))); % from rollout
            states(nn,end) = policy.fcn(policy,states(nn,poli)',zeros(length(poli)));% append policy
        end

        % store the state-actions
        trajectories(mm, tt, :, :) = states;      

        % predict one step from the posterior to get delta states
        delta_states = zeros(N_num, dimy);
        for ii = 1:dimy
            [~, ~, phistar] = ssgprfixed(opt_params(:,ii), X_tr, Y_tr(:, ii), states(:,[dyni acti]), true);
            delta_states(:, ii) = phistar * weights(:,ii); 
        end

        % update states
        states(:, dyno) = states(:, dyno) + delta_states + randn(size(dyno))*chol(plant.noise);
        states(:, 5:6) = [sin(states(:, angi)) cos(states(:, angi))] ;
    end

    fprintf('Finished %i of %i MC rollouts... \n', mm*tt*N_num, M_num*T_num*N_num);

end

% disentangle uncertainty trajectory costs
total = sum(var(trajectory_costs, 1, [1,2]));
epistemic = sum(var(mean(trajectory_costs, 2)));
aleatoric = sum(mean(var(trajectory_costs, 1, 2)));
uncertainty(j, :) = [total epistemic aleatoric];

% save all the data

name = basename + "_X_data_" + num2str(j); 
save(name, "X_tr")
name = basename + "_Y_data_" + num2str(j); 
save(name, "Y_tr")
name = basename +  "_trajectory_costs_" + num2str(j); 
save(name, "trajectory_costs")
name = basename +  "_trajectories_" + num2str(j); 
save(name, "trajectories");
name = basename +  "_uncertainty_" + num2str(j); 
save(name, "uncertainty");
name = basename +  "_fantasy_data_" + num2str(j); 
save(name, "fantasy");

fprintf("MC rollouts took %.2f seconds.\n",toc); 