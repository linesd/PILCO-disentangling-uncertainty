%% unicycle_learn.m
% *Summary:* Script to learn a controller for unicycling
%
% Copyright (C) 2008-2013 by
% Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
%
% Last modified: 2013-03-27
%
%% High-Level Steps
% # Load parameters
% # Create J initial trajectories by applying random controls
% # Controlled learning (train dynamics model, policy learning, policy
% application)

%% Code


% 1. Initialization
clear all; close all;
settings_unicycle;                     % load scenario-specific settings
basename = 'unicycle_';                % filename used for saving data

% 2. Initial J random rollouts
for jj = 1:J                                        % get the first observations
  [xx, yy, realCost{jj}, latent{jj}] = ...
    rollout(gaussian(mu0, S0), struct('maxU',policy.maxU/5), H, plant, cost);
  x = [x; xx]; y = [y; yy];
  if plotting.verbosity > 0;      % visualization of trajectory
    if ~ishandle(1); figure(1); else set(0,'CurrentFigure',1); end; clf(1);
    draw_rollout_unicycle;
  end
end

z(odei,:) = bsxfun(@plus, mu0, chol(S0)'*randn(length(odei),1000));   % compute
for i = 1:size(z,2), z(augi,i) = plant.augment(z(:,i)'); end % the distribution
mu0Sim = mean(z,2); S0Sim = cov(z');         % of augmented start state by MCMC
mu0Sim(odei) = mu0; S0Sim(odei,odei) = S0;        % Put in known correct values
mu0Sim = mu0Sim(dyno); S0Sim = S0Sim(dyno,dyno); clear z i;

uncertainty = nan(N, 4);

% 3. Controlled learning (N iterations)
for j = 1:N
  trainDynModel;
  learnPolicy;
  applyController;
  disp(['controlled trial # ' num2str(j)]);
  if plotting.verbosity > 0;      % visualization of trajectory
    if ~ishandle(1); figure(1); else set(0,'CurrentFigure',1); end; clf(1);
    draw_rollout_unicycle;
  end
  
   %% MY STUFF FROM HERE
 N_num = 100; % number of starts
 M_num = 100; % number of samples of weights
 T_num = H; % number of timesteps in rollout
 num_actions = 2;
 
% indices
unci = [1 2 3 4 5 6 7 8 9 10 12 13 14 15 17]; % get the y data
lossi = [5 6 7 8 9 11 12 13 14 15]; % losses
ang_in_target = [13 14 15];
ang_in_states = [13 14 15 16 17 18];
others_in_t_s = [1 2 3 4 5 6 7 8 9 10 11 12];
pol_states = [5 6 7 8 9 11 12]; % plus the thetas
% NEW Uncertainty training DATA
%  1  dx      x velocity
%  2  dy      y velocity
%  3  dxc     x velocity of origin (unicycle coordinates)
%  4  dyc     y velocity of origin (unicycle coordinates)
%  5  dtheta  roll angular velocity p
%  6  dphi    yaw angular velocity p
%  7  dpsiw   wheel angular velocity p
%  8  dpsif   pitch angular velocity p
%  9  dpsit   turn table angular velocity p
% 10  x       x position
% 11  xc      x position of origin (unicycle coordinates) p
% 12  yc      y position of origin (unicycle coordinates) p
% 13  sin(theta)   roll angle
% 14  cos(theta)   roll angle
% 15  sin(phi)     yaw angle
% 16  cos(phi)     yaw angle
% 17  sin(psif)    pitch angle
% 18  cos(psif)    pitch angle
% 19  ct      control torque for turn table
% 20  cw      control torque for wheel
 X_tr = [x(:,[1:10 12:13]) sin(x(:,14)) cos(x(:,14)) sin(x(:,15)) cos(x(:,15)) sin(x(:,17)) cos(x(:,17)) x(:,19:20)];

 %NEW TARGETS
%  1  dx      x velocity
%  2  dy      y velocity
%  3  dxc     x velocity of origin (unicycle coordinates)
%  4  dyc     y velocity of origin (unicycle coordinates)
%  5  dtheta  roll angular velocity p
%  6  dphi    yaw angular velocity p
%  7  dpsiw   wheel angular velocity p
%  8  dpsif   pitch angular velocity p
%  9  dpsit   turn table angular velocity p
% 10  x       x position
% 11  xc      x position of origin (unicycle coordinates) p
% 12  yc      y position of origin (unicycle coordinates) p
% 13  theta   roll angle p
% 14  phi     yaw angle p
% 15  psif    pitch angle p
 Y_tr = y(unci) - x(:, unci); % training differences
 
 % get some dimensions
 [x_len, dimx] = size(X_tr);
 [y_len, dimy] = size(Y_tr);

 Y_te = zeros(x_len, 1); % dummys which are not actually used
 X_te = rand(y_len, dimx); % dummys which are not actually used
 
 % hyper parameters
 loghyper = rand(dimx+2,1);
 nbf = 500; % number of basis functions
 % loop through the number of targets and get set of optimised hyperparams for each
 opt_params = zeros(dimx+2 + nbf*dimx, dimy);
 for ii = 1:dimy
     fprintf('Learning uncertainty model %i of %i ... \n', ii, dimy)
     [~, ~, ~, ~, loghyper, ~] = ssgprfixed_ui(X_tr, Y_tr(:, ii), X_te, Y_te, nbf, -1000, loghyper);
      opt_params(:,ii) = loghyper;
 end
 
% initialise trajectories
 trajectories = nan(M_num, T_num, N_num, dimx); % dims of x plus policy
 trajectory_costs = nan(M_num, T_num, N_num);
 
 % do Monte Carlo rollouts
 for mm =1:M_num
     
     % draw dimy samples of the weights
     weights = zeros(2*nbf, dimy);
     for ii = 1:dimy
         % posterior mu and cov only depend on data and not test data
         % which is just a dummy here.
         [mu_p, cov_p, ~] = ssgprfixed(opt_params(:,ii), X_tr, Y_tr(:, ii), X_te, true);
         weights(:,ii)  =  mvnrnd(mu_p', cov_p, 1);
     end
     
     % get N starting states
     states = zeros(N_num, 18); % should be N x 18
     thetas = zeros(N_num, 3);
     for nn = 1:N_num
         state = zeros(1, 18);
         state(odei) = gaussian(mu0, S0); 
         state(augi) = plant.augment(state);
         states(nn, :) = [state(1,[1:10 12:13]) sin(state(1,14)) cos(state(1,14)) ... 
                                sin(state(1,15)) cos(state(1,15)) sin(state(1,17)) cos(state(1,17))];
         thetas(nn, :) = state(1,[14 15 17]);  
     end
     
     %do the time rollouts
     for tt = 1:T_num
        
         % store the trajectories
         trajectories(mm, tt, :, :) = states;
         
         % calculate costs and actions for the N states
         actions = zeros(N_num, length(policy.maxU));
         pol_input = [states(:,pol_states) thetas];
         for nn = 1:N_num
            trajectory_costs(mm, tt, nn) = cost.fcn(cost,states(nn,lossi)',zeros(length(lossi))); % from rollout
            actions(i,:) = policy.fcn(policy,pol_input(nn, :),zeros(length(pol_states)+3));
         end
                  
         % augment states and actions and save them
         states_actions = [states actions];
                  
         % predict one step from the posterior to get delta states
         delta_states = zeros(N_num, dimy);
         for ii = 1:dimy 
             [~, ~, phistar] = ssgprfixed(opt_params(:,ii), X_tr, Y_tr(:, ii), states_actions, true); % dyni + controls
             delta_states(:, ii) = phistar * weights(:,ii);
         end
         
         % get next states
         next_states = zeros(N_num, xdim);
         next_states(:, others_in_t_s) = states(:, others_in_t_s) + delta_states(:, others_in_t_s);
         next_thetas = thetas + delta_states(:, ang_in_target);
         next_states(:, ang_in_states) = [sin(thetas(:,1)) cos(thetas(:,1)) sin(thetas(:,2)) ...
                                                          cos(thetas(:,2)) sin(thetas(:,3)) cos(thetas(:,3))];
         
         % make next current
         states = next_states;
     end
     fprintf('Finished %s of %s MC rollouts... \n', mm*tt*N_num, M_num*T_num*N_num);
 end
 
 %compute the average cost for the episode
 ave_cost_for_ep = mean(reshape(trajectory_costs,1,M_num*N_num*T_num));
 
 % disentangle uncertainty trajectory costs
 total = var(reshape(trajectory_costs,1,M_num*N_num*T_num), 1);
 epistemic = var(mean(reshape(trajectory_costs,M_num, T_num*N_num),2));
 aleatoric = mean(var(reshape(trajectory_costs,M_num, T_num*N_num),1,2));
 uncertainty(j, :) = [total epistemic aleatoric ave_cost_for_ep];
 
 % save all the data
 name = strcat('/homes/dl598/thesis_experiments/pilcoV0.9/cartDoublePen_plots/data_1/X_data_', num2str(j));
 save(name, 'X_tr')
 name = strcat('/homes/dl598/thesis_experiments/pilcoV0.9/cartDoublePen_plots/data_1/Y_data_', num2str(j));
 save(name, 'Y_tr')
 name = strcat('/homes/dl598/thesis_experiments/pilcoV0.9/cartDoublePen_plots/data_1/trajectory_costs_', num2str(j));
 save(name, 'trajectory_costs')
 name = strcat('/homes/dl598/thesis_experiments/pilcoV0.9/cartDoublePen_plots/data_1/trajectories_', num2str(j));
 save(name, 'trajectories');
 name = strcat('/homes/dl598/thesis_experiments/pilcoV0.9/cartDoublePen_plots/data_1/uncertainty_', num2str(j));
 save(name, 'uncertainty');
 name = strcat('/homes/dl598/thesis_experiments/pilcoV0.9/cartDoublePen_plots/data_1/fantasy_data_', num2str(j));
 save(name, 'fantasy');
  
end
