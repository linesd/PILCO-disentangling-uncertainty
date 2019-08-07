%% cartDouble_learn.m
% *Summary:* Script to learn a controller for the cart-doube-pendulum
% swingup
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
% settings_cdp;                 % load scenario-specific settings
basename = 'CartDoublePend_'; % filename used for saving data

% the data_# to save at
save_at = 1;

% % 2. Initial J random rollouts
% for jj = 1:J                                        
%   [xx, yy, realCost{jj}, latent{jj}] = ...
%     rollout(gaussian(mu0, S0), struct('maxU',policy.maxU), H, plant, cost);
%   x = [x; xx]; y = [y; yy];       % augment training sets for dynamics model
%   if plotting.verbosity > 0;      % visualization of trajectory
%     if ~ishandle(1); figure(1); else set(0,'CurrentFigure',1); end; clf(1);
%     draw_rollout_cdp; 
%   end 
%                    
% end
% 
% mu0Sim(odei,:) = mu0; S0Sim(odei,odei) = S0;
% mu0Sim = mu0Sim(dyno); S0Sim = S0Sim(dyno,dyno);
load('CartDoublePend_55_H100.mat');
New_N = 80;
uncertainty = [uncertainty; nan(N, 4)];

% 3. Controlled learning (N iterations)
for j = N+16:New_N
  fprintf("Running episode %i of %i. \n", j, New_N); tic;
  trainDynModel;   % train (GP) dynamics model
  fprintf("trainDynModel took %.2f seconds.\n",toc); tic;
  learnPolicy;     % learn policy 
  fprintf("learnPolicy took %.2f seconds.\n",toc); tic;
  applyController; % apply controller to system
  fprintf("applyController took %.2f seconds.\n",toc); 
  disp(['controlled trial # ' num2str(j)]);
  if plotting.verbosity > 0;      % visualization of trajectory
    if ~ishandle(1); figure(1); else set(0,'CurrentFigure',1); end; clf(1);
    draw_rollout_cdp; 
  end 
  
  %% MY STUFF FROM HERE
  tic;
  N_num = 100; % number of trajectory starts
  M_num = 100; % number of sets of weights
  T_num = H; % number of timesteps in rollout
  
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
  nbf = 500; % number of basis functions
  
  % loop through the number of targets and get set of optimised hyperparams for each
  opt_params = zeros(dimx+2 + nbf*dimx, dimy);
  for ii = 1:dimy
      fprintf("Learning uncertainty model %i of %i ... \n", ii, dimy)
      [~, ~, ~, ~, loghyper, ~] = ssgprfixed_ui(X_tr, Y_tr(:, ii), X_te, Y_te, nbf, -1000, loghyper);
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
          states(ii, 7:10) = [sin(states(ii, 5)) cos(states(ii, 5)) sin(states(ii, 6)) cos(states(ii, 6))] ;
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

          % update states + measurement noise
          states(:, dyno) = states(:, dyno) + delta_states + randn(size(dyno))*chol(plant.noise);
          states(:, 7:10) = [sin(states(:, 5)) cos(states(:, 5)) sin(states(:, 6)) cos(states(:, 6))];       
      end
      
      fprintf("Finished %i of %i MC rollouts... \n", mm*tt*N_num, M_num*T_num*N_num);
  end
  
  %compute the average cost for the episode
  ave_cost_for_ep = mean(trajectory_costs, [1,2,3]);
    
  % disentangle uncertainty trajectory costs
  total = var(trajectory_costs, 1, [1,2,3]);
  epistemic = var(mean(trajectory_costs, [2,3]));
  aleatoric = mean(var(trajectory_costs, 1, [2,3]));
  uncertainty(j, :) = [total epistemic aleatoric ave_cost_for_ep];
  
  % save all the data
  name = "../../myData/cartDoublePen_plots/data_" + num2str(save_at) +"/X_data_" + num2str(j); 
  save(name, "X_tr")
  name = "../../myData/cartDoublePen_plots/data_" + num2str(save_at) +"/Y_data_" + num2str(j); 
  save(name, "Y_tr")
  name = "../../myData/cartDoublePen_plots/data_" + num2str(save_at) +"/trajectory_costs_" + num2str(j); 
  save(name, "trajectory_costs")
  name = "../../myData/cartDoublePen_plots/data_" + num2str(save_at) +"/trajectories_" + num2str(j); 
  save(name, "trajectories");
  name = "../../myData/cartDoublePen_plots/data_" + num2str(save_at) +"/uncertainty_" + num2str(j); 
  save(name, "uncertainty");
  name = "../../myData/cartDoublePen_plots/data_" + num2str(save_at) +"/fantasy_data_" + num2str(j); 
  save(name, "fantasy");
  
  fprintf("MC rollouts took %.2f seconds.\n",toc); 
end
