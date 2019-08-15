%% settings_cdp.m
% *Summary:* Script set up the cart-double-pendulum scenario
%
% Copyright (C) 2008-2013 by 
% Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
%
% Last modified: 2013-03-27
%
%% High-Level Steps
% # Define state and important indices
% # Set up scenario
% # Set up the plant structure
% # Set up the policy structure
% # Set up the cost structure
% # Set up the GP dynamics model structure
% # Parameters for policy optimization
% # Plotting verbosity
% # Some array initializations


%% Code

warning('off','all'); format short; format compact

% include some paths
try
  rd = '../../';
  addpath([rd 'base'],[rd 'util'],[rd 'gp'],[rd 'control'],[rd 'loss']);
catch
end

% fix the random seed to be able to re-run the experiment
rand('seed', 20); randn('state', 5); 


% 1. Define state and important indices

% 1a. Full state representation (including all augmentations)
%  1   x             position of the cart
%  2   dx            velocity of the cart
%  3   dtheta1       angular velocity of inner pendulum
%  4   dtheta2       angular velocity of outer pendulum
%  5   theta1        angle of inner pendulum
%  6   theta2        angle of outer pendulum
%  7   sin(theta1)   complex representation ...
%  8   cos(theta1)   ... of theta1
%  9   sin(theta2)   complex representation ...
%  10  cos(theta2)   ... of theta2
%  11  u             force that can be applied at cart


% 1b. Important indices
% odei  indicies for the ode solver
% augi  indicies for variables augmented to the ode variables
% dyno  indicies for the output from the dynamics model and indicies to loss
% angi  indicies for variables treated as angles (using sin/cos representation)
% dyni  indicies for inputs to the dynamics model
% poli  indicies for the inputs to the policy
% difi  indicies for training targets that are differences (rather than values)

odei = [1 2 3 4 5 6];
augi = [];
dyno = [1 2 3 4 5 6];
angi = [5 6];
dyni = [1 2 3 4 7 8 9 10];
poli = [1 2 3 4 7 8 9 10];
difi = [1 2 3 4 5 6];
acti = [11];

% 2. Set up the scenario
dt = 0.05;                % [s] sampling time
T = 5.0;                  % [s] prediction time
H = ceil(T/dt);           % prediction steps (optimization horizon)
maxH = H;                 % max pred horizon
nc = 200;                 % size of controller training set
s = [0.1 0.1 0.1 0.1 0.01 0.01].^2; % initial state variances
S0 = diag(s);             % initial state covariance matrix
mu0 = [0 0 0 0 pi pi]';   % initial state mean
N = 40;                   % number of policy searches
J = 1;                    % J initial (random) trajectories, each of length H 
K = 1;                    % number of initial states for which we optimize


% 3. Set up the plant structure
plant.dynamics = @dynamics_cdp;           % handle to dynamics ODE function
plant.noise = diag(ones(1,6)*0.01.^2);    % measurement noise
plant.dt = dt;
plant.ctrl = @zoh;        % controller is zero order hold
plant.odei = odei;        % indices to the varibles for the ode solver
plant.augi = augi;        % indices of augmented variables
plant.angi = angi;
plant.poli = poli;
plant.dyno = dyno;
plant.dyni = dyni;
plant.difi = difi;
plant.prop = @propagated; % handle to function that propagates state over time

% 4. Set up the policy structure
policy.fcn = @(policy,m,s)conCat(@congp,@gSat,policy,m,s); % controller 
                                                           % representation
policy.maxU = 20;                                          % max. amplitude of 
                                                           % control
[mm ss cc] = gTrig(mu0, S0, plant.angi);                   % represent angles 
mm = [mu0; mm]; cc = S0*cc; ss = [S0 cc; cc' ss];          % in complex plane      
policy.p.inputs = gaussian(mm(poli), ss(poli,poli), nc)';  % init. location of 
                                                           % basis functions
policy.p.targets = 0.1*randn(nc, length(policy.maxU));  % init. policy targets 
                                                        % (close to zero)
policy.p.hyp = log([1 1 1 1 0.7 0.7 0.7 0.7 1 0.01]');  % initialize policy
                                                        % hyper-parameters

% 5. Set up the cost structure
cost.fcn = @loss_cdp;                             % handle to cost function
cost.gamma = 1;                                   % discount factor
cost.p = [1 1];                                   % lenghts of the links
cost.width = 0.5;                                 % cost function width
cost.expl = 0;                                    % exploration parameter
cost.angle = plant.angi;                          % angle variables in cost
cost.target = zeros(6,1);                         % target state

% 6. Set up the GP dynamics model structure
dynmodel.fcn = @gp1d;                % function for GP predictions
dynmodel.train = @train;             % function to train dynamics model
dynmodel.induce = zeros(400,0,1);    % shared inducing inputs (sparse GP)
trainOpt = [300 500];                % defines the max. number of line searches
                                     % when training the GP dynamics models
                                     % trainOpt(1): full GP,
                                     % trainOpt(2): sparse GP (FITC)

% 7. Parameters for policy optimization
opt.length = 150;                        % max. number of line searches
opt.MFEPLS = 30;                         % max. number of function evaluations
                                         % per line search
opt.verbosity = 1;                       % verbosity: specifies how much 
                                         % information is displayed during
                                         % policy learning. Options: 0-3
opt.method = 'BFGS';                     % optimization algorithm. Options:
                                         % 'BFGS' (default), 'LBFGS', 'CG'

% 8. Plotting verbosity
plotting.verbosity = 1;            % 0: no plots
                                   % 1: some plots
                                   % 2: all plots


% 9. Initialize various variables
x = []; y = [];                                  
fantasy.mean = cell(1,N); fantasy.std = cell(1,N);
realCost = cell(1,N); M = cell(N,1); Sigma = cell(N,1);


%10. uncertainty analysis
do_uncertainty = true;
uncertainty = nan(N, 3);
N_num = 100; % number of trajectory starts
M_num = 100; % number of sets of weights
T_num = H; % number of timesteps in rollout
nbf = 500; % number of basis functions