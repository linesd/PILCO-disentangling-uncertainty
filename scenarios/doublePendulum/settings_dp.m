%% settings_dp.m
% *Summary:* Script set up the double-pendulum scenario with two actuators
%
% Copyright (C) 2008-2013 by 
% Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
%
% Last modified: 2013-05-24
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

warning('on','all'); format short; format compact

% include some paths
try
  rd = '../../';
  addpath([rd 'base'],[rd 'util'],[rd 'gp'],[rd 'control'],[rd 'loss']);
catch
end

rand('seed',1); randn('seed',1); format short; format compact; 

% 1. Define state and important indices

% 1a. Full state representation (including all augmentations)
%  1  dtheta1        angular velocity of inner pendulum
%  2  dtheta2        angular velocity of outer pendulum
%  3  theta1         angle inner pendulum
%  4  theta2         angle outer pendulum
%  5  sin(theta1)    complex representation ...
%  6  cos(theta1)    ... of angle of inner pendulum
%  7  sin(theta2)    complex representation ...
%  8  cos(theta2)    ... of angle of outer pendulum
%  9  u1             torque applied to the inner joint
% 10  u2             torque applied to the outer joint

% 1b. Important indices
% odei  indicies for the ode solver
% augi  indicies for variables augmented to the ode variables
% dyno  indicies for the output from the dynamics model and indicies to loss
% angi  indicies for variables treated as angles (using sin/cos representation)
% dyni  indicies for inputs to the dynamics model
% poli  indicies for variables that serve as inputs to the policy
% difi  indicies for training targets that are differences (rather than values)

odei = [1 2 3 4];                           
augi = [];                                   
dyno = [1 2 3 4];          
angi = [3 4];                                           
dyni = [1 2 5 6 7 8]; 
poli = [1 2 5 6 7 8];        
difi = [1 2 3 4];              
acti = [9 10];

% 2. Set up the scenario
dt = 0.1;                          % [s] sampling time
T = 3.0;                           % [s] prediction horizon
H = ceil(T/dt);                    % prediction steps (optimization horizon)
mu0 = [0 0 pi pi]';                % initial state mean
S0 = diag([0.1 0.1 0.01 0.01].^2); % initial state covariance
N = 20;                            % no. of controller optimizations
J = 1;                             % no. of initial training rollouts (length H)
K = 1;                             % no. of initial states for which we optimize
nc = 100;                          % size of controller training set

% 3. Set up the plant structure
plant.dynamics = @dynamics_dp;                           % dynamics ODE function
plant.noise = diag(ones(1,4)*0.01.^2);                   % measurement noise
plant.dt = dt;
plant.ctrl = @zoh;                  % controler is zero-order-hold
plant.odei = odei;                  % indices to the varibles for the ode solver
plant.augi = augi;                  % indices of augmented variables
plant.angi = angi;
plant.poli = poli;
plant.dyno = dyno;
plant.dyni = dyni;
plant.difi = difi;
plant.prop = @propagated;   % handle to function that propagates state over time



% 4. Set up the policy structure
policy.fcn = @(policy,m,s)conCat(@congp,@gSat,policy,m,s);% controller 
                                                          % representation
policy.maxU = [2 2];                                      % max. amplitude of 
                                                          % torques
[mm ss cc] = gTrig(mu0, S0, plant.angi);                  % represent angles 
mm = [mu0; mm]; cc = S0*cc; ss = [S0 cc; cc' ss];         % in complex plane      
policy.p.inputs = gaussian(mm(poli), ss(poli,poli), nc)'; % init. location of 
                                                          % basis functions
policy.p.targets = 0.1*randn(nc, length(policy.maxU));    % init. policy targets 
                                                          % (close to zero)
policy.p.hyp = ...                                        % initialize policy
  repmat(log([1 1 0.7 0.7 0.7 0.7 1 0.01]'), 1,2);        % hyper-parameters


% 5. Set up the cost structure
cost.fcn = @loss_dp;                        % cost function
cost.gamma = 1;                             % discount factor
cost.p = [0.5 0.5];                         % lengths of pendulums
cost.width = 0.5;                           % cost function width
cost.expl = 0;                              % exploration parameter (UCB)
cost.angle = plant.angi;                    % index of angle (for cost function)
cost.target = [0 0 0 0]';                   % target state

% 6. Set up the GP dynamics model structure
dynmodel.fcn = @gp1d;                % function for GP predictions
dynmodel.train = @train;             % function to train dynamics model
dynmodel.induce = zeros(300,0,1);    % shared inducing inputs (sparse GP)
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

% 8. Plotting verbosity
plotting.verbosity = 1;            % 0: no plots
                                   % 1: some plots
                                   % 2: all plots

% 9. Some initializations
x = []; y = [];
fantasy.mean = cell(1,N); fantasy.std = cell(1,N);
realCost = cell(1,N); M = cell(N,1); Sigma = cell(N,1);