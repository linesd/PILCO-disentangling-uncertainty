clearvars;
close all;

%% Generate some data
X_tr = [-4;-2;-1.5;0.3;1;2;3.5;4.5];
Y_tr = [-0.35;0.5;1.5;1.9;0.2;-2.2;-1.5;-1];
a  =-5;
b = 5;
X_te = reshape(linspace(a,b,100), [100, 1]);
Y_te = zeros(100,1);

%% fit the data and plot the fit
nbf = 20; % number of basis functions
% hyper parameters
loghyper = rand(1+2,1);
        
% fit the data
[NMSE, mu, S2, NMLP, loghyper1, convergence] = ...
            ssgprfixed_ui(X_tr, Y_tr, X_te, Y_te, nbf, -1000, loghyper);

% pull some posterior samples
% draw a set of weights 
[mu_p, cov_p, phistar] = ssgprfixed(loghyper1, X_tr, Y_tr, X_te, true);
weights  =  mvnrnd(mu_p', cov_p, 5)';
samples = phistar * weights;

% get std dev
std = sqrt(S2);

% confidence bounds
lcb = mu - 2*std;
ucb = mu + 2*std;
x_plot = [X_te', fliplr(X_te')];
y_plot = [ucb', fliplr(lcb')];
        
% plot the fit
figure(1)
fill(x_plot, y_plot, [5,5,5]/256, 'FaceAlpha', 0.2)
hold on
plot(X_te, mu, 'k')
plot(X_tr, Y_tr, 'r*')
plot(X_te, samples, '--')
xlabel('$(\mathbf{x},\mathbf{u})$','fontsize', 10, 'interpreter', 'latex')
ylabel('$f(\mathbf{x},\mathbf{u})$','fontsize', 10, 'interpreter', 'latex')
grid on
set(gca,'GridLineStyle','--')
legend('95\% confidence interval','Predictive mean', 'Training data', 'Samples from $q(\mathbf{w})$', ...
            'fontsize', 10,'interpreter', 'latex')
xticks([-5 -4 -3 -2 -1 0 1 2 3 4 5])
%% do rollouts with an extra data point at each rollout
M = 100; % number of draws of weights
N = 100; % number of consecutive trajectories
T = 100; %number of timesteps
S = 3; %std dev for the cost calculation
target = 0;


traj_c = zeros(M, N, T);
traj_t = zeros(M, N, T);
 
X_te = reshape(linspace(a,b,N), [N, 1]);

loghyper = rand(1+2,1);

% fit the data
[NMSE, mu, S2, NMLP, loghyper1, convergence] = ...
        ssgprfixed_ui(X_tr, Y_tr, X_te, Y_te, nbf, -1000, loghyper);
loghyper1(1) =0.05;

for m = 1:M

    % draw a set of weights 
    [mu_p, cov_p, ~] = ssgprfixed(loghyper1, X_tr, Y_tr, X_te, true);
    weights  =  mvnrnd(mu_p', cov_p, 1)';

    % get N start states
    states = a + (b-a).*rand(N, 1);

    % compute the cost of the states and store in traj
    traj_c(m,:,1) = 1 - exp(-(1/(2*S^2)*(states - target).^2));
    traj_t(m,:,1) = states;

    % do the rollouts
    for t=2:T

        % predict one step from the posterior 
        [~, ~, phistar] = ssgprfixed(loghyper1, X_tr, Y_tr, states, true);
        next_states = phistar * weights; 

        % make next current and repeat
        states = next_states + normrnd(0,sqrt(0.3), [N,1]);

        % compute the cost of the states and store in traj
        traj_c(m,:,t) = 1 - exp(-(1/(2*S^2)*(states - target).^2))';

        traj_t(m,:,t) = states;

    end
end

%plot histogram of last rollout
figure(2)
nbins = 150;
histogram(reshape(traj_t, 1, M*N*T),nbins,'Normalization','pdf')
grid on
set(gca,'GridLineStyle','--')
xlabel('$(\mathbf{x},\mathbf{u})$','fontsize', 10, 'interpreter', 'latex')

name = "/home/dl00065/Documents/MATLAB/thesis/PILCO/pilcoV0.9/general_plots/transition_histgram.png";
print(gcf,name,'-dpng','-r400');