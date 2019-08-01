clearvars;
close all;

%% Generate some data
K = 50; % number of data points
a=-10; b=10; std = 0.25;
X_tr = a + (b-a).*rand(K, 1);
fx = 4*sin(0.5*X_tr) +4*cos(0.6*sqrt(2)*X_tr);
Y_tr = std.*randn(K,1) + fx; 

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
fill(x_plot, y_plot, 'y', 'FaceAlpha', 0.5)
hold on
plot(X_tr, Y_tr, '*')
plot(X_te, mu, 'k')
plot(X_te, samples, '--')
xlabel('Input')
ylabel('Posterior predictive')
grid on
set(gca,'GridLineStyle','--')

%% do rollouts with an extra data point at each rollout
M = 100; % number of draws of weights
N = 100; % number of consecutive trajectories
T = 100; %number of timesteps
S = 3; %std dev for the cost calculation
target = 0;

% get some values
start = 4;
step = 1;
num_k = length(start:step:K);

% initialise the uncertainty arrays
total = zeros(num_k, 1);
epistemic = zeros(num_k, 1);
aleatoric = zeros(num_k, 1);

count = 1;

traj_c = zeros(M, N, T, num_k);
traj_t = zeros(M, N, T);
% counting the number of k loops
ck=1;

% save some params for plots
loghyper_save = zeros(23, 3);
lsp = 1;

for k = start:step:K  
    
    loghyper = rand(1+2,1);
    
    % fit the data
    [NMSE, mu, S2, NMLP, loghyper1, convergence] = ...
            ssgprfixed_ui(X_tr(1:k), Y_tr(1:k), X_te, Y_te, nbf, -1000, loghyper);
    loghyper1(1) =0.05;
    
    for m = 1:M
        
        % draw a set of weights 
        [mu_p, cov_p, ~] = ssgprfixed(loghyper1, X_tr(1:k), Y_tr(1:k), X_te, true);
        weights  =  mvnrnd(mu_p', cov_p, 1)';
    
        % get N start states
        states = a + (b-a).*rand(N, 1);

        % compute the cost of the states and store in traj
        traj_c(m,:,1,ck) = 1 - exp(-(1/(2*S^2)*(states - target).^2));
        if k ==K
            traj_t(m,:,1) = states;
        end
        % do the rollouts
        for t=2:T
  
            % predict one step from the posterior 
            [~, ~, phistar] = ssgprfixed(loghyper1, X_tr(1:k), Y_tr(1:k), states, true);
            next_states = phistar * weights; 
           
            % make next current and repeat
            states = next_states + normrnd(0,sqrt(0.3), [N,1]);
            
            % compute the cost of the states and store in traj
            traj_c(m,:,t,ck) = 1 - exp(-(1/(2*S^2)*(states - target).^2))';
            
            if k ==K
                traj_t(m,:,t) = states;

            end
        end
    end
    
    % disentangle uncertainty trajectory costs
    total(ck,1) = var(traj_c(:,:,:,ck), 1, [1,2,3]);
    epistemic(ck,1) = var(mean(traj_c(:,:,:,ck), [2,3]));
    aleatoric(ck,1) = mean(var(traj_c(:,:,:,ck), 1, [2,3]));
    
    %
    % WILL NEED TO CHANGE THE LAST CONDITIONAL FOR STEPS > 1
    %
    
    if (ck == 1) || (ck == K/2) || (ck+start == K)
        loghyper_save(:,lsp) = loghyper1;
        ks(lsp) = k;
        lsp = lsp +1;
    end
    
    ck = ck+1;
end

%% Plot the data

samples = zeros(100, 5, 3);
mu1 = zeros(100, 3);
S12 = zeros(100, 3);

for i = 1:3

% need to get predictive distibrution here
[mu1(:,i), S12(:,i)] = ssgprfixed(loghyper_save(:,i), X_tr(1:ks(i)), Y_tr(1:ks(i)), X_te);
% draw posterior samples
[mu_p, cov_p, phistar] = ssgprfixed(loghyper_save(:,i), X_tr(1:ks(i)), Y_tr(1:ks(i)), X_te, true);
weights  =  mvnrnd(mu_p', cov_p, 5)';
samples(:, :, i) = phistar * weights;
% get std dev
std = sqrt(S12(:,i));

% confidence bounds
lcb(:,i) = mu1(:,i) - 2*std;
ucb(:,i) = mu1(:,i)+ 2*std;
x_plot(:,:,i) = [X_te', fliplr(X_te')];
y_plot(:,:,i) = [lcb(:,i)', fliplr(ucb(:,i)')];
    
end

% plot uncertainty
x = start:step:K;
shades = linspace(0, 0.5, num_k);
figure(2);

% subplot 1
subplot(2,2,1)
fill(x_plot(:,:,1), y_plot(:,:,1), [220 220 220]/256 , 'FaceAlpha', 0.5)
hold on
plot(X_te,  mu1(:,1), 'k')
plot(X_tr(1:ks(1)), Y_tr(1:ks(1)), '*')
plot(X_te, samples(:, :, 1) , '--');
grid on
set(gca,'GridLineStyle','--')
title("Predictive distribution for 4 data points",'fontsize',  14, 'interpreter', 'latex')
xlabel('$(\mathbf{x},\mathbf{u})$','fontsize', 10, 'interpreter', 'latex')
ylabel('$f(\mathbf{x},\mathbf{u})$','fontsize', 10, 'interpreter', 'latex')
legend('95\% confidence interval','Predictive mean', 'Training data', 'Samples from $q(\mathbf{w})$', ...
            'fontsize', 10,'interpreter', 'latex')

% subplot 2
subplot(2,2,2)
fill(x_plot(:,:,2), y_plot(:,:,2), [220 220 220]/256, 'FaceAlpha', 0.5)
hold on
plot(X_te,  mu1(:,2), 'k')
plot(X_tr(1:ks(2)), Y_tr(1:ks(2)), '*')
plot(X_te, samples(:, :, 2) , '--');
grid on
set(gca,'GridLineStyle','--')
title("Predictive distribution for 25 data points",'fontsize',  14, 'interpreter', 'latex')
xlabel('$(\mathbf{x},\mathbf{u})$','fontsize', 10, 'interpreter', 'latex')
ylabel('$f(\mathbf{x},\mathbf{u})$','fontsize', 10, 'interpreter', 'latex')
legend('95\% confidence interval','Predictive mean', 'Training data', 'Samples from $q(\mathbf{w})$', ...
            'fontsize', 10,'interpreter', 'latex')

%subplot 3
subplot(2,2,3)
fill(x_plot(:,:,3), y_plot(:,:,3), [220 220 220]/256, 'FaceAlpha', 0.5)
hold on
plot(X_te,  mu1(:,3), 'k')
plot(X_tr(1:ks(3)), Y_tr(1:ks(3)), '*')
plot(X_te, samples(:, :, 3) , '--');
grid on
set(gca,'GridLineStyle','--')
title("Predictive distribution for 50 data points", 'fontsize', 14, 'interpreter', 'latex')
xlabel('$(\mathbf{x},\mathbf{u})$','fontsize', 10, 'interpreter', 'latex')
ylabel('$f(\mathbf{x},\mathbf{u})$','fontsize', 10, 'interpreter', 'latex')
legend('95\% confidence interval','Predictive mean', 'Training data', 'Samples from $q(\mathbf{w})$', ...
            'fontsize', 10,'interpreter', 'latex')
        
%subplot 4
subplot(2,2,4)
plot(x, total, '--o')
hold on
plot(x, aleatoric, 's')
plot(x, epistemic, 'v')
grid on
set(gca,'GridLineStyle','--')
title("Uncertainty decomposition with added data points", 'fontsize', 14, 'interpreter', 'latex')
xlabel('Number of data points','fontsize', 10, 'interpreter', 'latex')
ylabel('Uncertainty','fontsize', 10, 'interpreter', 'latex')
xlim([start K])
ylim([5e-5 1])
set(gca, 'YScale', 'log')
legend('Total uncertainty','Aleatoric uncertainty','Epistemic uncertainty','fontsize', 10,'interpreter', 'latex')

name = "/home/dl00065/Documents/MATLAB/thesis/PILCO/pilcoV0.9/general_plots/function_uncertainty.png";
print(gcf,name,'-dpng','-r400');









