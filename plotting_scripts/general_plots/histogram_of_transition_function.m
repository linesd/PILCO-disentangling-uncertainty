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
plot(X_te, mu, 'k', 'linewidth', 1.5)
hold on
fill(x_plot, y_plot, [4,4,4]/256, 'FaceAlpha', 0.2)
plot(X_tr, Y_tr, 'r*','linewidth', 1.5)
plot(X_te, samples, '--','linewidth', 1.2)
xlabel('$\mathbf{x}$','fontsize', 15, 'interpreter', 'latex')
ylabel('$f(\mathbf{x})$','fontsize', 15, 'interpreter', 'latex')
grid on
set(gca,'GridLineStyle','--')
legend('Predictive mean', '95\% confidence interval','Training data', 'Samples from $q(\mathbf{w})$', ...
            'fontsize', 10,'interpreter', 'latex')
xticks([-5 -4 -3 -2 -1 0 1 2 3 4 5])

name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/general_plots/transition_function.png";
print(gcf,name,'-dpng','-r400');

%% do rollouts with an extra data point at each rollout
M = 1000; % number of draws of weights
N = 1000; % number of consecutive trajectories
T = 10; %number of timesteps
S = 1.5; %std dev for the cost calculation
target = 0;

times = [1, 2, 4, 10];

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
% figure(2)
% nbins = 150;
% histogram(reshape(traj_t, 1, M*N*T),nbins,'Normalization','pdf')
% grid on
% set(gca,'GridLineStyle','--')
% xlabel('$(\mathbf{x},\mathbf{u})$','fontsize', 10, 'interpreter', 'latex')
% 
% name = "/home/dl00065/Documents/MATLAB/thesis/PILCO/pilcoV0.9/general_plots/transition_histgram.png";
% print(gcf,name,'-dpng','-r400');
%% 
nbins = 100;

figure(2)
name = "Input distribution over states";
histogram(reshape(traj_t(:,:,times(1)), 1, M*N),nbins,'Normalization','pdf')
% title(name,'fontsize', 15, 'interpreter', 'latex')
hold on
line([target target],[0 0.3],'color','r', 'LineStyle', '--','linewidth',1.5)
text(-2,0.275,'Target \rightarrow ','fontsize', 13)
grid on
set(gca,'GridLineStyle','--')
xlabel('$\mathbf{x}$','fontsize', 15, 'interpreter', 'latex')
ylabel('$p(\mathbf{x})$','fontsize', 15, 'interpreter', 'latex')
xlim([-5 5])
ylim([0 0.3])
xticks([-5 -4 -3 -2 -1 0 1 2 3 4 5])
legend("State distribution",'fontsize', 10, 'interpreter', 'latex')
name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/general_plots/trans_traj_hist_1.png";
print(gcf,name,'-dpng','-r400');

figure(3)
name = "Distribution over states after " + num2str(times(2)-1) + " transitions";
histogram(reshape(traj_t(:,:,times(2)), 1, M*N),nbins,'Normalization','pdf')
% title(name,'fontsize', 15, 'interpreter', 'latex')
hold on
line([target target],[0 0.3],'color','r', 'LineStyle', '--','linewidth',1.5)
text(-2,0.275,'Target \rightarrow ','fontsize', 13)
grid on
set(gca,'GridLineStyle','--')
xlabel('$\mathbf{x}$','fontsize', 15, 'interpreter', 'latex')
ylabel('$p(\mathbf{x})$','fontsize', 15, 'interpreter', 'latex')
xlim([-5 5])
ylim([0 0.3])
xticks([-5 -4 -3 -2 -1 0 1 2 3 4 5])
legend("State distribution",'fontsize', 10, 'interpreter', 'latex')
name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/general_plots/trans_traj_hist_2.png";
print(gcf,name,'-dpng','-r400');

figure(4)
name = "Distribution over states after " + num2str(times(3)-1) + " transitions";
histogram(reshape(traj_t(:,:,times(3)), 1, M*N),nbins,'Normalization','pdf')
% title(name,'fontsize', 15, 'interpreter', 'latex')
hold on
line([target target],[0 0.3],'color','r', 'LineStyle', '--','linewidth',1.5)
text(-2,0.275,'Target \rightarrow ','fontsize', 13)
grid on
set(gca,'GridLineStyle','--')
xlabel('$\mathbf{x}$','fontsize', 15, 'interpreter', 'latex')
ylabel('$p(\mathbf{x})$','fontsize', 15, 'interpreter', 'latex')
xlim([-5 5])
ylim([0 0.3])
xticks([-5 -4 -3 -2 -1 0 1 2 3 4 5])
legend("State distribution",'fontsize', 10, 'interpreter', 'latex')
name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/general_plots/trans_traj_hist_3.png";
print(gcf,name,'-dpng','-r400');

figure(5)
name = "Distribution over states after " + num2str(times(4)-1) + " transitions";
histogram(reshape(traj_t(:,:,times(4)), 1, M*N),nbins,'Normalization','pdf')
% title(name,'fontsize', 15, 'interpreter', 'latex')
hold on
line([target target],[0 0.3],'color','r', 'LineStyle', '--','linewidth',1.5)
text(-2,0.275,'Target \rightarrow ','fontsize', 13)
grid on
set(gca,'GridLineStyle','--')
xlabel('$\mathbf{x}$','fontsize', 15, 'interpreter', 'latex')
ylabel('$p(\mathbf{x})$','fontsize', 15, 'interpreter', 'latex')
xlim([-5 5])
ylim([0 0.3])
xticks([-5 -4 -3 -2 -1 0 1 2 3 4 5])
legend("State distribution",'fontsize', 10, 'interpreter', 'latex')
name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/general_plots//trans_traj_hist_4.png";
print(gcf,name,'-dpng','-r400');

%
% COSTS
%

figure(6)
name = "Input distribution over costs";
histogram(reshape(traj_c(:,:,times(1)), 1, M*N),nbins,'FaceColor',[204,102,0]/256,'Normalization','pdf')
% title(name,'fontsize', 15, 'interpreter', 'latex')
grid on
set(gca,'GridLineStyle','--')
xlabel('$\mathcal{C}(\mathbf{x})$','fontsize', 15, 'interpreter', 'latex')
ylabel('$p(\mathcal{C}(\mathbf{x}))$','fontsize', 15, 'interpreter', 'latex')
xlim([0 1])
ylim([0 8])
xticks([0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0])
legend("Cost distribution",'fontsize', 10, 'interpreter', 'latex')
name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/general_plots/trans_cost_hist_1.png";
print(gcf,name,'-dpng','-r400');

figure(7)
name = "Distribution over costs after " + num2str(times(2)-1) + " transitions";
histogram(reshape(traj_c(:,:,times(2)), 1, M*N),nbins,'FaceColor',[204,102,0]/256,'Normalization','pdf')
% title(name,'fontsize', 15, 'interpreter', 'latex')
grid on
set(gca,'GridLineStyle','--')
xlabel('$\mathcal{C}(\mathbf{x})$','fontsize', 15, 'interpreter', 'latex')
ylabel('$p(\mathcal{C}(\mathbf{x}))$','fontsize', 15, 'interpreter', 'latex')
xlim([0 1])
ylim([0 8])
xticks([0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0])
legend("Cost distribution",'fontsize', 10, 'interpreter', 'latex')
name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/general_plots/trans_cost_hist_2.png";
print(gcf,name,'-dpng','-r400');

figure(8)
name = "Distribution over costs after " + num2str(times(3)-1) + " transitions";
histogram(reshape(traj_c(:,:,times(3)), 1, M*N),nbins,'FaceColor',[204,102,0]/256,'Normalization','pdf')
% title(name,'fontsize', 15, 'interpreter', 'latex')
grid on
set(gca,'GridLineStyle','--')
xlabel('$\mathcal{C}(\mathbf{x})$','fontsize', 15, 'interpreter', 'latex')
ylabel('$p(\mathcal{C}(\mathbf{x}))$','fontsize', 15, 'interpreter', 'latex')
xlim([0 1])
ylim([0 8])
xticks([0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0])
legend("Cost distribution",'fontsize', 10, 'interpreter', 'latex')
name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/general_plots/trans_cost_hist_3.png";
print(gcf,name,'-dpng','-r400');

figure(9)
name = "Distribution over costs after " + num2str(times(4)-1) + " transitions";
histogram(reshape(traj_c(:,:,times(4)), 1, M*N),nbins,'FaceColor',[204,102,0]/256,'Normalization','pdf')
% title(name,'fontsize', 15, 'interpreter', 'latex')
grid on
set(gca,'GridLineStyle','--')
xlabel('$\mathcal{C}(\mathbf{x})$','fontsize', 15, 'interpreter', 'latex')
ylabel('$p(\mathcal{C}(\mathbf{x}))$','fontsize', 15, 'interpreter', 'latex')
xlim([0 1])
ylim([0 8])
xticks([0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0])
legend("Cost distribution",'fontsize', 10, 'interpreter', 'latex')
name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/general_plots/trans_cost_hist_4.png";
print(gcf,name,'-dpng','-r400');












