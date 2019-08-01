clearvars;
close all;
%% load data

episode = 10;
dataset = 1;

% load Monte-Carlo trajectories (needs to be the same episode as X_data)
name = "../data_"+num2str(dataset) +"/trajectories_" + num2str(episode) + ".mat";
trajectories = load(name);
traj = trajectories.trajectories;

% load actual state action transitions (needs to be the same episode as trajectories)
name = "../data_"+num2str(dataset) +"/X_data_" + num2str(episode) + ".mat";
X_data = load(name);
X = X_data.X_tr;

% load uncertainty data (should always be the last episode)
name = "../data_"+num2str(dataset) +"/uncertainty_10.mat";
uncertainty = load(name);
unc = uncertainty.uncertainty;

% load fantasy data (should always be the last episode)
name = "../data_"+num2str(dataset) +"/fantasy_data_10.mat";
fantasy = load(name);
fant = fantasy.fantasy;

% get array size
[M, T, N, D] = size(traj);

%% learned enviroment
% [N, dim] = size(X);
% start = N-39;
% timesteps = start:N;
% 
% figure(1)
% plot(timesteps, X(start:end,1))
% grid on
% hold on
% plot(timesteps, X(start:end,2), '-o')
% plot(timesteps, X(start:end,3), '-v')
% plot(timesteps, X(start:end,4))
% plot(timesteps, X(start:end,5))
% plot(timesteps, X(start:end,6), 'r*')

%% plot of dimensions of real rollout vs MC rollout
%  1  x          cart position
%  2  v          cart velocity
%  3  dtheta     angular velocity
%  4  theta      angle of the pendulum (removed)
%  5  sin(theta) complex representation ...
%  6  cos(theta) of theta
%  7  u          force applied to cart


% define start for X_data
endi = 100 * episode;
start = endi - 99; 
timesteps = 1:100;

% redefine N & Mfor fewer trajectories
N = 20;
M = 20;

% make colour  map
ii = M*N;
light_blue = [230, 230, 230]/256;% [173,216,230]/256; %[232,203,192]/256;  
dark_blue = [10, 10, 10]/256; %[0,0,139]/256;%[99,111,164]/256;%%[0,0,139]/256;
Red = linspace(light_blue(1),dark_blue(1),ii);  %// Red from 1 to 0
Green = linspace(light_blue(2),dark_blue(2),ii);  %// Green from 0 to
Blue = linspace(light_blue(3),dark_blue(3),ii);  %// Blue from 0 to 1

ylabels = ["Cart position, m", "Cart velocity, m/s", "Angular velocity, $\dot\theta_{1}$", ...
                "Angular velocity, $\dot\theta_{2}$", "$sin(\theta_{1})$", "$cos(\theta_{1})$", ...
                "$sin(\theta_{2})$", "$cos(\theta_{2})$"];
targets = [0 0 0 0 0 1 0 1];

for dd = 1:D-1
    fig = figure(dd+1);
    jj = 1;
    for mm = 1:M
        for nn = 1:N
            plot(timesteps, traj(mm, :, nn, dd), 'color', [Red(jj), Green(jj), Blue(jj)])
            hold on 
            jj = jj+1;
        end
    end
    grid on
    set(gca,'GridLineStyle','--')
    plot(timesteps, repmat(targets(dd), 1, length(timesteps)),'--g', 'linewidth',1.5)
    plot(timesteps, X(start:endi,dd),'r', 'linewidth',1.5)
    xlabel("Timesteps",'fontsize', 15, 'interpreter','latex')
    xlim([1 100])
    ylabel(ylabels(dd),'fontsize', 15,'interpreter','latex')
    f=get(gca,'Children');
    legend([f(1),f(2),f(3)],'Actual trajectory', 'Target', 'Monte Carlo trajectories','interpreter', 'latex')
    
    name = "/home/dl00065/Documents/MATLAB/thesis/PILCO/pilcoV0.9/cartDoublePen_plots/trajectory_plots/MC_rollout_dimension_" + num2str(dd) + ".png";
    print(gcf,name,'-dpng','-r400'); 
%     saveas(fig,name);
    close(fig);
    
end


%% plot uncertainty
% dimensions of unceratainty are:
% 1. Total uncertainty per episode
% 2. Epistemic uncertainty per episode
% 3. Aleatoric uncertainty per episode
% 4. Average cost per episode

[num_episodes, ~] = size(unc);
episodes = 1:num_episodes;

% PILCO cost mean and std dev
ave = zeros(1, num_episodes);
std = zeros(1, num_episodes);
for ii = 1:num_episodes
    ave(1,ii) = mean(fant.mean{1,ii});
    std(1,ii) = mean(fant.std{1,ii});
end

% confidence bounds
lcb = ave - 2*std;
ucb = ave + 2*std;
x_plot = [episodes, fliplr(episodes)];
y_plot = [lcb, fliplr(ucb)];

figure(D+1)

yyaxis left
fill(x_plot, y_plot, 'y', 'FaceAlpha', 0.5)
hold on 
plot(episodes, ave, '--b', 'linewidth', 2)
ylabel("Average cost")
grid on
set(gca,'GridLineStyle','--')


yyaxis right
plot(episodes, unc(:,1), 'ko')
plot(episodes, unc(:,2), 'rv')
plot(episodes, unc(:,3), 'g+')
legend('95% confidence interval','Average cost','Total uncertainty','Epistemic uncertainty',' Aleatoric uncertainty')
xlabel("Episode")
ylabel("Uncertainty")
ylim([1e-5 10])
xlim([1 15])
set(gca, 'YScale', 'log')
xticks([1 2 3 4 5 6 7 8 9 10 11 12 13 14 15])

name = "/home/dl00065/Documents/MATLAB/thesis/PILCO/pilcoV0.9/cartDoublePen_plots/trajectory_plots/uncertainty.png";
print(gcf,name,'-dpng','-r400'); 
% saveas(fig,name);












