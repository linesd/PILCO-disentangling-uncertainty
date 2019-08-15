clearvars;
close all;
%% load data

episode = 40;
num_eps = 40;
dataset = 1;

name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/trajectory_plots_penbot/data_"+num2str(dataset)+"_mats"+"/pendubot_" + num2str(episode) + "_H60.mat";
mat = load(name);
Means = mat.M;
Sigma = mat.Sigma;

% load Monte-Carlo trajectories (needs to be the same episode as X_data)
name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/trajectory_plots_penbot/data_"+num2str(dataset) +"/trajectories_" + num2str(episode) + ".mat";
trajectories = load(name);
traje = trajectories.trajectories;
% traj(:,:,:,4) = []; % remove thetas (temporary)

% load actual state action transitions (needs to be the same episode as trajectories)
name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/trajectory_plots_penbot/data_"+num2str(dataset) +"/X_data_" + num2str(episode) + ".mat";
X_data = load(name);
X_ = X_data.X_tr;
X = X_(:,1:2);
theta = atan2(X_(:,3), X_(:,4));
theta(theta<-2) = theta(theta<-2)+2*pi;
X(:,3) = theta;
theta = atan2(X_(:,5), X_(:,6));
theta(theta<-2) = theta(theta<-2)+2*pi;
X(:,4) = theta;
% load uncertainty data (should always be the last episode)
% name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/trajectory_plots_penbot/data_"+num2str(dataset) +"/uncertainty_" + num2str(num_eps) + ".mat";
name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/trajectory_plots_penbot/data_1/uncertainty_new.mat";
uncertainty = load(name);
unc = uncertainty.new_uncertainty;

% load fantasy data (should always be the last episode)
name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/trajectory_plots_penbot/data_"+num2str(dataset) +"/fantasy_data_" + num2str(num_eps) + ".mat";
fantasy = load(name);
fant = fantasy.fantasy;

% pull out the states I need
traj = traje(:,:,:,[1,2,3,4]);
num_timesteps = 60;

% define start for X_data
endi = num_timesteps * episode;
start = endi - num_timesteps+1; 
timesteps = 0:num_timesteps-1;
% get array size
[M, T, N, D] = size(traj);

%% plot of dimensions of real rollout vs MC rollout
% 1a. Full state representation (including all augmentations)
%  1  dtheta1        angular velocity of inner pendulum
%  2  dtheta2        angular velocity of outer pendulum
%  3  theta1         angle inner pendulum
%  4  theta2         angle outer pendulum
%  5  sin(theta1)    complex representation ...
%  6  cos(theta1)    ... of angle of inner pendulum
%  7  sin(theta2)    complex representation ...
%  8  cos(theta2)    ... of angle of outer pendulum
%  9  u              torque applied to the inner joint

M = 50;
N = 50;

% make colour  map
ii = M*N;
light_blue = [237, 237, 237]/256;% [173,216,230]/256; %[232,203,192]/256;  
dark_blue = [54, 54, 54]/256; %[0,0,139]/256;%[99,111,164]/256;%%[0,0,139]/256;
Red = linspace(light_blue(1),dark_blue(1),ii);  %// Red from 1 to 0
Green = linspace(light_blue(2),dark_blue(2),ii);  %// Green from 0 to
Blue = linspace(light_blue(3),dark_blue(3),ii);  %// Blue from 0 to 1

ylabels = ["Angular velocity, $\dot\theta_{2}$", "Angular velocity, $\dot\theta_{3}$", ...
                 "Angle, $\theta_2$", "Angle, $\theta_3$"];
                
%"$sin(\theta_{1})$", "$cos(\theta_{1})$","$sin(\theta_{2})$", "$cos(\theta_{2})$"];
targets = [0 0 0 0];
UL = [4 11 3.7 4.5];
LL = [-7 -12 -1 -0.5];

for dd = 1:D
%     fig = figure(dd+1);
    fig = figure('Renderer', 'painters', 'Position', [10 10 700 500]);
    
    jj = 1;
    for mm = 1:M
        for nn = 1:N
            plot(timesteps, traj(mm, :, nn, dd), 'color', [Red(jj), Green(jj), Blue(jj)],'linewidth',0.005,'LineStyle','-', 'MarkerSize',0.001);
            hold on 
            jj = jj+1;
        end
    end
    grid on
    set(gca,'GridLineStyle','--')
    yyaxis left
    errorbar( 0:length(Means{episode}(dd,:))-1, Means{episode}(dd,:),2*sqrt(squeeze(Sigma{episode}(dd,dd,:))), 'Color',[0,191,255]/255,'LineStyle','-');
    plot(timesteps, repmat(targets(dd), 1, length(timesteps)),'--g', 'linewidth',1.5)
    plot(timesteps, X(start:endi,dd),'r', 'linewidth',1,'LineStyle','-')
    xlabel("Timesteps",'fontsize', 15, 'interpreter','latex')
    xlim([1 59])
    ylim([LL(dd) UL(dd)])
    ylabel(ylabels(dd),'fontsize', 15,'interpreter','latex')
    f1=get(gca,'Children');
    
    
    yyaxis right
    STD2 = 2*sqrt(squeeze(Sigma{episode}(dd,dd,:))); % PILCO std dev
    MEAN = Means{episode}(dd,:);
    percentage=zeros(1,T);
    for tt = 1:T % count number of values inside this
        all_values = reshape(traj(:, tt, :, dd), 1, 100*100);
        num_in = sum(abs(all_values-MEAN(tt)) < STD2(tt));
        percentage(tt) = num_in / 10000;
    end
    plot(timesteps, percentage, 'm','linewidth',1)
    ylim([0 1])
    ylabel("Percentage MC within prediction, \%", 'fontsize', 15, 'interpreter','latex')
    
    f2=get(gca,'Children');
    legend([f1(1),f1(2),f1(3),f1(4),f2(1)],'Actual trajectory', 'Target', 'PILCO prediction', 'Monte-Carlo trajectories','MC Percentage','Location','southeast','fontsize', 10,'interpreter', 'latex')
    
    name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/trajectory_plots_penbot/data_"+num2str(dataset)+"_plots/MC_rollout_Ep_"+num2str(episode) + "_Dim_" + num2str(dd) + ".png";
    print(gcf,name,'-dpng','-r200'); 
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


% plot the results
% fig = figure(D+1);
fig = figure('Renderer', 'painters', 'Position', [10 10 700 500]);
yyaxis left
h(1) = plot(episodes, ave, '--b', 'linewidth', 2);
hold on 
h(2) = fill(x_plot, y_plot, 'y', 'FaceAlpha', 0.2);
ylabel("Average cost",'fontsize',  15, 'interpreter', 'latex')
grid on
set(gca,'GridLineStyle','--')
line([5 5],[0.3 1.4],'color',[200,140,0]/255, 'LineStyle', '--','linewidth',1.5)
line([10 10],[0.3 1.4],'color',[255,140,0]/255, 'LineStyle', '--','linewidth',1.5)
text(5,1.25,'\leftarrow pendulum upright but cannot balance','fontsize', 10)
text(10,0.95,'\leftarrow environment solved ','fontsize', 10)
ylim([0.3 1.3])

yyaxis right
h(3) = plot(episodes, unc(:,1), 'ko','MarkerFaceColor','k');
h(4) = plot(episodes, unc(:,2), 'rv','MarkerFaceColor','r');
h(5) = plot(episodes, unc(:,3), 'g*','MarkerFaceColor','g');
legend(h([1:5]),'Average cost per episode','95\% confidence interval','Total uncertainty','Epistemic uncertainty',' Aleatoric uncertainty','Location','Best','interpreter','latex')
xlabel("Episode",'fontsize',  15, 'interpreter', 'latex')
ylabel("Uncertainty",'fontsize',  15, 'interpreter', 'latex')
ylim([0 25])
xlim([1 20])
% set(gca, 'YScale', 'log')
% xticks([1 2 3 4 5 6 7 8 9 10 11 12 13 14 15])

name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/trajectory_plots_penbot/data_"+num2str(dataset)+"_plots/pen_uncertainty.png";
print(gcf,name,'-dpng','-r200'); 
% close(fig)

%% plot uncertainty ratio
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


% plot the results
% fig = figure(D+2);
fig = figure('Renderer', 'painters', 'Position', [10 10 700 500]);
yyaxis left
h(1) = plot(episodes, ave, '--b', 'linewidth', 2);
hold on 
h(2) = fill(x_plot, y_plot, 'y', 'FaceAlpha', 0.2);
ylabel("Average cost",'fontsize',  15, 'interpreter', 'latex')
grid on
set(gca,'GridLineStyle','--')
line([5 5],[0.3 1.4],'color',[200,140,0]/255, 'LineStyle', '--','linewidth',1.5)
line([10 10],[0.3 1.4],'color',[255,140,0]/255, 'LineStyle', '--','linewidth',1.5)
text(5,1.25,'\leftarrow pendulum upright but cannot balance','fontsize', 10)
text(10,0.95,'\leftarrow environment solved ','fontsize', 10)
ylim([0.3 1.3])

yyaxis right
h(3) = plot(episodes, unc(:,2)./unc(:,1), 'rv','MarkerFaceColor','r');
legend('Average cost per episode','95\% confidence interval','Ratio of epistemic to total','interpreter','latex')
xlabel("Episode",'fontsize',  15, 'interpreter', 'latex')
ylabel("Uncertainty",'fontsize',  15, 'interpreter', 'latex')
xlim([20 40])
ylim([0.08 0.24])
name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/trajectory_plots_penbot/data_"+num2str(dataset)+"_plots/pen_uncertainty_norm_20_40.png";
print(gcf,name,'-dpng','-r200'); 






