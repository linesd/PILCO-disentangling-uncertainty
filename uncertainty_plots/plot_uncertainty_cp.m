clearvars;
close all;
%% load data

episode = 15;
num_eps = 15;
dataset = 1;

name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/trajectory_plots_cp/data_"+num2str(dataset)+"_mats"+"/cartPole_" + num2str(episode) + "_H40.mat";
mat = load(name);
Means = mat.M;
Sigma = mat.Sigma;

% load Monte-Carlo trajectories (needs to be the same episode as X_data)
name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/trajectory_plots_cp/data_"+num2str(dataset) +"/trajectories_" + num2str(episode) + ".mat";
trajectories = load(name);
traje = trajectories.trajectories;
% traj(:,:,:,4) = []; % remove thetas (temporary)

% load actual state action transitions (needs to be the same episode as trajectories)
name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/trajectory_plots_cp/data_"+num2str(dataset) +"/X_data_" + num2str(episode) + ".mat";
X_data = load(name);
X_ = X_data.X_tr;
X = X_(:,1:3);
theta = atan2(X_(:,4), X_(:,5));
theta(theta<-2) = theta(theta<-2)+2*pi;
X(:,4) = theta;
% load uncertainty data (should always be the last episode)
name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/trajectory_plots_cp/data_"+num2str(dataset) +"/uncertainty_new.mat";
uncertainty = load(name);
unc = uncertainty.new_uncertainty;

% load fantasy data (should always be the last episode)
name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/trajectory_plots_cp/data_"+num2str(dataset) +"/fantasy_data_" + num2str(num_eps) + ".mat";
fantasy = load(name);
fant = fantasy.fantasy;

%% plot of dimensions of real rollout vs MC rollout
%  1  x          cart position
%  2  v          cart velocity
%  3  dtheta     angular velocity
%  4  theta      angle of the pendulum (removed)
%  5  sin(theta) complex representation ...
%  6  cos(theta) of theta
%  7  u          force applied to cart

% pull out the states I need
% traj = traje(:,:,:,[1,2,3,5,6]);
traj = traje(:,:,:,[1,2,3,4]);

% define start for X_data
endi = 40 * episode;
start = endi - 39; 
timesteps = 0:39;
% get array size
[M, T, N, D] = size(traj);

M = 50;
N = 50;

% make colour  map
ii = M*N;
light_blue = [237, 237, 237]/256;% [173,216,230]/256; %[232,203,192]/256;  
dark_blue = [54, 54, 54]/256; %[0,0,139]/256;%[99,111,164]/256;%%[0,0,139]/256;
Red = linspace(light_blue(1),dark_blue(1),ii);  %// Red from 1 to 0
Green = linspace(light_blue(2),dark_blue(2),ii);  %// Green from 0 to
Blue = linspace(light_blue(3),dark_blue(3),ii);  %// Blue from 0 to 1

ylabels = ["Cart position, m", "Cart velocity, m/s", "Angular velocity, $\dot\theta$", ...
                "Angle, $\theta$"];
targets = [0 0 0 pi];
UL = [1 4 13 4];
LL = [-1 -4 -13 -1];
for dd = 1:D
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
    xlim([1 39])
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
    
    
    
    name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/trajectory_plots_cp/data_"+num2str(dataset)+"_plots/MC_rollout_Ep_"+num2str(episode) + "_Dim_" + num2str(dd) + ".png";
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
fig = figure('Renderer', 'painters', 'Position', [10 10 700 500]);

yyaxis left
h(1) = plot(episodes, ave, '--b', 'linewidth', 2);
hold on 
h(2) = fill(x_plot, y_plot,'y', 'FaceAlpha', 0.2);
ylabel("Average cost",'fontsize',  15, 'interpreter', 'latex')
grid on
set(gca,'GridLineStyle','--')
line([3 3],[0 1.4],'color',[255,140,0]/255, 'LineStyle', '--','linewidth',1.5)
text(3,1.2,'\leftarrow environment solved ','fontsize', 13)

yyaxis right
h(3)=plot(episodes, unc(:,1), 'ko','MarkerFaceColor','k');
h(4)=plot(episodes, unc(:,3), 'g*','MarkerFaceColor','g');
h(5)=plot(episodes, unc(:,2), 'rv','MarkerFaceColor','r');
legend(h([1 2 3 4 5]),'Average cost per episode','95\% confidence interval','Total uncertainty',' Aleatoric uncertainty','Epistemic uncertainty','Location','northeast','interpreter','latex', 'fontsize',10)

ylabel("Uncertainty",'fontsize',  15, 'interpreter', 'latex')
ylim([0 25])
xlim([1 15])
% set(gca, 'YScale', 'log')
xticks([1 2 3 4 5 6 7 8 9 10 11 12 13 14 15])

name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/trajectory_plots_cp/data_"+num2str(dataset)+"_plots/cp_uncertainty.png";
print(gcf,name,'-dpng','-r200'); 


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
fig = figure('Renderer', 'painters', 'Position', [10 10 700 500]);

yyaxis left
h(1) = plot(episodes, ave, '--b', 'linewidth', 2);
hold on 
h(2) = fill(x_plot, y_plot, 'y', 'FaceAlpha', 0.2);
ylabel("Average cost",'fontsize',  15, 'interpreter', 'latex')
grid on
set(gca,'GridLineStyle','--')
line([3 3],[0 1.4],'color',[255,140,0]/255, 'LineStyle', '--','linewidth',1.5)
text(3,1,'\leftarrow environment solved ','fontsize', 13)

yyaxis right
h(3) = plot(episodes, unc(:,2)./unc(:,1), 'rv','MarkerFaceColor','r');
legend(h([1 2 3]),'Average cost per episode','95\% confidence interval','Ratio of epistemic to total uncertainty', 'Location','northeast','interpreter','latex')
xlabel("Episode",'fontsize',  15, 'interpreter', 'latex')
ylabel("Uncertainty",'fontsize',  15, 'interpreter', 'latex')
xlim([1 15])
xticks([1 2 3 4 5 6 7 8 9 10 11 12 13 14 15])

name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/trajectory_plots_cp/data_"+num2str(dataset)+"_plots/cp_uncertainty_norm.png";
print(gcf,name,'-dpng','-r200'); 








