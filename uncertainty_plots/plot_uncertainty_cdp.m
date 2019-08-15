clearvars;
close all;
%% load data

episode = 80;
dataset = 1;
max_ep = 80;

name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/trajectory_plots_cdp/data_"+num2str(dataset)+"_mats"+"/CartDoublePend_" + num2str(episode) + "_H100.mat";
mat = load(name);
Means = mat.M;
Sigma = mat.Sigma;

% new_data = true;
% load Monte-Carlo trajectories (needs to be the same episode as X_data)
name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/trajectory_plots_cdp/data_"+num2str(dataset) +"/trajectories_" + num2str(episode) + ".mat";
trajectories = load(name);
traje = trajectories.trajectories;

% load actual state action transitions (needs to be the same episode as trajectories)
name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/trajectory_plots_cdp/data_"+num2str(dataset) +"/X_data_" + num2str(episode) + ".mat";
X_data = load(name);
X_ = X_data.X_tr;
X = X_(:,1:4);
theta = atan2(X_(:,5), X_(:,6));
theta(theta<-1) = theta(theta<-1)+2*pi;
X(:,5) = theta;
theta = atan2(X_(:,7), X_(:,8));
theta(theta<1) = theta(theta<1)+2*pi;
X(:,6) = theta;

% load uncertainty data (should always be the last episode)
name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/trajectory_plots_cdp/data_"+num2str(dataset) +"/uncertainty_new.mat";
uncertainty = load(name);
unc = uncertainty.new_uncertainty;

% load fantasy data (should always be the last episode)
name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/trajectory_plots_cdp/data_"+num2str(dataset) +"/fantasy_data_" + num2str(max_ep) + ".mat";
fantasy = load(name);
fant = fantasy.fantasy;

% extract the dimensions neede
% if new_data == true
% traj = traje(:,:,:,[1,2,3,4,7,8,9,10,11]);
% thetas = traje(:,:,:,[5,6]);
traj = traje(:,:,:,[1,2,3,4,5,6]);
% else
%     traj = traje(:,:,:,[1,2,3,4,5,6,7,8]);
% end
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


% define start for X_data
endi = 100 * episode;
start = endi - 99; 
timesteps = 0:99;

% redefine N & Mfor fewer trajectories
N = 30;
M = 30;

% make colour  map
ii = M*N;
light_blue = [237, 237, 237]/256;% [173,216,230]/256; %[232,203,192]/256;  
dark_blue = [54, 54, 54]/256; %[0,0,139]/256;%[99,111,164]/256;%%[0,0,139]/256;
Red = linspace(light_blue(1),dark_blue(1),ii);  %// Red from 1 to 0
Green = linspace(light_blue(2),dark_blue(2),ii);  %// Green from 0 to
Blue = linspace(light_blue(3),dark_blue(3),ii);  %// Blue from 0 to 1
%,"Angle, $\theta_{1}$","Angle, $\theta_{2}$"
ylabels = ["Cart position, m", "Cart velocity, m/s", "Angular velocity, $\dot\theta_{2}$", ...
                "Angular velocity, $\dot\theta_{3}$", "Angle $\theta_{2}$","Angle $\theta_{3}$"];
           
targets = [0 0 0 0 0 2*pi];
UL = [2 5 8.5 14.5 4.3 7];
LL = [-2 -4 -11.5 -6 -0.5 1.7];

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
    xlim([1 99])
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
    
    name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/trajectory_plots_cdp/data_"+num2str(dataset)+"_plots/MC_rollout_Ep_"+num2str(episode) + "_Dim_" + num2str(dd) + ".png";
    print(gcf,name,'-dpng','-r200'); 
%     saveas(fig,name);
    close(fig);
    
end


%% plot uncertainty
% dimensions of unceratainty are:
% 1. Total uncertainty per episode
% 2. Epistemic uncertainty per episode
% 3. Aleatoric uncertainty per episode
% 4. Average cost per episode

% [num_episodes, ~] = size(unc);
episodes = 1:max_ep;

% PILCO cost mean and std dev
ave = zeros(1, max_ep);
std = zeros(1, max_ep);
for ii = 1:max_ep
    ave(1,ii) = mean(fant.mean{1,ii});
    std(1,ii) = mean(fant.std{1,ii});
end

% confidence bounds
lcb = ave - 2*std;
ucb = ave + 2*std;
x_plot = [episodes, fliplr(episodes)];
y_plot = [lcb, fliplr(ucb)];

% fig = figure('units','normalized','outerposition',[0 0 1 1]);%figure(D+1);
fig = figure('Renderer', 'painters', 'Position', [10 10 900 600]);
yyaxis left
h(1) = plot(episodes, ave, '--b', 'linewidth', 2);
hold on 
h(2) = fill(x_plot, y_plot, 'y', 'FaceAlpha', 0.2);
ylabel("Average cost",'fontsize',15, 'interpreter', 'latex')
grid on
set(gca,'GridLineStyle','--')
line([10 10],[0 1.4],'color',[100,100,100]/255, 'LineStyle', '--','linewidth',1.5)
text(10,1.15,'\leftarrow pendulums upright but cannot balance','fontsize', 10)
line([43 43],[0 1.4],'color',[100,100,100]/255, 'LineStyle', '--','linewidth',1.5)
text(43,1.15,'\leftarrow pendulums balanced but cart not centred ','fontsize', 10)
line([50 50],[0 1.4],'color',[100,100,100]/255, 'LineStyle', '--','linewidth',1.5)
text(50,1.1,'\leftarrow environment solved ','fontsize', 10)
ylim([0 1.2])

yyaxis right
h(3) = plot(episodes, unc(1:max_ep,1), 'ko','MarkerFaceColor','k');
h(4) = plot(episodes, unc(1:max_ep,2), 'rv','MarkerFaceColor','r');
h(5) = plot(episodes, unc(1:max_ep,3), 'g*','MarkerFaceColor','g');
legend(h(1:5),'Average cost','95\% confidence interval','Total uncertainty','Epistemic uncertainty',' Aleatoric uncertainty','Location','Best','fontsize',10, 'interpreter', 'latex')
xlabel("Episode",'fontsize',  15, 'interpreter', 'latex')
ylabel("Uncertainty",'fontsize',  15, 'interpreter', 'latex')
ylim([0 22])
xlim([1 max_ep])
% set(gca, 'YScale', 'log')
% xticks([1 2 3 4 5 6 7 8 9 10 11 12 13 14 15])

name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/trajectory_plots_cdp/data_"+num2str(dataset)+"_plots/uncertainty.png";
print(gcf,name,'-dpng','-r400'); 
% saveas(fig,name);
% close(fig)

%% plot uncertainty ratio
% dimensions of unceratainty are:
% 1. Total uncertainty per episode
% 2. Epistemic uncertainty per episode
% 3. Aleatoric uncertainty per episode
% 4. Average cost per episode

% [num_episodes, ~] = size(unc);
episodes = 1:max_ep;

% PILCO cost mean and std dev
ave = zeros(1, max_ep);
std = zeros(1, max_ep);
for ii = 1:max_ep
    av = fant.mean{1,ii};
    st = fant.std{1,ii};
    ave(1,ii) = mean(av(1:101));
    std(1,ii) = mean(st(1:101));
end

% confidence bounds
lcb = ave - 2*std;
ucb = ave + 2*std;
x_plot = [episodes, fliplr(episodes)];
y_plot = [lcb, fliplr(ucb)];


% plot the results
% fig = figure(D+2);
fig = figure('Renderer', 'painters', 'Position', [10 10 900 600]);
yyaxis left
h(1) = plot(episodes, ave, '--b', 'linewidth', 2);
hold on 
h(2) = fill(x_plot, y_plot, 'y', 'FaceAlpha', 0.2);
ylabel("Average cost",'fontsize',  15, 'interpreter', 'latex')
grid on
set(gca,'GridLineStyle','--')
line([10 10],[0 1.4],'color',[100,100,100]/255, 'LineStyle', '--','linewidth',1.5)
text(10,1.15,'\leftarrow pendulums upright but cannot balance','fontsize', 10)
line([43 43],[0 1.4],'color',[100,100,100]/255, 'LineStyle', '--','linewidth',1.5)
text(43,1.15,'\leftarrow pendulums balanced but cart not centred ','fontsize', 10)
line([50 50],[0 1.4],'color',[100,100,100]/255, 'LineStyle', '--','linewidth',1.5)
text(50,1.1,'\leftarrow environment solved ','fontsize', 10)
ylim([0 1.2])

yyaxis right
h(3) = plot(episodes, unc(1:max_ep,2)./unc(1:max_ep,1), 'rv','MarkerFaceColor','r');
legend(h(1:3),'Average cost per episode','95\% confidence interval','Ratio of epistemic to total uncertainty','Location','southeast','fontsize',10, 'interpreter', 'latex')
xlabel("Episode",'fontsize',  15, 'interpreter', 'latex')
ylabel("Uncertainty",'fontsize',  15, 'interpreter', 'latex')
xlim([1 max_ep])

name = "/home/dl00065/Documents/MATLAB/active-exploration/plotting_scripts/trajectory_plots_cdp/data_"+num2str(dataset)+"_plots/uncertainty_normalised.png";
print(gcf,name,'-dpng','-r400'); 
% close(fig)







