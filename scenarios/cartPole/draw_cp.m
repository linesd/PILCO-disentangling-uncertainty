%% draw_cp.m
% *Summary:* Draw the cart-pole system with reward, applied force, and 
% predictive uncertainty of the tip of the pendulum
%
%    function draw_cp(x, theta, force, cost, text1, text2, M, S)
%
%
% *Input arguments:*
%
%		x          position of the cart
%   theta      angle of pendulum
%   force      force applied to cart
%   cost       cost structure
%     .fcn     function handle (it is assumed to use saturating cost)
%     .<>      other fields that are passed to cost
%   M          (optional) mean of state
%   S          (optional) covariance of state
%   text1      (optional) text field 1
%   text2      (optional) text field 2
%
%
% Copyright (C) 2008-2013 by
% Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
%
% Last modified: 2013-03-07

function draw_cp(x, theta, force, cost, text1, text2, M, S)
%% Code

l = 0.6;
xmin = -3; 
xmax = 3;    
height = 0.1;
width  = 0.3;
maxU = 10;

% Compute positions 
cart = [ x + width,  height
         x + width, -height
         x - width, -height
         x - width,  height
         x + width,  height ];
pendulum = [x, 0; x+2*l*sin(theta), -cos(theta)*2*l];


clf; hold on
plot(0,2*l,'k+','MarkerSize',20,'linewidth',2)
plot([xmin, xmax], [-height-0.03, -height-0.03],'k','linewidth',2)

% Plot force
plot([0 force/maxU*xmax],[-0.3, -0.3],'g','linewidth',10)

% Plot reward
reward = 1-cost.fcn(cost,[x, 0, 0, theta]', zeros(4));
plot([0 reward*xmax],[-0.5, -0.5],'y','linewidth',10)

% Plot the cart-pole
fill(cart(:,1), cart(:,2),'k','edgecolor','k');
plot(pendulum(:,1), pendulum(:,2),'r','linewidth',4)

% Plot the joint and the tip
plot(x,0,'y.','markersize',24)
plot(pendulum(2,1),pendulum(2,2),'y.','markersize',24)

% plot ellipse around tip of pendulum (if M, S exist)
try
  [M1 S1] = getPlotDistr_cp(M,S,2*l);
  error_ellipse(S1,M1,'style','b');
catch
end

% Text
text(0,-0.3,'applied force')
text(0,-0.5,'immediate reward')
if exist('text1','var')
  text(0,-0.9, text1)
end
if exist('text2','var')
  text(0,-1.1, text2)
end

set(gca,'DataAspectRatio',[1 1 1],'XLim',[xmin xmax],'YLim',[-1.4 1.4]);
axis off;
drawnow;