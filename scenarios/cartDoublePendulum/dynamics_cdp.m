%% dynamics_cdp.m
% *Summary:* Implements ths ODE for simulating the cart-double pendulum 
% dynamics. 
%
%    function dz = dynamics_cdp(t,z,f)
%
%
% *Input arguments:*
%
%		t     current time step (called from ODE solver)
%   z     state                                                    [6 x 1]
%   f     (optional): force f(t)
%
% *Output arguments:*
%   
%   dz    if 3 input arguments:      state derivative wrt time
%         if only 2 input arguments: total mechanical energy
%
%   Note: It is assumed that the state variables are of the following order:
%         x:        [m]     position of cart
%         dx:       [m/s]   velocity of cart
%         dtheta1:  [rad/s] angular velocity of inner pendulum
%         dtheta2:  [rad/s] angular velocity of outer pendulum
%         theta1:   [rad]   angle of inner pendulum
%         theta2:   [rad]   angle of outer pendulum
%
%
% A detailed derivation of the dynamics can be found in:
%
% M.P. Deisenroth: 
% Efficient Reinforcement Learning Using Gaussian Processes, Appendix C, 
% KIT Scientific Publishing, 2010.
%
% Copyright (C) 2008-2013 by
% Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
%
% Last modified: 2013-03-05

function dz = dynamics_cdp(t,z,f)
%% Code

% set up the system
m1 = 0.5;  % [kg]     mass of cart
m2 = 0.5;  % [kg]     mass of 1st pendulum
m3 = 0.5;  % [kg]     mass of 2nd pendulum
l2 = 0.6;  % [m]      length of 1st pendulum
l3 = 0.6;  % [m]      length of 2nd pendulum
b  = 0.1;  % [Ns/m]   coefficient of friction between cart and ground
g  = 9.82; % [m/s^2]  acceleration of gravity

if nargin == 3
  
  A = [2*(m1+m2+m3) -(m2+2*m3)*l2*cos(z(5)) -m3*l3*cos(z(6))
       -(3*m2+6*m3)*cos(z(5)) (2*m2+6*m3)*l2 3*m3*l3*cos(z(5)-z(6))
       -3*cos(z(6)) 3*l2*cos(z(5)-z(6)) 2*l3];
  b = [2*f(t)-2*b*z(2)-(m2+2*m3)*l2*z(3)^2*sin(z(5))-m3*l3*z(4)^2*sin(z(6))
       (3*m2+6*m3)*g*sin(z(5))-3*m3*l3*z(4)^2*sin(z(5)-z(6))
       3*l2*z(3)^2*sin(z(5)-z(6))+3*g*sin(z(6))];
  x = A\b;

  dz = zeros(6,1);
  dz(1) = z(2);
  dz(2) = x(1);
  dz(3) = x(2);
  dz(4) = x(3);
  dz(5) = z(3);
  dz(6) = z(4);
  
else
  
  dz = (m1+m2+m3)*z(2)^2/2+(m2/6+m3/2)*l2^2*z(3)^2+m3*l3^2*z(4)^2/6 ...
       -(m2/2+m3)*l2*z(2)*z(3)*cos(z(5))-m3*l3*z(2)*z(4)*cos(z(6))/2 ...
       +m3*l2*l3*z(3)*z(4) *cos(z(5)-z(6))/2+(m2/2+m3)*l2*g*cos(z(5)) ...
       +m3*l3*g*cos(z(6))/2;
     
% I2 = m2*l2^2/12;  % moment of inertia around pendulum midpoint (1st link)
% I3 = m3*l3^2/12;  % moment of inertia around pendulum midpoint (2nd link)
% 
% 
% dz = m1*z(2)^2/2 + m2/2*(z(2)^2-l2*z(2)*z(3)*cos(z(5))) ...
%     + m3/2*(z(2)^2 - 2*l2*z(2)*z(3)*cos(z(5)) - l3*z(2)*z(4)*cos(z(6))) ...
%     + m2*l2^2*z(3)^2/8 + I2*z(3)^2/2 ...
%     + m3/2*(l2^2*z(3)^2 + l3^2*z(4)^2/4 + l2*l3*z(3)*z(4)*cos(z(5)-z(6))) ...
%     + I3*z(4)^2/2 ...
%     + m2*g*l2*cos(z(5))/2 + m3*g*(l2*cos(z(5))+l3*cos(z(6))/2);

end