% Numerically place second segment of continuum CC model where it needs to
% be to hit a specified ray.
%
% Author: M. Kokko
% Updated: 04-Aug-2024

% restart
close all; clear; clc;

% configure parameters for 2 segment CC model
params = [];
params.x0 = 0;
params.y0 = 0;
params.L1 = 80; % [mm]
params.theta1 = pi/11; % 20*pi/180; % [rad]
params.L2 = 80; % [mm]
params.phi = 45*pi/180;
params.doPlot = false;

% compute some helpful lengths
r1 = params.L1/params.theta1; % [mm]
EF = r1*sin(params.phi)/sin(pi-params.phi-params.theta1); % verifed
BF = r1*(1-((sin(params.phi))/(sin(pi-params.phi-params.theta1)))); % verified

% find r2 that puts second segment tip on the ray specified by phi
% note: depends upon ICs because there are local minima in the objective
% function!
r2_0 = 2*BF;  % NOTE: try both 2*BF and 0.5*BF
f = @(x)cc_cost(params,x);
opts = optimset();
[r2, finalCost] = fminsearch(f,r2_0,opts);
theta2 = params.L2/r2;

% check to make sure solution is reasonable
if(theta2 > 2*pi)
    warning('Too much curvature! Don''t trust this result...');
end

% now plot with optimal r2
params.doPlot = true;
cc_cost(params,r2);

% TODO: compute and store z unit vector for second segment at tip
% TODO: iterate through handful of rays in first quadrant
% TODO: for each ray, iterate through choices of theta1
% TODO: for each choice of theta1, try initial r2 as 2*BF and 0.5*BF...
% solutions may differ if there are two ways to hit the ray

% cost function (also used for plotting)
function err = cc_cost(params,r2)

% compute r1 given L1 and theta1
r1 = params.L1/params.theta1;

% compute theta2 given parameter r2 and fixed arclength
theta2 = params.L2/r2;

% identify point where two segments are joined
xb = r1*(1-cos(params.theta1));
yb = r1*sin(params.theta1);

% unit vectors
uv_i = [cos(params.phi); sin(params.phi)]; % along our ray
uv_j = unitvec([xb-r1;yb]); % along shared radial line from center of first arc

% compute location of center of curvature for second segment
r_b_to_d = -1*uv_j*r2;
r_o_to_d = [xb;yb] + r_b_to_d;
xd = r_o_to_d(1);
yd = r_o_to_d(2);

% compute location of endpont of second segment
xc = xd - r2*cos(params.theta1+theta2);
yc = yd + r2*sin(params.theta1+theta2);
r_o_to_c = [xc;yc];

% compute displacement of endpoint of second segment from ray
err = norm(r_o_to_c-dot(r_o_to_c,uv_i)*uv_i);

% plot if we're asked to do that
if(params.doPlot)
    theta_arc_1 = 0:0.01:params.theta1;
    x_arc_1 = params.x0 + r1 - r1*cos(theta_arc_1);
    y_arc_1 = params.y0 + r1*sin(theta_arc_1);
    theta_arc_2 = params.theta1:0.01:(params.theta1+theta2);
    x_arc_2 = xd - r2*cos(theta_arc_2);
    y_arc_2 = yd + r2*sin(theta_arc_2);

    figure;
    hold on; grid on;
    plot(x_arc_1,y_arc_1,'-','LineWidth',1.6,'Color',[0 0 0.8]);
    plot(x_arc_2,y_arc_2,'-','LineWidth',1.6,'Color',[0.8 0 0.8]);
    plot([params.x0 params.x0+r1 xb],[params.y0 params.y0 yb],'.--','MarkerSize',20,'MarkerFaceColor',[0 0 0],'Color',[0 0 0]);
    plot([params.x0 (2*params.L1*cos(params.phi))],[params.y0 (2*params.L1*sin(params.phi))],'--','Color',[0.8 0 0]);
    plot(xd,yd,'.','MarkerSize',20,'Color',[0.8 0 0]);
    plot(xc,yc,'.','MarkerSize',20,'Color',[0.8 0 0.8]);
    axis equal;
end

end





