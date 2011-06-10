% File: bike_inertia_jason_rigid.m
% Date: June 10, 2011
% Author: Jason Moore
% Description: Calculates the center of mass and moment of inertia of Jason
% sitting on the rigid rider instrumented bicycle.

clear
close all
clc
warning off MATLAB:fzero:UndeterminedSyntax
%-------------------Frame Parameters (Instrumented Bicycle)
rf      = 0.336;                         % front wheel radius [m]
rr      = 0.333;                         % rear wheel radius [m]
wb      = 1.08;                          % wheel base [m]
hb      = 0.291;                         % bottom bracket height [m]
lcs     = 0.423;                         % chain stay length [m]
alphast = 72.2*pi/180;                   % seat tube angle [rad]
lst     = 0.545;                         % seat tube length [m]
tr      = 0.072;                         % trail [m]
lambda  = pi/2 - 0.350;                  % head tube angle [rad]
fo      = rf*cos(lambda)-tr*sin(lambda); % fork offset [m]
lf      = 0.467;                         % fork length [m]
lsp     = 0.206;                         % seat post length [m]
wr      = 0.130;                         % rear hub width [m]
wf      = 0.124;                         % front hub width [m]
whb     = 0.535;                         % handle bar width [m]
lhb     = 0.024;                         % handle bar length [m]
hs      = 0.335;                         % stem height [m]
rho     = 7850;                          % frame density [kg/m^3]
%-------------------Human Parameters (Jason Moore)
lth     = 0.46;                          % thigh length [m]
lc      = 0.46;                          % calf length [m]
lto     = 0.48;                          % torso length [m]
alphah  = 73.7*pi/180;                   % hunch angle [rad]
lua     = 0.2794;                        % upper arm length [m]
lla     = 0.3302;                        % lower arm length [m]
ch      = 0.58;                          % head circumference [m]
rh      = ch/2/pi;                       % head radius [m]
mr      = 83.5;                          % mass of rider [kg]
lhh     = 0.26;                          % hip joint to hip joint [m]
cth     = 0.5;                           % thigh circumference [m]
cc      = 0.38;                          % calf circumferenece [m]
cua     = 0.30;                          % upper arm circumference [m]
cla     = 0.23;                          % lower arm circumference [m]
cch     = 0.94;                          % chest circumference [m]
lss     = 0.44;                          % shoulder joint to shoulder joint [m]
%-------------------Grid Point Matrix
% calculates important points from the given frame and human dimensions
grid(1,1:3)  = [0 0 0];
grid(2,1:3)  = [0 0 rr];
grid(3,1:3)  = [0 wr/2 grid(2,3)];
grid(4,1:3)  = [0 -wr/2 grid(2,3)];
grid(5,1:3)  = [sqrt(lcs^2-(rr-hb)^2) 0 hb];
grid(6,1:3)  = [wb 0 0];
grid(7,1:3)  = [grid(6,1) 0 rf];
grid(8,1:3)  = [grid(6,1) wf/2 grid(7,3)];
grid(9,1:3)  = [grid(6,1) -wf/2 grid(7,3)];
grid(10,1:3) = [grid(5,1)-lst*cos(alphast) 0 grid(5,3)+lst*sin(alphast)];
grid(11,1:3) = [grid(7,1)-fo*sin(lambda)-sqrt(lf^2-fo^2)*cos(lambda),  0,     ...
                grid(7,3)-fo*cos(lambda)+sqrt(lf^2-fo^2)*sin(lambda)];
grid(12,1:3) = [grid(11,1)+(grid(11,3)-grid(10,3))/tan(lambda) 0 grid(10,3)];
grid(13,1:3) = [grid(10,1)-lsp*cos(alphast) 0 grid(10,3)+lsp*sin(alphast)];
% find the zero of the nonlinear relationship for leg position
x5 = grid(5,1); x13 = grid(13,1); z5 = grid(5,3); z13 = grid(13,3);
d = fzero(@leg,0.25,[],x5,x13,z5,z13,lth,lc);
b=sqrt(lc^2-d^2);
grid(14,1:3) = [grid(5,1)+d 0 grid(5,3)+b];
grid(15,1:3) = [grid(13,1)+lto*cos(alphah) 0 grid(13,3)+lto*sin(alphah)];
grid(16,1:3) = [grid(12,1)-hs*cos(lambda) 0 grid(12,3)+hs*sin(lambda)];
grid(17,1:3) = [grid(16,1) lss/2 grid(16,3)];
grid(18,1:3) = [grid(16,1) -lss/2 grid(16,3)];
grid(19,1:3) = [grid(17,1)-lhb grid(17,2) grid(17,3)];
grid(20,1:3) = [grid(18,1)-lhb grid(18,2) grid(18,3)];
grid(21,1:3) = [grid(15,1) grid(17,2) grid(15,3)];
grid(22,1:3) = [grid(15,1) grid(18,2) grid(15,3)];
% find the zero of the nonlinear relationship for arm position
z21 = grid(21,3); z19 = grid(19,3); x19 = grid(19,1); x21 = grid(21,1);
d = fzero(@arm,0.25,[],x19,x21,z19,z21,lla,lua);
c=sqrt(lla^2-d^2);
grid(23,1:3) = [grid(19,1)-d grid(17,2) grid(19,3)+c];
grid(24,1:3) = [grid(23,1) grid(18,2) grid(23,3)];
grid(25,1:3) = [grid(15,1)+rh*cos(alphah) 0 grid(15,3)+rh*sin(alphah)];
grid(26,1:3) = grid(5,:) + [0 lhh/2 0]; % left foot
grid(27,1:3) = grid(5,:) + [0 -lhh/2 0]; % right foot
grid(28,1:3) = grid(14,:) + [0 lhh/2 0]; % left knee
grid(29,1:3) = grid(14,:) + [0 -lhh/2 0]; % right knee
grid(30,1:3) = grid(13,:) + [0 lhh/2 0]; % left hip
grid(31,1:3) = grid(13,:) + [0 -lhh/2 0]; % right hip
%-------------------Element Data Matrix
% column 1: grid point (center for wheels and spheres), starting point for
% others
% column 2: grid point (center for wheels and spheres), ending point for
% others
% column 3: element description
% column 4: element type
% column 5: rigid body element belongs to
% column 6: outer radius for rings and spheres [m], outer radius for tubes [m],
% outer radius for cylinders [m], thickness for rect. prisms [m]
% column 7: tube radius for rings [m], wall thickness for tubes [m], width for
% rect. prisms [m]
ele( 1,1:7) = {  2  2 'rear wheel'       'ring'     'rwheel' rr     0.0215 };
ele( 2,1:7) = {  3  5 'left chain stay'  'tube'     'framex'  0.009  0.0014 };
ele( 3,1:7) = {  4  5 'right chain stay' 'tube'     'framex'  0.009  0.0014 };
ele( 4,1:7) = {  3 10 'left seat stay'   'tube'     'framex'  0.007  0.0014 };
ele( 5,1:7) = {  4 10 'right seat stay'  'tube'     'framex'  0.007  0.0014 };
ele( 6,1:7) = {  5 13 'seat tube'        'tube'     'framex'  0.014  0.0014 };
ele( 7,1:7) = {  5 11 'down tube'        'tube'     'framex'  0.014  0.0014 };
ele( 8,1:7) = { 10 12 'top tube'         'tube'     'framex'  0.0125 0.0014 };
ele( 9,1:7) = { 11 12 'head tube'        'tube'     'framex'  0.016   0.0014 };
ele(10,1:7) = { 11 16 'fork tube'        'tube'     'fork'   0.0125   0.0014 };
ele(11,1:7) = {  8 11 'left fork blade'  'tube'     'fork'   0.011    0.0014 };
ele(12,1:7) = {  9 11 'right fork blade' 'tube'     'fork'   0.011    0.0014 };
ele(13,1:7) = {  7  7 'front wheel'      'ring'     'fwheel' rf       0.0215 };
ele(14,1:7) = { 17 18 'handle bar'       'tube'     'fork'   0.011    0.0014 };
ele(15,1:7) = { 17 19 'left handle'      'tube'     'fork'   0.011    0.0014 };
ele(16,1:7) = { 18 20 'right handle'     'tube'     'fork'   0.011    0.0014 };
ele(17,1:7) = { 26 28 'left calf'        'cylinder' 'frame'  cc/2/pi  0      };
ele(18,1:7) = { 27 29 'right calf'       'cylinder' 'frame'  cc/2/pi  0      };
ele(19,1:7) = { 30 28 'left thigh'       'cylinder' 'frame'  cth/2/pi 0      };
ele(20,1:7) = { 31 29 'right thigh'      'cylinder' 'frame'  cth/2/pi 0      };
ele(21,1:7) = { 13 15 'torso'            'rprism'   'frame'  (cch-2*lss+2/pi*cua)/(pi-2)  lss-cua/pi};
ele(22,1:7) = { 25 25 'head'             'sphere'   'frame'  rh       0      };
ele(23,1:7) = { 21 23 'left upper arm'   'cylinder' 'frame'  cua/2/pi 0      };
ele(24,1:7) = { 22 24 'right upper arm'  'cylinder' 'frame'  cua/2/pi 0      };
ele(25,1:7) = { 19 23 'left lower arm'   'cylinder' 'frame'  cla/2/pi 0      };
ele(26,1:7) = { 20 24 'right lower arm'  'cylinder' 'frame'  cla/2/pi 0      };
%-------------------Draw 2D Bicycle
figure(1)
axis([-rr-0.1 wb+rf+0.1 0 grid(ele{22,1},3)+rh+0.1])
hold on
for i = 1:size(ele,1)
    switch ele{i,4}
        case {'ring','sphere'} % plot circles
            center = [grid(ele{i,1},1);grid(ele{i,1},3)];
            radius = ele{i,6};
            thetaplot = 0:0.001:2*pi;
            xplot = radius*cos(thetaplot)+center(1);
            yplot = radius*sin(thetaplot)+center(2);
            plot(xplot,yplot)
        case {'tube','rprism','cylinder'} % plot lines
            if i == 14 % don't plot handle bar
            elseif grid(ele{i,1},1) < grid(ele{i,2},1)
                x1 = grid(ele{i,1},1);
                x2 = grid(ele{i,2},1);
                z1 = grid(ele{i,1},3);
                z2 = grid(ele{i,2},3);
                xplot=x1:0.001:x2;
                yplot=(z2-z1)/(x2-x1).*(xplot-x1)+z1;
                plot(xplot,yplot)
            else
                x2 = grid(ele{i,1},1);
                x1 = grid(ele{i,2},1);
                z2 = grid(ele{i,1},3);
                z1 = grid(ele{i,2},3);
                xplot=x1:0.001:x2;
                yplot=(z2-z1)/(x2-x1).*(xplot-x1)+z1;
                plot(xplot,yplot)
            end
    end
end
hold off
%-------------------Calculate Element Length
elelength=zeros(size(ele,1),1);
for i = 1:size(ele,1)
    x1 = grid(ele{i,1},1);
    x2 = grid(ele{i,2},1);
    y1 = grid(ele{i,1},2);
    y2 = grid(ele{i,2},2);
    z1 = grid(ele{i,1},3);
    z2 = grid(ele{i,2},3);
    elelength(i)=sqrt((x2-x1)^2+(y2-y1)^2+(z2-z1)^2);
end
%-------------------Calculate Element Mass
elemass = zeros(size(ele,1),1); % initialize mass vector
% define masses of human body parts and wheels
elemass(1)  = 6.67*rr; % rear wheel weight
elemass(13) = 8.57*rf; % front wheel weight
elemass(17) = .122*mr/2;  % lef calf weight
elemass(18) = .122*mr/2;  % right calf weight
elemass(19) = .2*mr/2;  % left thigh weight
elemass(20) = .2*mr/2;  % right thigh weight
elemass(21) = .51*mr;   % torso weight
elemass(22) = .068*mr;   % head weight
elemass(23) = .028*mr; % left upper arm weight
elemass(24) = .028*mr; % right upper arm weight
elemass(25) = .022*mr; % left lower arm weight
elemass(26) = .022*mr; % right lower arm weightt
% calculate the mass of the frame and fork tube elements
for i = 1:size(ele,1)
    switch ele{i,4}
        case 'tube'
        elemass(i) = rho*pi*elelength(i)*(2*ele{i,6}*ele{i,7}-(ele{i,7})^2);
    end
end
%-------------------Calculate Total Mass of Each Rigid Body
% initialize mass summation variables
framemass = 0;
forkmass = 0;
rwheelmass = 0;
fwheelmass = 0;
% sum the mass from each element for each body
for i = 1:size(ele,1)
    switch ele{i,5}
        case 'frame'
            framemass = framemass + elemass(i);
        case 'fork'
            forkmass = forkmass + elemass(i);
        case 'rwheel'
            rwheelmass = rwheelmass + elemass(i);
        case 'fwheel'
            fwheelmass = fwheelmass + elemass(i);
    end
end
%-------------------Element Center of Mass
hold on
elecenter=zeros(size(ele,1),3);
for i = 1:size(ele,1)
    x1 = grid(ele{i,1},1);
    x2 = grid(ele{i,2},1);
    y1 = grid(ele{i,1},2);
    y2 = grid(ele{i,2},2);
    z1 = grid(ele{i,1},3);
    z2 = grid(ele{i,2},3);
    elecenter(i,1:3) = [(x1+x2)/2,(y1+y2)/2,(z1+z2)/2]; % midpoint formula
    plot(elecenter(i,1),elecenter(i,3),'ro') % plot element centers on 2D figure
end
hold off
%-------------------Rigid Body Center of Mass
% initialize center of mass vectors
framecenter  = [0,0,0];
forkcenter   = [0,0,0];
rwheelcenter = [0,0,0];
fwheelcenter = [0,0,0];
for i = 1:size(ele,1)
    switch ele{i,5}
        case 'frame'
            framecenter  = framecenter  + [elecenter(i,1).*elemass(i),...
                    elecenter(i,2).*elemass(i),elecenter(i,3).*elemass(i)];
        case 'fork'
            forkcenter   = forkcenter   + [elecenter(i,1).*elemass(i),...
                    elecenter(i,2).*elemass(i),elecenter(i,3).*elemass(i)];
        case 'rwheel'
            rwheelcenter = rwheelcenter + [elecenter(i,1).*elemass(i),...
                    elecenter(i,2).*elemass(i),elecenter(i,3).*elemass(i)];
        case 'fwheel'
            fwheelcenter = fwheelcenter + [elecenter(i,1).*elemass(i),...
                    elecenter(i,2).*elemass(i),elecenter(i,3).*elemass(i)];
    end
end
% divide summation by total mass of each body
framecenter  = framecenter./framemass;
forkcenter   = forkcenter./forkmass;
rwheelcenter = rwheelcenter./rwheelmass;
fwheelcenter = fwheelcenter./fwheelmass;
hold on
% plot rigid body centers of mass on 2D figure
plot([framecenter(1);forkcenter(1);rwheelcenter(1);fwheelcenter(1)],...
     [framecenter(3);forkcenter(3);rwheelcenter(3);fwheelcenter(3)],'go')
hold off
%-------------------Element Inertia Tensor
% calculate inertia tensors about local reference frame and element center of 
% mass
eleinertialocal=cell(size(ele,1));
for i = 1:size(ele,1)
    switch ele{i,4}
        case 'tube' % z axis is always along length of tube
            eleinertialocal{i} = ...
  [1/12*elemass(i)*(3*((ele{i,6})^2+(ele{i,6}-ele{i,7})^2)+(elelength(i))^2),0,0;
   0,1/12*elemass(i)*(3*((ele{i,6})^2+(ele{i,6}-ele{i,7})^2)+(elelength(i))^2),0;
   0,0,1/2*elemass(i)*((ele{i,6})^2+(ele{i,6}-ele{i,7})^2)];
        case 'rprism' % z axis is always along length of prism
            eleinertialocal{i} = ...
                [1/12*elemass(i)*((ele{i,7})^2+(elelength(i))^2),0,0;
                 0,1/12*elemass(i)*((ele{i,6})^2+(elelength(i))^2),0;
                 0,0,1/12*elemass(i)*((ele{i,6})^2+(ele{i,7})^2)];
        case 'cylinder' % z axis is always along length of cylinder
            eleinertialocal{i} = ...
                [1/12*elemass(i)*(3*(ele{i,6})^2+(elelength(i))^2),0,0;
                 0,1/12*elemass(i)*(3*(ele{i,6})^2+(elelength(i))^2),0;
                 0,0,1/2*elemass(i)*(ele{i,6})^2];
        case 'ring'
            eleinertialocal{i} = ...
                [1/8*(5*(ele{i,7})^2+4*(ele{i,6}-ele{i,7})^2)*elemass(i),0,0;
                 0, (3/4*(ele{i,7})^2+(ele{i,6}-ele{i,7})^2)*elemass(i),0;
                 0,0,1/8*(5*(ele{i,7})^2+4*(ele{i,6}-ele{i,7})^2)*elemass(i)];
        case 'sphere'
            eleinertialocal{i} = [2/5*elemass(i)*(ele{i,6})^2,0,0;
                                  0,2/5*elemass(i)*(ele{i,6})^2,0;
                                  0,0,2/5*elemass(i)*(ele{i,6})^2];
    end
end
%-------------------Element Unit Vectors
% calculate unit vectors for each element in local reference frame 
% local z vector
uveczlocal=zeros(size(ele,1),3);
veczlocal=zeros(size(ele,1),3);
mag=zeros(size(ele,1),1);
for i = 1:size(ele,1)
    switch ele{i,4}
        case {'ring','sphere'}
            uveczlocal(i,1:3) = [0,0,1]; % always in global z direction
        otherwise
            veczlocal(i,1:3) = [grid(ele{i,2},1)-grid(ele{i,1},1),...
                                grid(ele{i,2},2)-grid(ele{i,1},2),...
                                grid(ele{i,2},3)-grid(ele{i,1},3)];
            mag(i,1) = sqrt(sum((veczlocal(i,1:3).^2)'));
            uveczlocal(i,1:3) = veczlocal(i,1:3)./mag(i);
    end
end
% local y vector
vecylocal=zeros(size(ele,1),3);
uvecylocal=zeros(size(ele,1),3);
mag=zeros(size(ele,1),1);
for i = 1:size(ele,1)
    switch ele{i,3}
        % make y unit vector normal to the plane defined by the seat stays
        case {'right seat stay','right chain stay'}
            vecylocal(i,1:3) = cross(uveczlocal(5,1:3),uveczlocal(3,1:3));
            mag(i,1) = sqrt(sum((vecylocal(i,1:3).^2)'));
            uvecylocal(i,1:3) = vecylocal(i,1:3)./mag(i);
        % make y unit vector normal to the plane defined by the chain stays    
        case {'left seat stay','left chain stay'}
            vecylocal(i,1:3) = cross(uveczlocal(4,1:3),uveczlocal(2,1:3));
            mag(i,1) = sqrt(sum((vecylocal(i,1:3).^2)'));
            uvecylocal(i,1:3) = vecylocal(i,1:3)./mag(i);
        % make y unit vector normal to the plane defined by the fork blades    
        case {'right fork blade','left fork blade'}
            vecylocal(i,1:3) = cross(uveczlocal(11,1:3),uveczlocal(12,1:3));
            mag(i,1) = sqrt(sum((vecylocal(i,1:3).^2)'));
            uvecylocal(i,1:3) = vecylocal(i,1:3)./mag(i);
        case 'handle bar'
            uvecylocal(i,1:3) = [1,0,0];
        otherwise
            uvecylocal(i,1:3) = [0,1,0];
    end
end
% local x vector
uvecxlocal=zeros(size(ele,1),3);
for i = 1:size(ele,1)
    uvecxlocal(i,1:3) = cross(uvecylocal(i,1:3),uveczlocal(i,1:3));
end
%-------------------Element Direction Cosines Relative to Global Frame
xhat = [1,0,0];
yhat = [0,1,0];
zhat = [0,0,1];
% construct direction cosine matrices for each element
dircos=cell(size(ele,1));
for i = 1:size(ele,1)
    dircos{i} = [dot(xhat,uvecxlocal(i,1:3)), dot(xhat,uvecylocal(i,1:3)),...
                 dot(xhat,uveczlocal(i,1:3));
                 dot(yhat,uvecxlocal(i,1:3)), dot(yhat,uvecylocal(i,1:3)),...
                 dot(yhat,uveczlocal(i,1:3));
                 dot(zhat,uvecxlocal(i,1:3)), dot(zhat,uvecylocal(i,1:3)),...
                 dot(zhat,uveczlocal(i,1:3))];
end
%-------------------Rotate Element Interia Tensors to Global Frame
eleinertiaglobal=cell(size(ele,1));
for i = 1:size(ele,1)
    eleinertiaglobal{i} = dircos{i}*eleinertialocal{i}*(dircos{i})';
end
%-------------------Translate Inertia Tensors to the Center of Mass of Each Body
% use the parallel axis thereom to translate the inertia tensors
eleinertiatrans=cell(size(ele,1));
for i = 1:size(ele,1)
    switch ele{i,5}
        case 'frame'
            a = framecenter(1) - elecenter(i,1);
            b = framecenter(2) - elecenter(i,2);
            c = framecenter(3) - elecenter(i,3);
            eleinertiatrans{i} = eleinertiaglobal{i} + elemass(i)*[b^2+c^2,...
                                 -a*b, -a*c;-a*b,c^2+a^2,-b*c;-a*c,-b*c,a^2+b^2];
        case 'fork'
            a = forkcenter(1) - elecenter(i,1);
            b = forkcenter(2) - elecenter(i,2);
            c = forkcenter(3) - elecenter(i,3);
            eleinertiatrans{i} = eleinertiaglobal{i} + elemass(i)*[b^2+c^2,...
                                 -a*b, -a*c;-a*b,c^2+a^2,-b*c;-a*c,-b*c,a^2+b^2];
        case 'rwheel'
            a = rwheelcenter(1) - elecenter(i,1);
            b = rwheelcenter(2) - elecenter(i,2);
            c = rwheelcenter(3) - elecenter(i,3);
            eleinertiatrans{i} = eleinertiaglobal{i} + elemass(i)*[b^2+c^2,...
                                 -a*b, -a*c;-a*b,c^2+a^2,-b*c;-a*c,-b*c,a^2+b^2];
        case 'fwheel'
            a = fwheelcenter(1) - elecenter(i,1);
            b = fwheelcenter(2) - elecenter(i,2);
            c = fwheelcenter(3) - elecenter(i,3);
            eleinertiatrans{i} = eleinertiaglobal{i} + elemass(i)*[b^2+c^2,...
                                 -a*b, -a*c;-a*b,c^2+a^2,-b*c;-a*c,-b*c,a^2+b^2];
    end
end
%-------------------Sum the Inertia Tensors for Each Rigid Body
% intialize inertia matrices
frameinertia = zeros(3);
forkinertia = zeros(3);
rwheelinertia = zeros(3);
fwheelinertia = zeros(3);
for i = 1:size(ele,1)
    switch ele{i,5}
        case 'frame'
            frameinertia  = frameinertia  + eleinertiatrans{i};
        case 'fork'
            forkinertia   = forkinertia   + eleinertiatrans{i};
        case 'rwheel'
            rwheelinertia = rwheelinertia + eleinertiatrans{i};
        case 'fwheel'
            fwheelinertia = fwheelinertia + eleinertiatrans{i};
    end
end
%-------------------Rotate to the benchmark coordinate system
rot=[1 0 0;0 -1 0;0 0 -1];
framecenter  = rot*framecenter';
forkcenter   = rot*forkcenter';
rwheelcenter = rot*rwheelcenter';
fwheelcenter = rot*fwheelcenter';
frameinertia=rot*frameinertia*rot';
forkinertia=rot*forkinertia*rot';
rwheelinertia=rot*rwheelinertia*rot';
fwheelinertia=rot*fwheelinertia*rot';
%-------------------Generate Benchmark Parameters
bench_par=zeros(25,1);
bench_par(1)=tr;                  % C:      TRAIL                              [M]
bench_par(2)=9.81;                % G:      GRAVITY                            [N/KG]
bench_par(3)=frameinertia(1,1);   % IB_XX:  REAR BODY MASS MOMENT OF INERTIA   [KG*M^2] 
bench_par(4)=frameinertia(1,3);   % IB_XZ:  REAR BODY MASS MOMENT OF INERTIA   [KG*M^2]
bench_par(5)=frameinertia(3,3);   % IB_ZZ:  REAR BODY MASS MOMENT OF INERTIA   [KG*M^2]
bench_par(6)=fwheelinertia(1,1);  % IF_XX:  FRONT WHEEL MASS MOMENT OF INERTIA [KG*M^2]
bench_par(7)=fwheelinertia(2,2);  % IF_YY:  FRONT WHEEL MASS MOMENT OF INERTIA [KG*M^2]
bench_par(8)=forkinertia(1,1);    % IH_XX:  FORK MASS MOMENT OF INERTIA        [KG*M^2]
bench_par(9)=forkinertia(1,3);    % IH_XZ:  FORK MASS MOMENT OF INERTIA        [KG*M^2]
bench_par(10)=forkinertia(3,3);   % IH_ZZ:  FORK MASS MOMENT OF INERTIA        [KG*M^2]
bench_par(11)=rwheelinertia(1,1); % I_RXX:  REAR WHEEL MASS MOMENT OF INERTIA  [KG*M^2]
bench_par(12)=rwheelinertia(2,2); % I_RYY:  REAR WHEEL MASS MOMENT OF INERTIA  [KG*M^2]
bench_par(13)=pi/2 - lambda;      % LAMBDA: STEER AXIS TILT                    [RAD]
bench_par(14)=framemass;          % M_B:    REAR BODY MASS                     [KG] 
bench_par(15)=fwheelmass;         % M_F:    FRONT WHEEL MASS                   [KG]
bench_par(16)=forkmass;           % M_H:    FORK MASS                          [KG]
bench_par(17)=rwheelmass;         % M_R:    REAR WHEEL MASS                    [KG]
bench_par(18)=rf;                 % RF:     FRONT WHEEL RADIUS                 [M]
bench_par(19)=rr;                 % RR:     REAR WHEEL RADIUS                  [M]
bench_par(21)=wb;                 % W:      WHEELBASE                          [M]
bench_par(22)=framecenter(1);     % X_B:    REAR BODY CENTER OF MASS LOCATION  [M]
bench_par(23)=forkcenter(1);      % X_H:    FORK CENTER OF MASS LOCATION       [M]
bench_par(24)=framecenter(3);     % Z_B:    REAR BODY CENTER OF MASS LOCATION  [M]
bench_par(25)=forkcenter(3);      % Z_H:    FORK CENTER OF MASS LOCATION       [M]
% %-------------------Plot Autolev Representative Frame on 2D Figure
% hold on
% x1 = grid(ele{1,1},1);
% z1 = grid(ele{1,1},3);
% xplot=x1:0.001:L1;
% for i=1:length(xplot)
%     yplot(i)=z1;
% end
% plot(xplot,yplot,'k')
% endptx = xplot(length(xplot));
% endptz = yplot(length(yplot));
% x2 = grid(ele{13,1},1);
% z2 = grid(ele{13,1},3);
% x1 = x2 - L3*cos(THETA);
% z1 = z2 - L3*sin(THETA);
% xplot=x1:0.001:x2;
% yplot=(z2-z1)/(x2-x1).*(xplot-x1)+z1;
% plot(xplot,yplot,'k')
% endptx2 = x1;
% endptz2 = z1;
% if endptx < endptx2
%     x1 = endptx;
%     z1 = endptz;
%     x2 = endptx2;
%     z2 = endptz2;
%     xplot=x1:0.001:x2;
%     yplot=(z2-z1)/(x2-x1).*(xplot-x1)+z1;
%     plot(xplot,yplot,'k')
% else
%     x2 = endptx;
%     z2 = endptz;
%     x1 = endptx2;
%     z1 = endptz2;
%     xplot=x1:0.001:x2;
%     yplot=(z2-z1)/(x2-x1).*(xplot-x1)+z1;
%     plot(xplot,yplot,'k')
% end
% hold off
