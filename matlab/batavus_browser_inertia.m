% File: batavus_browser_inertia.m
% Date: March 22, 2010
% Author: Jason Moore
% Description: Calculates the benchmark inertia matrices of the Batavus
% Browser frame and fork/handlebar from the experiments.
clear all
close all
clc
% set the y component equal to zero
IBFyy  = 0;
IHyy   = 0;
IBRAyy = 0;
IBRJyy = 0;
% Inertia tensors are defined as follows:
% I = [ Ixx -Ixy -Ixz
%      -Ixy  Iyy -Iyz
%      -Ixz -Iyz  Izz]
%------------Batavus Browser rear frame ----------------------------------%
mBF = 9.86;  % [kg] mass of the rear frame
xBF = 0.2799135; % [m] cg location in benchmark coordinate system
zBF = -0.53478492; % [m] cg location in the benchmark coordinate system
% a period of ten oscillations was measured with a pocket watch through 3
% different planar axes each 3 different times and averaged
TBF1 = 2.094431797861789679;   % [s] period of oscillation when hung from the top tube
TBF2 = 2.376844356021118987;   % [s] period of oscillation when hung from the down tube
TBF3 = 1.839869878050422791;   % [s] period of oscillation when hung from the fender
k   =  5.6248209199363393;  % [Nm/rad] stiffness of the torsion rod, from page 43 of
% moments of inertia [Nm*s^2] about the 1st, 2nd, and 3rd test axes
IBFx1 = k*TBF1^2/4/pi^2;
IBFx2 = k*TBF2^2/4/pi^2;
IBFx3 = k*TBF3^2/4/pi^2;
% angle [rad] between the test axes (x1,x2,x3) and the global benchmark
% x-axis (positive rotation about the y-axis)
alpha1 =  199.5/180*pi;
alpha2 =  64.2/180*pi;
alpha3 =  -26.90/180*pi;
% intermediate variables for sines and cosines
sa1 = sin(alpha1);
sa2 = sin(alpha2);
sa3 = sin(alpha3);
ca1 = cos(alpha1);
ca2 = cos(alpha2);
ca3 = cos(alpha3);
% define the relationship between the measured inertias and the unknown
% inertias
% x = [IBFxx;IBFxz;IBFzz]
% b = A*x
A = [ca1^2 2*sa1*ca1 sa1^2
     ca2^2 2*sa2*ca2 sa2^2
     ca3^2 2*sa3*ca3 sa3^2];
b = [IBFx1
     IBFx2
     IBFx3];
% solve the system
x = A\b;
IBFxx = x(1);
IBFxz = x(2);
IBFzz = x(3);
IBF = [IBFxx 0 -IBFxz;0 IBFyy 0;-IBFxz 0 IBFzz];
%-------------Batavus Browser front frame (handlbar and fork)-------------%
mH  =  3.22;    % [kg] mass of the front handlebar and fork assembly
xH  =  8.632286058007860863e-01;   % [m] cg location in benchmark coordinate system
zH  = -7.467237728401425745e-01;   % [m] cg location in the benchmark coordinate system
% a period of ten oscillations was measured with a pocket watch through 3
% different planar axes each 3 different times and averaged
TH1 =  1.36895152;     % [s] period of oscillation when hung from the steer tube
TH2 =  1.29537358;     % [s] period of oscillation when hung from the fender behind the fork
TH3 =  0.76099552;     % [s] period of oscillation when hung from the fender in front of the fork
k   =  5.6248209199363393;  % [Nm/rad] stiffness of the torsion rod, from page 43 of
% moments of inertia [Nm*s^2] about the 1st, 2nd, and 3rd test axes
IHx1 = k*TH1^2/4/pi^2;
IHx2 = k*TH2^2/4/pi^2;
IHx3 = k*TH3^2/4/pi^2;
% angle [rad] between the test axes (x1,x2,x3) and the global benchmark
% x-axis (positive rotation about the y-axis)
beta1 =  -143.8/180*pi;
beta2 = 174.8/180*pi;
beta3 = -84.1/180*pi;
% intermediate variables for sines and cosines
sb1 = sin(beta1);
sb2 = sin(beta2);
sb3 = sin(beta3);
cb1 = cos(beta1);
cb2 = cos(beta2);
cb3 = cos(beta3);
% define the relationship between the measured inertias and the unknown
% inertias
% x = [IBFxx;IBFxz;IBFzz]
% b = A*x
A = [cb1^2 2*sb1*cb1 sb1^2
     cb2^2 2*sb2*cb2 sb2^2
     cb3^2 2*sb3*cb3 sb3^2];
b = [IHx1
     IHx2
     IHx3];
% solve the system
x = A\b;
IHxx = x(1);
IHxz = x(2);
IHzz = x(3);
IH = [IHxx 0 -IHxz;0 IHyy 0;-IHxz 0 IHzz];
% input the inertia matrices about the CG of Arend and Jason from
% bike_inertia_arend_browser.m and bike_inertia_jason_browser.m
% respectively, these are with respect to the benchmark coordinate system
% Arend
IBRA = [12.2937 0 -3.5547;0 IBRAyy 0;-3.5547 0 4.6504];
mBRA = 102;      % [kg] total body mass
xBRA  = 0.3072;  % [m] x cg location
zBRA  = -1.1457; % [m] z cg location
% Jason
IBRJ = [7.9985 0 -1.9272;0 IBRJyy 0;-1.9272 0 2.3624];
mBRJ = 72;       % [kg] total body mass
xBRJ  = 0.2909;   % [m] x cg location
zBRJ  = -1.1091; % [m] z cg location
% find the new cg locations (frame combine with person)
mBA = mBRA+mBF;
mBJ = mBRJ+mBF;
xBA = (mBRA*xBRA+mBF*xBF)/(mBA);
zBA = (mBRA*zBRA+mBF*zBF)/(mBA);
xBJ = (mBRJ*xBRJ+mBF*xBF)/(mBJ);
zBJ = (mBRJ*zBRJ+mBF*zBF)/(mBJ);
% translate each inertia to the new CG location using the parrellel axis
% theorem
IBA = IBF + mBF*[(zBA-zBF)^2,0,-(xBA-xBF)*(zBA-zBF);0,(xBA-xBF)^2+(zBA-...
      zBF)^2,0;-(xBA-xBF)*(zBA-zBF),0,(xBA-xBF)^2] +...
      IBRA + mBRA*[(zBA-zBRA)^2,0,-(xBA-xBRA)*(zBA-zBRA);0,(xBA-xBRA)^2+...
      (zBA-zBRA)^2,0;-(xBA-xBRA)*(zBA-zBRA),0,(xBA-xBRA)^2];
IBJ = IBF + mBF*[(zBJ-zBF)^2,0,-(xBJ-xBF)*(zBJ-zBF);0,(xBJ-xBF)^2+(zBJ-...
      zBF)^2,0;-(xBJ-xBF)*(zBJ-zBF),0,(xBJ-xBF)^2] +...
      IBRJ + mBRJ*[(zBJ-zBRJ)^2,0,-(xBJ-xBRJ)*(zBJ-zBRJ);0,(xBJ-xBRJ)^2+...
      (zBJ-zBRJ)^2,0;-(xBJ-xBRJ)*(zBJ-zBRJ),0,(xBJ-xBRJ)^2];
% change coordinates from the benchmark paper to JBike6
C = [1 0 0;0 0 -1;0 1 0];
IBF_J6  = C*IBF*C';
IBA_J6 = C*IBA*C';
IBJ_J6 = C*IBJ*C';
IH_J6  = C*IH*C';
% find the Jbike6 principal inertia matrix and the principal axis angle
[vBPF,IBPF_J6] = eig(IBF_J6(1:2,1:2));
alphaBPF1 = (atan(vBPF(2,1)/vBPF(1,1)))*180/pi; % [deg] angle to the I1 axis
alphaBPF2 = (atan(vBPF(2,2)/vBPF(1,2)))*180/pi; % [deg] angle to the I2 axis

[vBPA,IBPA_J6] = eig(IBA_J6(1:2,1:2));
alphaBPA1 = (atan(vBPA(2,1)/vBPA(1,1)))*180/pi; % [deg] angle to the I1 axis
alphaBPA2 = (atan(vBPA(2,2)/vBPA(1,2)))*180/pi; % [deg] angle to the I2 axis

[vBPJ,IBPJ_J6] = eig(IBJ_J6(1:2,1:2)); 
alphaBPJ1 = (atan(vBPJ(2,1)/vBPJ(1,1)))*180/pi; % [deg] angle to the I1 axis
alphaBPJ2 = (atan(vBPJ(2,2)/vBPJ(1,2)))*180/pi; % [deg] angle to the I2 axis

[vHP,IHP_J6] = eig(IH_J6(1:2,1:2)); 
alphaHP1 = (atan(vHP(2,1)/vHP(1,1)))*180/pi; % [deg] angle to the I1 axis
alphaHP2 = (atan(vHP(2,2)/vHP(1,2)))*180/pi; % [deg] angle to the I2 axis
% display the results with NaN for y related components
disp('Browser rear frame')
IBF(2,2) = NaN;
IBF
mBF
xBF
zBF
disp('Browser fork/handlebar assembly')
IH(2,2) = NaN;
IH
mH
xH
zH
disp('Arend')
IBRA(2,2) = NaN;
IBRA
mBRA
xBRA
zBRA
disp('Jason')
IBRJ(2,2) = NaN;
IBRJ
mBRJ
xBRJ
zBRJ
disp('Arend + Frame')
IBA(2,2)= NaN;
IBA
mBA
xBA
zBA
disp('Jason + Frame')
IBJ(2,2) = NaN;
IBJ
mBJ
xBJ
zBJ
disp('Browser Frame, JBike6')
IBF_J6(3,3)= NaN;
IBF_J6
disp('Principal Inertia')
IBPF_J6
alphaBPF1
mBF_J6 = mBF
xBF_J6 = xBF
yBF_J6 = -zBF
disp('Arend + Frame, JBike6')
IBA_J6(3,3)= NaN;
IBA_J6
disp('Principal Inertia')
IBPA_J6
alphaBPA1
mBA_J6 = mBA
xBA_J6 = xBA
yBA_J6 = -zBA
disp('Jason + Frame, JBike6')
IBJ_J6(3,3) = NaN;
IBJ_J6
disp('Principal Inertia')
IBPJ_J6
alphaBPJ1
mBJ_J6 = mBJ
xBJ_J6 = xBJ
yBJ_J6 = -zBJ
disp('Browser fork/handlebar assembly, JBike6')
IH_J6(3,3) = NaN;
IH_J6
disp('Principal Inertia')
IHP_J6
alphaHP1
mH_J6 = mH
xH_J6 = xH
yH_J6 = -zH
