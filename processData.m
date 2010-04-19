clear all
close all
clc
load('data/data')
% calculate the wheel radii
rearWheelRadius = rearWheelDist./2./pi./rearWheelRot;
frontWheelRadius = frontWheelDist./2./pi./frontWheelRot;
% steer axis tilt in radians
steerAxisTilt = pi./180.*(90-headTubeAngle);
% calculate the front wheel trail
trail = (frontWheelRadius.*sin(steerAxisTilt)-forkOffset)./cos(steerAxisTilt);
% calculate the frame rotation angle
beta = frameAngle.*pi./180-pi/2*ones(size(frameAngle))-[steerAxisTilt;steerAxisTilt;steerAxisTilt];
% calculate the slope of the CoM line
frameM = tan(beta-pi/2);
% calculate the z-intercept of the CoM line
frameB = -frameMassDist./sin(beta)-[rearWheelRadius;rearWheelRadius;rearWheelRadius];
% calculate the center of mass position
for i = 1:length(frameM)
    CoM(1:2,i) = [-frameM(:,i) ones(3,1)]\frameB(:,i);
end
% calculate the fork rotation angle
betaFork = forkAngle.*pi./180-pi/2*ones(size(forkAngle))-[steerAxisTilt;steerAxisTilt;steerAxisTilt];
% calculate the slope of the fork CoM line
forkM = tan(betaFork-pi/2);
% calculate the z-intercept of the CoM line
forkB = [wheelbase;wheelbase;wheelbase]./tan(betaFork)-forkMassDist./sin(betaFork)-[frontWheelRadius;frontWheelRadius;frontWheelRadius];
% calculate the center of mass position
for i = 1:length(forkM)
    CoMFork(1:2,i) = [-forkM(:,i) ones(3,1)]\forkB(:,i);
end
% plot the CoM lines
for i = 1:length(frameM)
    figure(i)
    hold on
    axis([-1 2 -1 1])
    rectangle('position',[-rearWheelRadius(i),-2*rearWheelRadius(i),2*rearWheelRadius(i),2*rearWheelRadius(i)],'curvature',[1 1])
    first  = refline(frameM(1,i),frameB(1,i));
    second = refline(frameM(2,i),frameB(2,i));
    third  = refline(frameM(3,i),frameB(3,i));
    xAxis = refline(0,0);
    set(xAxis,'color','k')
    set(second,'color','r')
    set(third,'color','g')
    axis([-1 2 -1 1])
    title(bikes{i})
    hold off
end
% plot the fork CoM lines
for i = 1:length(forkM)
    figure(i)
    hold on
    first  = refline(forkM(1,i),forkB(1,i));
    second = refline(forkM(2,i),forkB(2,i));
    third  = refline(forkM(3,i),forkB(3,i));
    plot(CoM(1, i), CoM(2, i), 'k+');
    plot(CoMFork(1, i), CoMFork(2, i), 'k+');
    set(first,'color','c')
    set(second,'color','m')
    set(third,'color','y')
    axis equal
    axis([-1 2 -1 1])
    legend('First','Second','Third','','Fork First','Fork Second','Fork Third')
    hold off
end
