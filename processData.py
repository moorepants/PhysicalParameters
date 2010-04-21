import scipy.io.matlab.mio as mio
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import os
import re
import pickle as p

# load the main data file into a dictionary
d = {}
mio.loadmat('data/data.mat', mdict=d)
# the number of different bikes
nBk = len(d['bikes'])
print "Number of bikes =", nBk
# make a list of the bikes' names
bikeNames = []
for bike in d['bikes']:
    # get rid of the weird matlab unicoding
    bikeNames.append(bike[0][0].encode('ascii'))
print "List of bikes:\n", bikeNames
# calculate the wheel radii
rearWheelRadius = d['rearWheelDist'][0]/2/np.pi/d['rearWheelRot'][0]
print "Rear wheel radii [m] =\n", rearWheelRadius
frontWheelRadius = d['frontWheelDist'][0]/2/np.pi/d['frontWheelRot'][0]
print "Front wheel radii [m] =\n", frontWheelRadius
# steer axis tilt in radians
steerAxisTilt = np.pi/180*(90-d['headTubeAngle'][0])
print "Steer axis tilt [deg] =\n", steerAxisTilt*180./np.pi
# calculate the front wheel trail
forkOffset = d['forkOffset'][0]
trail = (frontWheelRadius*np.sin(steerAxisTilt) - forkOffset)/np.cos(steerAxisTilt)
print "Trail [m] =\n", trail
# calculate the frame rotation angle
satMat = np.array([steerAxisTilt, steerAxisTilt, steerAxisTilt])
alphaFrame = d['frameAngle']
print "alphaFrame =\n", alphaFrame
betaFrame = steerAxisTilt - alphaFrame*np.pi/180
#betaFrame = alphaFrame*np.pi/180 - np.pi/2.*np.ones_like(alphaFrame) - steerAxisTilt
#betaFrame = np.pi + steerAxisTilt - d['alphaFrame']*np.pi/180
# this flips beta such that it is always in the first two quadrants
#for i, row in enumerate(betaFrame):
#    for j, v in enumerate(row):
#        if v < 0:
#            betaFrame[i, j] = v + np.pi
#        else:
#            pass
#for i, row in enumerate(betaFrame):
#    for j, v in enumerate(row):
#        if v > np.pi:
#            betaFrame[i, j] = v - np.pi
#        else:
#            pass
print "Frame rotation angle, beta [deg] =\n", betaFrame/np.pi*180.
# calculate the slope of the CoM line
frameM = -np.tan(betaFrame)
print "Frame CoM line slope =\n", frameM
# calculate the z-intercept of the CoM line
print "Frame CoM distance =\n", d['frameMassDist']
#rwrMat =  np.array([rearWheelRadius, rearWheelRadius, rearWheelRadius])
#frameB = -d['frameMassDist']/np.sin(betaFrame) - rwrMat
frameMassDist = d['frameMassDist']
frameB = frameMassDist/np.cos(betaFrame) - rearWheelRadius
#frameB = d['frameMassDist']/np.cos(betaFrame) - rearWheelRadius
print "Frame CoM line intercept =\n", frameB
# calculate the fork rotation angle
betaFork = d['forkAngle']*np.pi/180-np.pi/2*np.ones_like(d['forkAngle']) - satMat
#betaFork = np.pi + steerAxisTilt - d['forkAngle']*np.pi/180
print "Fork rotation angle [deg] =\n", betaFork*180./np.pi
# calculate the slope of the fork CoM line
forkM = np.tan(betaFork - np.pi/2)
print "Fork CoM line slope =\n", frameM
# calculate the z-intercept of the CoM line
wbMat = np.array([d['wheelbase'][0], d['wheelbase'][0], d['wheelbase'][0]])
fwrMat =  np.array([frontWheelRadius, frontWheelRadius, frontWheelRadius])
forkB = wbMat/np.tan(betaFork) - d['forkMassDist']/np.sin(betaFork) - fwrMat
print "Fork CoM line intercept =\n", frameB
plt.figure()
# plot the CoM lines
frameCoM = np.zeros((2, np.shape(frameM)[1]))
forkCoM = np.zeros((2, np.shape(forkM)[1]))
for i in range(np.shape(frameM)[1]):
    # calculate the frame center of mass position
    a = np.vstack([-frameM[:, i], np.ones((3))]).T
    b = frameB[:, i]
    frameCoM[:, i] = np.linalg.lstsq(a, b)[0]
    # calculate the fork center of mass position
    a = np.vstack([-forkM[:, i], np.ones((3))]).T
    b = forkB[:, i]
    forkCoM[:, i] = np.linalg.lstsq(a, b)[0]
    plt.subplot(2, 4, i + 1)
    # plot the rear wheel
    c=plt.Circle((0, rearWheelRadius[i]), radius=rearWheelRadius[i])
    plt.gca().add_patch(c)
    # plot the front wheel
    c=plt.Circle((d['wheelbase'][0][i], frontWheelRadius[i]), radius=frontWheelRadius[i])
    plt.gca().add_patch(c)
    x = np.linspace(-rearWheelRadius[i], d['wheelbase'][0][i] +
            frontWheelRadius[i], 2)
    for j in range(len(frameM)):
        framey = -frameM[j, i]*x - frameB[j, i]
        forky = -forkM[j, i]*x - forkB[j, i]
        plt.plot(x,framey, 'r')
        plt.plot(x,forky, 'g')
    plt.plot(x, np.zeros_like(x), 'k')
    plt.plot(frameCoM[0, i], -frameCoM[1, i], 'k+', markersize=12)
    plt.plot(forkCoM[0, i], -forkCoM[1, i], 'k+', markersize=12)
    plt.axis('equal')
    plt.ylim((0, 1))
    plt.title(d['bikes'][i])
plt.show()
print "Frame CoM =\n", frameCoM
print "Fork CoM =\n", forkCoM
#m = forkM[:, i]
#b = forkB[:, i]
# least squares test
#x_lsq = (3.*(-m[0]*b[1]-m[1]*b[1]-m[2]*b[2])+(m[0]+m[1]+m[2])*(b[0]+b[1]+b[2]))/(3.*(m[0]**2+m[1]**2+m[2]**2)-(-m[0]-m[1]-m[2])**2)
#y_lsq = ((m[0]+m[1]+m[2])*(-m[0]*b[0]-m[1]*b[1]-m[2]*b[2])+(m[0]**2+m[1]**2+m[2]**2)*(b[0]+b[1]+b[2])/(3.*(m[0]**2+m[1]**2+m[2]**2)-(-m[0]-m[1]-m[2])**2))
#x_avg = 1./3.*((b[0]-b[1])/(-m[0]+m[1])+(b[0]-b[2])/(-m[0]+m[2])+(b[1]-b[2])/(-m[1]+m[2]))
#y_avg = 1./3.*((m[1]*b[0]-m[0]*b[1])/(-m[0]+m[1])+(m[2]*b[0]-m[0]*b[2])/(-m[0]+m[2])+(m[2]*b[1]-m[1]*b[2])/(-m[1]+m[2]))
f = open('avgPer.p', 'r')
avgPer = p.load(f)
f.close()
tor = avgPer['tor']
com = avgPer['com']
tRod = avgPer['rodPer']
# calculate the stiffness of the torsional pendulum
mRod = 5.56 # mass of the calibration rod [kg]
lRod = 1.05 # length of the calibration rod [m]
rRod = 0.015 # radius of the calibration rod [m]
iRod = 1./12.*mRod*(3*rRod**2 + lRod**2)
k = 4.*iRod*np.pi**2/tRod**2
# calculate the wheel y inertias
g = 9.81
IRyy = (com[3, :]/2./np.pi)**2*d['rearWheelMass']*g*d['rWheelPendLength']-d['rearWheelMass']*d['rWheelPendLength']**2
IFyy = (com[2, :]/2./np.pi)**2*d['frontWheelMass']*g*d['fWheelPendLength']-d['frontWheelMass']*d['fWheelPendLength']**2
# calculate the wheel x/z inertias
IRxx = k*tor[6, :]**2/4./np.pi**2
IFxx = k*tor[9, :]**2/4./np.pi**2
# calculate the y inertias for the frame and fork
com[1, 0] = com[1, 1]
com[0, 7] = com[0, 6]
framePendLength = np.sqrt(frameCoM[0, :]**2 + (frameCoM[1, :] + rearWheelRadius)**2)
IByy = (com[0, :]/2./np.pi)**2*d['frameMass'][0]*g*framePendLength-d['frameMass'][0]*framePendLength**2
forkPendLength = np.sqrt((forkCoM[0, :] - d['wheelbase'][0])**2 + (forkCoM[1, ] + frontWheelRadius)**2)
IHyy = (com[1, :]/2./np.pi)**2*d['forkMass'][0]*g*forkPendLength-d['forkMass'][0]*forkPendLength**2

