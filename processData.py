import scipy.io.matlab.mio as mio
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import os
import re
import pickle as p
from BenchmarkBikeTools import *

# load the main data file into a dictionary
d = {}
mio.loadmat('data/data.mat', mdict=d)
# the number of different bikes
nBk = len(d['bikes'])
#print "Number of bikes =", nBk
# make a list of the bikes' names
bikeNames = []
for bike in d['bikes']:
    # get rid of the weird matlab unicoding
    bikeNames.append(bike[0][0].encode('ascii'))
#print "List of bikes:\n", bikeNames
# calculate the wheel radii
rearWheelRadius = d['rearWheelDist'][0]/2/np.pi/d['rearWheelRot'][0]
#print "Rear wheel radii [m] =\n", rearWheelRadius
frontWheelRadius = d['frontWheelDist'][0]/2/np.pi/d['frontWheelRot'][0]
#print "Front wheel radii [m] =\n", frontWheelRadius
# steer axis tilt in radians
steerAxisTilt = np.pi/180*(90-d['headTubeAngle'][0])
#print "Steer axis tilt [deg] =\n", steerAxisTilt*180./np.pi
# calculate the front wheel trail
forkOffset = d['forkOffset'][0]
trail = (frontWheelRadius*np.sin(steerAxisTilt) - forkOffset)/np.cos(steerAxisTilt)
#print "Trail [m] =\n", trail
# calculate the frame rotation angle
alphaFrame = d['frameAngle']
#print "alphaFrame =\n", alphaFrame
betaFrame = steerAxisTilt - alphaFrame*np.pi/180
#print "Frame rotation angle, beta [deg] =\n", betaFrame/np.pi*180.
# calculate the slope of the CoM line
frameM = -np.tan(betaFrame)
#print "Frame CoM line slope =\n", frameM
# calculate the z-intercept of the CoM line
frameMassDist = d['frameMassDist']
#print "Frame CoM distance =\n", d['frameMassDist']
frameB = frameMassDist/np.cos(betaFrame) - rearWheelRadius
#print "Frame CoM line intercept =\n", frameB
# calculate the fork rotation angle
betaFork = steerAxisTilt - d['forkAngle']*np.pi/180.
#print "Fork rotation angle [deg] =\n", betaFork*180./np.pi
# calculate the slope of the fork CoM line
forkM = -np.tan(betaFork)
#print "Fork CoM line slope =\n", frameM
# calculate the z-intercept of the CoM line
wheelBase = d['wheelbase'][0]
forkMassDist = d['forkMassDist']
forkB = - frontWheelRadius + forkMassDist/np.cos(betaFork) + wheelBase*np.tan(betaFork)
#print "Fork CoM line intercept =\n", frameB
# plot the CoM lines
plt.figure()
# intialize the matrices for the center of mass locations
frameCoM = np.zeros((2, np.shape(frameM)[1]))
forkCoM = np.zeros((2, np.shape(forkM)[1]))
# for each of the bikes...
for i in range(np.shape(frameM)[1]):
    comb = np.array([[0, 1], [0, 2], [1, 2]])
    # calculate the frame center of mass position
    # initialize the matrix to store the line intersections
    lineX = np.zeros((3, 2))
    # for each line intersection...
    for j, row in enumerate(comb):
        a = np.vstack([-frameM[row, i], np.ones((2))]).T
        b = frameB[row, i]
        lineX[j] = np.linalg.solve(a, b)
    frameCoM[:, i] = np.mean(lineX, axis=0)
    # calculate the fork center of mass position
    # reinitialize the matrix to store the line intersections
    lineX = np.zeros((3, 2))
    # for each line intersection...
    for j, row in enumerate(comb):
        a = np.vstack([-forkM[row, i], np.ones((2))]).T
        b = forkB[row, i]
        lineX[j] = np.linalg.solve(a, b)
    forkCoM[:, i] = np.mean(lineX, axis=0)
    # make a subplot for this bike
    plt.subplot(2, 4, i + 1)
    # plot the rear wheel
    c=plt.Circle((0, rearWheelRadius[i]), radius=rearWheelRadius[i])
    plt.gca().add_patch(c)
    # plot the front wheel
    c=plt.Circle((d['wheelbase'][0][i], frontWheelRadius[i]), radius=frontWheelRadius[i])
    plt.gca().add_patch(c)
    # plot the lines (pendulum axes)
    x = np.linspace(-rearWheelRadius[i], d['wheelbase'][0][i] +
            frontWheelRadius[i], 2)
    # for each line...
    for j in range(len(frameM)):
        framey = -frameM[j, i]*x - frameB[j, i]
        forky = -forkM[j, i]*x - forkB[j, i]
        plt.plot(x,framey, 'r')
        plt.plot(x,forky, 'g')
    # plot the ground line
    plt.plot(x, np.zeros_like(x), 'k')
    # plot the centers of mass
    plt.plot(frameCoM[0, i], -frameCoM[1, i], 'k+', markersize=12)
    plt.plot(forkCoM[0, i], -forkCoM[1, i], 'k+', markersize=12)
    plt.axis('equal')
    plt.ylim((0, 1))
    plt.title(bikeNames[i])
plt.show()
#print "Frame CoM =\n", frameCoM
#print "Fork CoM =\n", forkCoM
# load the average period data
f = open('avgPer.p', 'r')
avgPer = p.load(f)
f.close()
# torsional, compound and rod periods
tor = avgPer['tor']
com = avgPer['com']
tRod = avgPer['rodPer']
# calculate the stiffness of the torsional pendulum
mRod = 5.56 # mass of the calibration rod [kg]
lRod = 1.05 # length of the calibration rod [m]
rRod = 0.015 # radius of the calibration rod [m]
iRod = tube_inertia(lRod, mRod, rRod, 0.)[1]
k = tor_stiffness(iRod, tRod)
# calculate the wheel y inertias
g = 9.81
IRyy = com_inertia(d['rearWheelMass'], g, d['rWheelPendLength'], com[3, :])
IFyy = com_inertia(d['frontWheelMass'], g, d['fWheelPendLength'], com[2, :])
# calculate the wheel x/z inertias
IRxx = tor_inertia(k, tor[6, :])
IFxx = tor_inertia(k, tor[9, :])
# calculate the y inertias for the frame and fork
com[1, 0] = com[1, 1]
com[0, 7] = com[0, 6]
framePendLength = np.sqrt(frameCoM[0, :]**2 + (frameCoM[1, :] + rearWheelRadius)**2)
IByy = com_inertia(d['frameMass'][0], g, framePendLength, com[0, :])
forkPendLength = np.sqrt((forkCoM[0, :] - d['wheelbase'][0])**2 + (forkCoM[1, ] + frontWheelRadius)**2)
IHyy = com_inertia(d['forkMass'][0], g, forkPendLength, com[1, :])
# calculate the fork in-plane moments of inertia
Ipend = tor_inertia(k, tor)
IHxx = []
IHxz = []
IHzz = []
for i, row in enumerate(Ipend[3:6, :].T):
    Imat = inertia_components(row, betaFork[:, i])
    IHxx.append(Imat[0, 0])
    IHxz.append(Imat[0, 1])
    IHzz.append(Imat[1, 1])
IHxx = np.array(IHxx)
IHxz = np.array(IHxz)
IHzz = np.array(IHzz)
# calculate the frame in-plane moments of inertia
IBxx = []
IBxz = []
IBzz = []
for i, row in enumerate(Ipend[:3, :].T):
    Imat = inertia_components(row, betaFrame[:, i])
    IBxx.append(Imat[0, 0])
    IBxz.append(Imat[0, 1])
    IBzz.append(Imat[1, 1])
IBxx = np.array(IBxx)
IBxz = np.array(IBxz)
IBzz = np.array(IBzz)
# write the parameter files
for i, name in enumerate(bikeNames):
    file = open(''.join(name.split() + 'Par.txt', 'w'))

