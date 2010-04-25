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
par = {}
# calculate the wheel radii
par['rR'] = d['rearWheelDist'][0]/2/np.pi/d['rearWheelRot'][0]
#print "Rear wheel radii [m] =\n", par['rR']
par['rF'] = d['frontWheelDist'][0]/2/np.pi/d['frontWheelRot'][0]
#print "Front wheel radii [m] =\n", par['rF']
# steer axis tilt in radians
par['lambda'] = np.pi/180*(90-d['headTubeAngle'][0])
#print "Steer axis tilt [deg] =\n", par['lambda']*180./np.pi
# calculate the front wheel trail
forkOffset = d['forkOffset'][0]
par['c'] = (par['rF']*np.sin(par['lambda']) - forkOffset)/np.cos(par['lambda'])
#print "Trail [m] =\n", par['c']
# calculate the frame rotation angle
alphaFrame = d['frameAngle']
#print "alphaFrame =\n", alphaFrame
betaFrame = par['lambda'] - alphaFrame*np.pi/180
#print "Frame rotation angle, beta [deg] =\n", betaFrame/np.pi*180.
# calculate the slope of the CoM line
frameM = -np.tan(betaFrame)
#print "Frame CoM line slope =\n", frameM
# calculate the z-intercept of the CoM line
frameMassDist = d['frameMassDist']
#print "Frame CoM distance =\n", d['frameMassDist']
frameB = frameMassDist/np.cos(betaFrame) - par['rR']
#print "Frame CoM line intercept =\n", frameB
# calculate the fork rotation angle
betaFork = par['lambda'] - d['forkAngle']*np.pi/180.
#print "Fork rotation angle [deg] =\n", betaFork*180./np.pi
# calculate the slope of the fork CoM line
forkM = -np.tan(betaFork)
#print "Fork CoM line slope =\n", frameM
# calculate the z-intercept of the CoM line
par['w'] = d['wheelbase'][0]
forkMassDist = d['forkMassDist']
forkB = - par['rF'] + forkMassDist/np.cos(betaFork) + par['w']*np.tan(betaFork)
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
    c=plt.Circle((0, par['rR'][i]), radius=par['rR'][i])
    plt.gca().add_patch(c)
    # plot the front wheel
    c=plt.Circle((par['w'][i], par['rF'][i]), radius=par['rF'][i])
    plt.gca().add_patch(c)
    # plot the lines (pendulum axes)
    x = np.linspace(-par['rR'][i], par['w'][i] +
            par['rF'][i], 2)
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
par['xB'] = frameCoM[0, :]
par['zB'] = frameCoM[1, :]
par['xH'] = forkCoM[0, :]
par['zH'] = forkCoM[1, :]
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
# masses
par['mR'] = d['rearWheelMass'][0]
par['mF'] = d['frontWheelMass'][0]
par['mB'] = d['frameMass'][0]
par['mH'] = d['forkMass'][0]
# calculate the wheel y inertias
par['g'] = 9.81*np.ones_like(par['rR'])
par['IRyy'] = com_inertia(par['mR'], par['g'], d['rWheelPendLength'][0], com[3, :])
par['IFyy'] = com_inertia(par['mF'], par['g'], d['fWheelPendLength'][0], com[2, :])
# calculate the wheel x/z inertias
par['IRxx'] = tor_inertia(k, tor[6, :])
par['IFxx'] = tor_inertia(k, tor[9, :])
# calculate the y inertias for the frame and fork
com[1, 0] = com[1, 1]
com[0, 7] = com[0, 6]
framePendLength = np.sqrt(frameCoM[0, :]**2 + (frameCoM[1, :] + par['rR'])**2)
par['IByy'] = com_inertia(par['mB'], par['g'], framePendLength, com[0, :])
forkPendLength = np.sqrt((forkCoM[0, :] - par['w'])**2 + (forkCoM[1, ] + par['rF'])**2)
par['IHyy'] = com_inertia(par['mH'], par['g'], forkPendLength, com[1, :])
# calculate the fork in-plane moments of inertia
Ipend = tor_inertia(k, tor)
par['IHxx'] = []
par['IHxz'] = []
par['IHzz'] = []
for i, row in enumerate(Ipend[3:6, :].T):
    Imat = inertia_components(row, betaFork[:, i])
    par['IHxx'].append(Imat[0, 0])
    par['IHxz'].append(Imat[0, 1])
    par['IHzz'].append(Imat[1, 1])
par['IHxx'] = np.array(par['IHxx'])
par['IHxz'] = np.array(par['IHxz'])
par['IHzz'] = np.array(par['IHzz'])
# calculate the frame in-plane moments of inertia
par['IBxx'] = []
par['IBxz'] = []
par['IBzz'] = []
for i, row in enumerate(Ipend[:3, :].T):
    Imat = inertia_components(row, betaFrame[:, i])
    par['IBxx'].append(Imat[0, 0])
    par['IBxz'].append(Imat[0, 1])
    par['IBzz'].append(Imat[1, 1])
par['IBxx'] = np.array(par['IBxx'])
par['IBxz'] = np.array(par['IBxz'])
par['IBzz'] = np.array(par['IBzz'])
par['v'] = np.ones_like(par['rR'])
# write the parameter files
for i, name in enumerate(bikeNames):
    file = open(''.join(name.split()) + 'Par.txt', 'w')
    for k, v in par.items():
        print v
        print v[i]
        line = k + ',' + str(v[i]) + '\n'
        file.write(line)
    file.close()

