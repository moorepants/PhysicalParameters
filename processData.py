import scipy.io.matlab.mio as mio
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pickle as p
import uncertainties as u
import uncertainties.umath as umath
from BenchmarkBikeTools import *
from math import pi

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

# clean up the matlab imports
d['bikes'] = bikeNames
del(d['__globals__'], d['__header__'], d['__version__'])
for k, v in d.items():
    if np.shape(v)[0] == 1:
        d[k] = v[0]

# make a dictionary for the measurement standard deviations
dU = {}
f = open('MeasUncert.txt', 'r')
for line in f:
    l = line.split(',')
    dU[l[0]] = eval(l[1])
f.close()

# add the uncertainties to the data
ddU = {}
for k, v in dU.items():
    for pair in zip(d[k].flatten(), np.ones_like(d[k].flatten())*v):
        if k in ddU.keys():
            ddU[k].append(u.num_with_uncert((float(pair[0]), pair[1])))
        else:
            ddU[k] = []
            ddU[k].append(u.num_with_uncert((float(pair[0]), pair[1])))
    ddU[k] = np.array(ddU[k])
    if ddU[k].shape[0] > 8:
        ddU[k] = ddU[k].reshape((ddU[k].shape[0]/8, -1))

par = {}
# calculate the wheel radii
par['rR'] = ddU['rearWheelDist']/2./pi/ddU['rearWheelRot']
#print "Rear wheel radii [m] =\n", par['rR']
par['rF'] = ddU['frontWheelDist']/2./pi/ddU['frontWheelRot']
#print "Front wheel radii [m] =\n", par['rF']

# steer axis tilt in radians
par['lambda'] = pi/180.*(90. - ddU['headTubeAngle'])
#print "Steer axis tilt [deg] =\n", par['lambda']*180./np.pi

# calculate the front wheel trail
forkOffset = ddU['forkOffset']
par['c'] = np.zeros_like(forkOffset)
for i, v in enumerate(par['rF']):
    par['c'][i] = (par['rF'][i]*umath.sin(par['lambda'][i]) - forkOffset[i])/umath.cos(par['lambda'][i])
#print "Trail [m] =\n", par['c']

# calculate the frame rotation angle
alphaFrame = ddU['frameAngle']
#print "alphaFrame =\n", alphaFrame
betaFrame = par['lambda'] - alphaFrame*pi/180
#print "Frame rotation angle, beta [deg] =\n", betaFrame/np.pi*180.

# calculate the slope of the CoM line
frameM = np.zeros_like(betaFrame)
for i, row in enumerate(betaFrame):
    for j, v in enumerate(row):
        frameM[i, j] = -umath.tan(v)
#print "Frame CoM line slope =\n", frameM

# calculate the z-intercept of the CoM line
frameMassDist = ddU['frameMassDist']
#print "Frame CoM distance =\n", d['frameMassDist']
frameB = np.zeros_like(frameMassDist)
for i, row in enumerate(frameMassDist):
    for j, col in enumerate(row):
        frameB[i, j] = frameMassDist[i, j]/umath.cos(betaFrame[i, j]) - par['rR'][j]
#print "Frame CoM line intercept =\n", frameB

# calculate the fork rotation angle
betaFork = par['lambda'] - ddU['forkAngle']*np.pi/180.
#print "Fork rotation angle [deg] =\n", betaFork*180./np.pi

# calculate the slope of the fork CoM line
forkM = np.zeros_like(betaFork)
for i, row in enumerate(betaFork):
    for j, v in enumerate(row):
        forkM[i, j] = -umath.tan(v)
#print "Fork CoM line slope =\n", forkM

# calculate the z-intercept of the CoM line
par['w'] = ddU['wheelbase']
forkMassDist = ddU['forkMassDist']
forkB = np.zeros_like(forkMassDist)
for i, row in enumerate(forkMassDist):
    for j, col in enumerate(row):
        forkB[i, j] = - par['rF'][j] + forkMassDist[i, j]/umath.cos(betaFork[i, j]) + par['w'][j]*umath.tan(betaFork[i, j])
#print "Fork CoM line intercept =\n", forkB

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
# the yellow bikes have the same frame
tor[:3, 7] = tor[:3, 6]
com[0, 7] = com[0, 6]
# the browser's have the same fork
tor[3:6, 0] = tor[3:6, 1]
com[1, 0] = com[1, 1]
# the browsers have the same front and rear wheels
tor[6, 1] = tor[6, 0]
tor[9, 1] = tor[9, 0]
com[2, 1] = com[2, 0]
com[3, 1] = com[3, 0]
# the yellow bikes have the same front and rear wheels
tor[6, 7] = tor[6, 6]
tor[9, 7] = tor[9, 6]
com[2, 7] = com[2, 6]
com[3, 7] = com[3, 6]
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
    dir = 'bikeParameters/'
    fname = ''.join(name.split()) + 'Par.txt'
    file = open(dir + fname, 'w')
    for k, v in par.items():
        line = k + ',' + str(v[i]) + '\r\n'
        file.write(line)
    file.close()
    M, C1, K0, K2, p = bmp2cm(dir + fname)
    A, B = abMatrix(M, C1, K0, K2, p)
    dir = 'bikeCanonical/'
    file = open(dir + fname[:-7] + 'Can.txt', 'w')
    for mat in ['M','C1', 'K0', 'K2', 'A']:
        file.write(mat + '\r\n')
        file.write(str(eval(mat)) + '\r\n')
    file.close()
# Jason's parameters (sitting on the browser)
IBJ = np.array([[7.9985, 0 , -1.9272], [0, 8.0689, 0], [ -1.9272, 0, 2.3624]])
mBJ = 72.
xBJ = 0.2909
zBJ = -1.1091
# compute the total mass
mB = par['mB'] + mBJ
# compute the new CoM
xB = (mBJ*xBJ + par['mB']*par['xB'])/mB
zB = (mBJ*zBJ + par['mB']*par['zB'])/mB
# compute the new moment of inertia
dJ = np.vstack((xB, np.zeros(nBk), zB)) - np.vstack((xBJ, 0., zBJ))
dB = np.vstack((xB, np.zeros(nBk), zB)) - np.vstack((par['xB'], np.zeros(nBk), par['zB']))
IB = np.zeros((3, 3))
for i in range(nBk):
    IB[0] = np.array([par['IBxx'][i], 0., par['IBxz'][i]])
    IB[1] = np.array([0., par['IByy'][i], 0.])
    IB[2] = np.array([par['IBxz'][i], 0., par['IBzz'][i]])
    I = parallel_axis(IBJ, mBJ, dJ[:, i]) + parallel_axis(IB, par['mB'][i], dB[:, i])
    par['IBxx'][i] = I[0, 0]
    par['IBxz'][i] = I[0, 2]
    par['IByy'][i] = I[1, 1]
    par['IBzz'][i] = I[2, 2]
    par['mB'][i] = mB[i]
    par['xB'][i] = xB[i]
    par['zB'][i] = zB[i]
# write the parameter files
for i, name in enumerate(bikeNames):
    dir = 'bikeRiderParameters/'
    fname = ''.join(name.split()) + 'RiderPar.txt'
    file = open(dir + fname, 'w')
    for k, v in par.items():
        line = k + ',' + str(v[i]) + '\r\n'
        file.write(line)
    file.close()
    M, C1, K0, K2, p = bmp2cm(dir + fname)
    A, B = abMatrix(M, C1, K0, K2, p)
    dir = 'bikeRiderCanonical/'
    file = open(dir + fname[:-7] + 'Can.txt', 'w')
    for mat in ['M','C1', 'K0', 'K2', 'A']:
        file.write(mat + '\r\n')
        file.write(str(eval(mat)) + '\r\n')
    file.close()
