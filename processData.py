import os
import re
import pickle as p
import scipy.io.matlab.mio as mio
import numpy as np
import matplotlib.pyplot as plt
import uncertainties as u
import uncertainties.umath as umath
from math import pi

from BenchmarkBikeTools import *
from control_tools import *

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
    par['c'][i] = (par['rF'][i]*umath.sin(par['lambda'][i])
                  - forkOffset[i])/umath.cos(par['lambda'][i])
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
        cb = umath.cos(betaFrame[i, j])
        frameB[i, j] = frameMassDist[i, j]/cb - par['rR'][j]
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
        cb = umath.cos(betaFork[i, j])
        tb = umath.tan(betaFork[i, j])
        forkB[i, j] = - par['rF'][j] + forkMassDist[i, j]/cb + par['w'][j]*tb
#print "Fork CoM line intercept =\n", forkB

# plot the CoM lines
comFig = plt.figure(num=1)
# intialize the matrices for the center of mass locations
frameCoM = np.zeros((2, np.shape(frameM)[1]), dtype='object')
forkCoM = np.zeros((2, np.shape(forkM)[1]), dtype='object')
# for each of the bikes...
for i in range(np.shape(frameM)[1]):
    comb = np.array([[0, 1], [0, 2], [1, 2]])
    # calculate the frame center of mass position
    # initialize the matrix to store the line intersections
    lineX = np.zeros((3, 2), dtype='object')
    # for each line intersection...
    for j, row in enumerate(comb):
        a = np.matrix(np.vstack([-frameM[row, i], np.ones((2))]).T)
        b = frameB[row, i]
        lineX[j] = np.dot(a.I, b)
    frameCoM[:, i] = np.mean(lineX, axis=0)
    # calculate the fork center of mass position
    # reinitialize the matrix to store the line intersections
    lineX = np.zeros((3, 2), dtype='object')
    # for each line intersection...
    for j, row in enumerate(comb):
        a = np.matrix(np.vstack([-forkM[row, i], np.ones((2))]).T)
        b = forkB[row, i]
        lineX[j] = np.dot(a.I, b)
    forkCoM[:, i] = np.mean(lineX, axis=0)
    # make a subplot for this bike
    plt.subplot(2, 4, i + 1)
    # plot the rear wheel
    c = plt.Circle((0, par['rR'][i].nominal_value), radius=par['rR'][i].nominal_value)
    plt.gca().add_patch(c)
    # plot the front wheel
    c = plt.Circle((par['w'][i].nominal_value, par['rF'][i].nominal_value), radius=par['rF'][i].nominal_value)
    plt.gca().add_patch(c)
    # plot the lines (pendulum axes)
    x = np.linspace(-par['rR'][i].nominal_value, par['w'][i].nominal_value + par['rF'][i].nominal_value, 2)
    # for each line...
    for j in range(len(frameM)):
        framey = -frameM[j, i].nominal_value*x - frameB[j, i].nominal_value
        forky = -forkM[j, i].nominal_value*x - forkB[j, i].nominal_value
        plt.plot(x,framey, 'r')
        plt.plot(x,forky, 'g')
    # plot the ground line
    plt.plot(x, np.zeros_like(x), 'k')
    # plot the centers of mass
    plt.plot(frameCoM[0, i].nominal_value, -frameCoM[1, i].nominal_value, 'k+', markersize=12)
    plt.plot(forkCoM[0, i].nominal_value, -forkCoM[1, i].nominal_value, 'k+', markersize=12)
    plt.axis('equal')
    plt.ylim((0, 1))
    plt.title(bikeNames[i])
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
mRod = u.num_with_uncert((5.56, 0.02)) # mass of the calibration rod [kg]
lRod = u.num_with_uncert((1.05, 0.001)) # length of the calibration rod [m]
rRod = u.num_with_uncert((0.015, 0.0001)) # radius of the calibration rod [m]
iRod = tube_inertia(lRod, mRod, rRod, 0.)[1]
k = tor_stiffness(iRod, tRod)
# masses
par['mR'] = ddU['rearWheelMass']
par['mF'] = ddU['frontWheelMass']
par['mB'] = ddU['frameMass']
par['mH'] = ddU['forkMass']
# calculate the wheel y inertias
par['g'] = 9.81*np.ones_like(par['rR'])
par['IRyy'] = com_inertia(par['mR'], par['g'], ddU['rWheelPendLength'], com[3, :])
par['IFyy'] = com_inertia(par['mF'], par['g'], ddU['fWheelPendLength'], com[2, :])
# calculate the wheel x/z inertias
par['IRxx'] = tor_inertia(k, tor[6, :])
par['IFxx'] = tor_inertia(k, tor[9, :])
# calculate the y inertias for the frame and fork
framePendLength = (frameCoM[0, :]**2 + (frameCoM[1, :] + par['rR'])**2)**(0.5)
par['IByy'] = com_inertia(par['mB'], par['g'], framePendLength, com[0, :])
forkPendLength = ((forkCoM[0, :] - par['w'])**2 + (forkCoM[1, ] + par['rF'])**2)**(0.5)
par['IHyy'] = com_inertia(par['mH'], par['g'], forkPendLength, com[1, :])
# calculate the fork in-plane moments of inertia
Ipend = tor_inertia(k, tor)
par['IHxx'] = []
par['IHxz'] = []
par['IHzz'] = []
for i, row in enumerate(Ipend[3:6, :].T):
    Imat = inertia_components_uncert(row, betaFork[:, i])
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
    Imat = inertia_components_uncert(row, betaFrame[:, i])
    par['IBxx'].append(Imat[0, 0])
    par['IBxz'].append(Imat[0, 1])
    par['IBzz'].append(Imat[1, 1])
par['IBxx'] = np.array(par['IBxx'])
par['IBxz'] = np.array(par['IBxz'])
par['IBzz'] = np.array(par['IBzz'])
par['v'] = np.ones_like(d['forkMass'])
# write the parameter files
for i, name in enumerate(bikeNames):
    dir = 'bikeParameters/'
    fname = ''.join(name.split()) + 'Par.txt'
    file = open(dir + fname, 'w')
    for k, v in par.items():
        if type(v[i]) == type(par['rF'][0]) or type(v[i]) == type(par['mF'][0]):
            line = k + ',' + str(v[i].nominal_value) + ',' + str(v[i].std_dev()) + '\n'
        else:
            line = k + ',' + str(v[i]) + ',' + '0.0' + '\n'
        file.write(line)
    file.close()
    M, C1, K0, K2, param = bmp2cm(dir + fname)
    A, B = abMatrix(M, C1, K0, K2, param['v'], param['g'])
    dir = 'bikeCanonical/'
    file = open(dir + fname[:-7] + 'Can.txt', 'w')
    for mat in ['M','C1', 'K0', 'K2', 'A', 'B']:
        if mat == 'A' or mat == 'B':
            file.write(mat + ' (v = ' + str(par['v'][i]) + ')\n')
        else:
            file.write(mat + '\n')
        file.write(str(eval(mat)) + '\n')
    file.close()
par_n = {}
for k, v in par.items():
    if type(v[i]) == type(par['rF'][0]) or type(v[i]) == type(par['mF'][0]):
        par_n[k] = u.nominal_values(v)
    else:
        par_n[k] = par[k]
# plot all the parameters to look for crazy numbers
parFig = plt.figure(num=2)
i = 1
xt = ['B', 'BI', 'C', 'F', 'P', 'S', 'Y', 'YR']
for k, v in par_n.items():
    plt.subplot(3, 9, i)
    plt.plot(v, '-D', markersize=14)
    plt.title(k)
    plt.xticks(np.arange(8), tuple(xt))
    i += 1
# Jason's parameters (sitting on the browser)
IBJ = np.array([[7.9985, 0 , -1.9272], [0, 8.0689, 0], [ -1.9272, 0, 2.3624]])
mBJ = 72.
xBJ = 0.2909
zBJ = -1.1091
# compute the total mass
mB = par_n['mB'] + mBJ
# compute the new CoM
xB = (mBJ*xBJ + par_n['mB']*par_n['xB'])/mB
zB = (mBJ*zBJ + par_n['mB']*par_n['zB'])/mB
# compute the new moment of inertia
dJ = np.vstack((xB, np.zeros(nBk), zB)) - np.vstack((xBJ, 0., zBJ))
dB = np.vstack((xB, np.zeros(nBk), zB)) - np.vstack((par_n['xB'], np.zeros(nBk), par_n['zB']))
IB = np.zeros((3, 3))
for i in range(nBk):
    IB[0] = np.array([par_n['IBxx'][i], 0., par_n['IBxz'][i]])
    IB[1] = np.array([0., par_n['IByy'][i], 0.])
    IB[2] = np.array([par_n['IBxz'][i], 0., par_n['IBzz'][i]])
    I = parallel_axis(IBJ, mBJ, dJ[:, i]) + parallel_axis(IB, par_n['mB'][i], dB[:, i])
    par_n['IBxx'][i] = I[0, 0]
    par_n['IBxz'][i] = I[0, 2]
    par_n['IByy'][i] = I[1, 1]
    par_n['IBzz'][i] = I[2, 2]
    par_n['mB'][i] = mB[i]
    par_n['xB'][i] = xB[i]
    par_n['zB'][i] = zB[i]
colors = ['k', 'r', 'b', 'g', 'y', 'm', 'c', 'orange']
vd = np.zeros(nBk)
vw = np.zeros(nBk)
vc = np.zeros(nBk)
eigFig = plt.figure(num=3)
bodeFig = plt.figure(num=4)
# write the par_nameter files
for i, name in enumerate(bikeNames):
    dir = 'bikeRiderParameters/'
    fname = ''.join(name.split()) + 'RiderPar.txt'
    file = open(dir + fname, 'w')
    for k, v in par_n.items():
        line = k + ',' + str(v[i]) + '\n'
        file.write(line)
    file.close()
    M, C1, K0, K2, param = bmp2cm(dir + fname)
    A, B = abMatrix(M, C1, K0, K2, param['v'], param['g'])
    dir = 'bikeRiderCanonical/'
    file = open(dir + fname[:-7] + 'Can.txt', 'w')
    for mat in ['M','C1', 'K0', 'K2', 'A', 'B']:
        if mat == 'A' or mat == 'B':
            file.write(mat + ' (v = ' + str(par_n['v'][i]) + ')\n')
        else:
            file.write(mat + '\n')
        file.write(str(eval(mat)) + '\n')
    file.close()
    vel = np.linspace(0, 20, num=1000)
    evals, evecs = bike_eig(M, C1, K0, K2, vel, 9.81)
    wea, cap, cas = sort_modes(evals, evecs)
    vd[i], vw[i], vc[i] = critical_speeds(vel, wea['evals'], cap['evals'])
    plt.figure(3)
    for j, line in enumerate(evals.T):
        if j == 0:
            label = bikeNames[i]
        else:
            label = '_nolegend_'
        plt.plot(vel, np.real(line), '.', color=colors[i], label=label, figure=eigFig)
    plt.plot(vel, np.abs(np.imag(evals)), '.', markersize=2, color=colors[i], figure=eigFig)
    plt.ylim((-10, 10), figure=eigFig)
    # make some bode plots
    A, B = abMatrix(M, C1, K0, K2, 4., 9.81)
    C_phi = np.array([0., 0., 1., 0.])
    C_del = np.array([0., 0., 0., 1.])
    freq = np.logspace(0, 2, 5000)
    Aul = A[2:, 2:]
    Aur = A[2:, :2]
    All = A[:2, 2:]
    Alr = A[:2, :2]
    AFlip = np.vstack((np.hstack((Aul, Aur)), np.hstack((All, Alr))))
    BFlip = np.vstack([B[2:, :], B[:2, :]])
    plt.figure(4)
    bode(ABCD=(A, B[:, 0], C_phi, 0.), w=freq, fig=bodeFig)
    #bode(ABCD=(AFlip, BFlip[:, 0], C_phi, 0.), w=freq, fig=bodeFig)
for i, line in enumerate(bodeFig.ax1.lines):
    plt.setp(line, color=colors[i])
    plt.setp(bodeFig.ax2.lines[i], color=colors[i])
plt.figure(3)
plt.legend()
critFig = plt.figure(num=5)
bike = np.arange(len(vd))
plt.plot(vd, bike, '|', markersize=50)
plt.plot(vc, bike, '|', markersize=50, linewidth=6)
plt.plot(vw, bike, '|', markersize=50, linewidth=6)
plt.plot(vc - vw, bike)
plt.yticks(np.arange(8), tuple(bikeNames))
plt.show()
