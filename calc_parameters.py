import os
import re
import pickle as p
import scipy.io.matlab.mio as mio
import numpy as np
import matplotlib.pyplot as plt
import uncertainties as u
import uncertainties.umath as umath
import uncertainties.unumpy as unumpy
from math import pi

from benchmark_bike_tools import *

# load the main data file into a dictionary
d = {}
mio.loadmat('data/data.mat', mdict=d)

# the number of different bikes
nBk = len(d['bikes'])

# make a list of the bikes' names
bikeNames = []
for bike in d['bikes']:
    # get rid of the weird matlab unicoding
    bikeNames.append(bike[0][0].encode('ascii'))

# clean up the matlab imports
d['bikes'] = bikeNames
del(d['__globals__'], d['__header__'], d['__version__'])
for k, v in d.items():
    if np.shape(v)[0] == 1:
        d[k] = v[0]

# pickle the data dictionary
f = open('data/data.p', 'w')
p.dump(d, f)
f.close()

# make a dictionary for the measurement standard deviations
dU = {}
f = open('data/MeasUncert.txt', 'r')
for line in f:
    l = line.split(',')
    dU[l[0]] = eval(l[1])
f.close()

# add the uncertainties to the data
ddU = {}
for k, v in dU.items():
    for pair in zip(d[k].flatten(), np.ones_like(d[k].flatten())*v):
        if k in ddU.keys():
            ddU[k].append(u.ufloat((float(pair[0]), pair[1])))
        else:
            ddU[k] = []
            ddU[k].append(u.ufloat((float(pair[0]), pair[1])))
    ddU[k] = np.array(ddU[k])
    if ddU[k].shape[0] > 8:
        ddU[k] = ddU[k].reshape((ddU[k].shape[0]/8, -1))

# pickle the data with the uncertainties
f = open('data/dataWithUncert.p', 'w')
p.dump(ddU, f)
f.close()

# calculate all the benchmark parameters
par = {}

# calculate the wheel radii
par['rR'] = ddU['rearWheelDist']/2./pi/ddU['rearWheelRot']
par['rF'] = ddU['frontWheelDist']/2./pi/ddU['frontWheelRot']

# steer axis tilt in radians
par['lambda'] = pi/180.*(90. - ddU['headTubeAngle'])

# calculate the front wheel trail
forkOffset = ddU['forkOffset']
par['c'] = (par['rF']*unumpy.sin(par['lambda'])
              - forkOffset)/unumpy.cos(par['lambda'])

# calculate the frame rotation angle
alphaFrame = ddU['frameAngle']
betaFrame = par['lambda'] - alphaFrame*pi/180

# calculate the slope of the CoM line
frameM = -unumpy.tan(betaFrame)

# calculate the z-intercept of the CoM line
frameMassDist = ddU['frameMassDist']
cb = unumpy.cos(betaFrame)
frameB = frameMassDist/cb - par['rR']

# calculate the fork rotation angle
betaFork = par['lambda'] - ddU['forkAngle']*np.pi/180.

# calculate the slope of the fork CoM line
forkM = -unumpy.tan(betaFork)

# calculate the z-intercept of the CoM line
par['w'] = ddU['wheelbase']
forkMassDist = ddU['forkMassDist']
cb = unumpy.cos(betaFork)
tb = unumpy.tan(betaFork)
forkB = - par['rF'] + forkMassDist/cb + par['w']*tb

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
        a = unumpy.matrix(np.vstack([-frameM[row, i], np.ones((2))]).T)
        b = frameB[row, i]
        lineX[j] = np.dot(a.I, b)
    frameCoM[:, i] = np.mean(lineX, axis=0)
    # calculate the fork center of mass position
    # reinitialize the matrix to store the line intersections
    lineX = np.zeros((3, 2), dtype='object')
    # for each line intersection...
    for j, row in enumerate(comb):
        a = unumpy.matrix(np.vstack([-forkM[row, i], np.ones((2))]).T)
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

# load the average period data
f = open('data/avgPer.p', 'r')
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
mRod = u.ufloat((5.56, 0.02)) # mass of the calibration rod [kg]
lRod = u.ufloat((1.05, 0.001)) # length of the calibration rod [m]
rRod = u.ufloat((0.015, 0.0001)) # radius of the calibration rod [m]
iRod = tube_inertia(lRod, mRod, rRod, 0.)[1]
k = tor_stiffness(iRod, tRod)

# masses
par['mR'] = ddU['rearWheelMass']
par['mF'] = ddU['frontWheelMass']
par['mB'] = ddU['frameMass']
par['mH'] = ddU['forkMass']

# calculate the wheel y inertias
par['g'] = 9.81*np.ones_like(d['forkMass'])
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

# round the numbers according to the calculated uncertainty
par_test = {}
for k, v in par.items():
    par_test[k] = np.zeros_like(v)
    for i, value in enumerate(v):
        try:
            uncert = value.std_dev()
            nom = value.nominal_value
            s = str(uncert)
            for j, number in enumerate(s):
                if number == '0' or number == '.':
                    pass
                else:
                    digit = j
                    break
            newUncert = round(uncert, digit-1)
            newNom = round(nom, len(str(newUncert)) - 2)
            newValue = u.ufloat((newNom, newUncert))
            par_test[k][i] = newValue
        except:
            pass
del par
par = par_test

# make a dictionary with only the nominal values
par_n = {}
for k, v in par.items():
    if type(v[0]) == type(par['rF'][0]) or type(v[0]) == type(par['mF'][0]):
        par_n[k] = unumpy.nominal_values(v)
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
plt.show()

# write the parameter files
for i, name in enumerate(bikeNames):
    dir = 'data/bikeParameters/'
    fname = ''.join(name.split())
    file = open(dir + fname + 'Par.txt', 'w')
    for k, v in par.items():
        if type(v[i]) == type(par['rF'][0]) or type(v[i]) == type(par['mF'][0]):
            line = k + ',' + str(v[i].nominal_value) + ',' + str(v[i].std_dev()) + '\n'
        else:
            line = k + ',' + str(v[i]) + ',' + '0.0' + '\n'
        file.write(line)
    file.close()

# pickle the parameters too
file = open('data/par.p', 'w')
p.dump(par, file)
file.close()
