import numpy as np
import pickle as p
from uncertainties import unumpy
import os
from scipy.io.matlab.mio import savemat

from benchmark_bike_tools import *

# load in the base data file
f = open('data/data.p', 'r')
data = p.load(f)
f.close()

# load in the parameter file
f = open('data/par.p', 'r')
par = p.load(f)
f.close()

nBk = len(data['bikes'])

# remove the uncertainties
par_n = {}
for k, v in par.items():
    if type(v[0]) == type(par['rF'][0]) or type(v[0]) == type(par['mF'][0]):
        par_n[k] = unumpy.nominal_values(v)
    else:
        par_n[k] = par[k]

# Jason's leg parameters (sitting on the Batavus Browser)
IBJ = np.array([[ 1.371635407777139, 0. ,               -0.355678341037248],
                [ 0.,                1.248341633510700,  0.               ],
                [-0.355678341037248, 0.,                 0.721984553624527]])
mBJ = 23.183999999999997
xBJ = 0.41651301147631
zBJ = -0.742728766690162

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

file = open('data/parWithLegs.p', 'w')
p.dump(par_n, file)
file.close()

savemat('data/parWithLegs.mat', par_n)

speeds = [4.0, 5.8, 12]
# write the parameter files
for i, name in enumerate(data['bikes']):
    direct = 'data/bikeLegParameters/'
    fname = ''.join(name.split())
    try:
        file = open(direct + fname  + 'LegsPar.txt', 'w')
    except:
        os.system('mkdir data/bikeLegParameters/')
        file = open(direct + fname  + 'LegsPar.txt', 'w')
    for k, v in par_n.items():
        line = k + ',' + str(v[i]) + '\n'
        file.write(line)
    file.close()
    M, C1, K0, K2, param = bmp2cm(direct + fname + 'LegsPar.txt')
    direct = 'data/bikeLegsCanonical/'
    try:
        file = open(direct + fname + 'LegsCan.txt', 'w')
    except:
        os.system('mkdir data/bikeLegsCanonical/')
        file = open(direct + fname + 'LegsCan.txt', 'w')
    for mat in ['M','C1', 'K0', 'K2']:
        file.write(mat + '\n')
        file.write(str(eval(mat)) + '\n')
    file.write("The states are [roll rate, steer rate, roll angle, steer angle]\n")
    for v in speeds:
        A, B = abMatrix(M, C1, K0, K2, v, param['g'])
        for mat in ['A', 'B']:
            file.write(mat + ' (v = ' + str(v) + ')\n')
            file.write(str(eval(mat)) + '\n')
    file.close()
    file = open(direct + fname + 'LegsCan.p', 'w')
    p.dump({'M':M, 'C1':C1, 'K0':K0, 'K2':K2},
            file)
    file.close()
