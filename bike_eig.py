import pickle
import numpy as np
import matplotlib.pyplot as plt
from BenchmarkBikeTools import *


# load in the base data file
f = open('data/data.p', 'r')
data = pickle.load(f)
f.close()

nBk = len(data['bikes'])

colors = ['k', 'r', 'b', 'g', 'y', 'm', 'c', 'orange']
vd = np.zeros(nBk)
vw = np.zeros(nBk)
vc = np.zeros(nBk)
eigFig = plt.figure(num=1)
dir = 'data/bikeRiderCanonical/'
vel = np.linspace(0, 20, num=1000)
for i, name in enumerate(data['bikes']):
    fname = ''.join(name.split()) + 'RiderCan.p'
    file = open(dir + fname)
    can = pickle.load(file)
    file.close()
    evals, evecs = bike_eig(can['M'], can['C1'], can['K0'], can['K2'], vel, 9.81)
    wea, cap, cas = sort_modes(evals, evecs)
    vd[i], vw[i], vc[i] = critical_speeds(vel, wea['evals'], cap['evals'])
    plt.figure(1)
    for j, line in enumerate(evals.T):
        if j == 0:
            label = data['bikes'][i]
        else:
            label = '_nolegend_'
        plt.plot(vel, np.real(line), '.', color=colors[i], label=label, figure=eigFig)
    plt.plot(vel, np.abs(np.imag(evals)), '.', markersize=2, color=colors[i], figure=eigFig)
    plt.ylim((-10, 10), figure=eigFig)
# plot the bike names on the eigenvalue plot
plt.figure(1)
plt.legend()
# make a plot comparing the critical speeds of each bike
critFig = plt.figure(num=2)
bike = np.arange(len(vd))
plt.plot(vd, bike, '|', markersize=50)
plt.plot(vc, bike, '|', markersize=50, linewidth=6)
plt.plot(vw, bike, '|', markersize=50, linewidth=6)
plt.plot(vc - vw, bike)
plt.legend([r'$v_d$', r'$v_c$', r'$v_w$', 'stable speed range'])
plt.yticks(np.arange(8), tuple(data['bikes']))
plt.show()

