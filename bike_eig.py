import pickle
import numpy as np
import matplotlib.pyplot as plt

from benchmark_bike_tools import *

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
    plt.figure(1)
    plt.plot(vel, np.abs(np.imag(wea['evals'])), color=colors[i], label='_nolegend_', linestyle='--')
    plt.plot(vel, np.zeros_like(vel), 'k-', label='_nolegend_')
    plt.plot(vel, np.real(wea['evals']), color=colors[i], label='_nolegend_')
    plt.plot(vel, np.real(cap['evals']), color=colors[i], label=data['bikes'][i])
    plt.plot(vel, np.real(cas['evals']), color=colors[i], label='_nolegend_')
    vd[i], vw[i], vc[i] = critical_speeds(vel, wea['evals'], cap['evals'])
# plot the bike names on the eigenvalue plot
plt.legend()
plt.ylim((-10, 10))
#plt.xlim((0, 10))
plt.title('Eigenvalues vs Speed')
plt.xlabel('Speed [m/s]')
plt.ylabel('Real and Imaginary Parts of the Eigenvalue [1/s]')
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
