import pickle
import numpy as np
import matplotlib.pyplot as plt
from uncertainties.unumpy import nominal_values as nom
from uncertainties.unumpy import std_devs

from benchmark_bike_tools import *

# choose the bike alone, bike with rigid legs or bike with rider
# bike alone
direct = 'data/bikeCanonical/'
fend = 'Can.p'
# bike with rider's legs
#direct = 'data/bikeLegsCanonical/'
#fend = 'LegsCan.p'
# bike with rider
#direct = 'data/bikeRiderCanonical/'
#fend = 'RiderCan.p'


# load in the base data file
f = open('data/data.p', 'r')
data = pickle.load(f)
f.close()

# add Jodi's thesis bike for comparison (only for bikeCanonical)
#data['bikes'].append('Jodi Bike')

nBk = len(data['bikes']) # number of bikes

# one color for each bike
colors = ['k', 'r', 'b', 'g', 'y', 'm', 'c', 'orange', 'red']

vd = np.zeros(nBk)
vw = np.zeros(nBk)
vc = np.zeros(nBk)

# figure properties
figwidth = 3.25 # in inches
goldenMean = (np.sqrt(5)-1.0)/2.0
figsize = [figwidth, figwidth*goldenMean]
params = {#'backend': 'ps',
    #'axes.labelsize': 10,
    #'text.fontsize': 10,
    'legend.fontsize': 10,
    #'xtick.labelsize': 8,
    #'ytick.labelsize': 8,
    #'text.usetex': True,
    #'figure.figsize': figsize
    }
plt.rcParams.update(params)
eigFig = plt.figure(num=nBk + 1)#, figsize=figsize)
#plt.clf()
#plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])

# caluculate eigenvalues/vectors over this range
vel = np.linspace(0, 10, num=1000)

# color scale for root loci plot


for i, name in enumerate(data['bikes']):
    fname = ''.join(name.split()) + fend
    f = open(direct + fname)
    can = pickle.load(f)
    f.close()
    for k, v in can.items():
        can[k] = nom(v)
    evals, evecs = bike_eig(can['M'], can['C1'], can['K0'], can['K2'], vel, 9.81)
    wea, cap, cas = sort_modes(evals, evecs)
    vd[i], vw[i], vc[i] = critical_speeds(vel, wea['evals'], cap['evals'])
    # plot individual plot
    plt.figure(i)
    plt.plot(vel, np.abs(np.imag(wea['evals'])), color='blue', label='Imaginary Weave', linestyle='--')
    plt.plot(vel, np.abs(np.imag(cap['evals'])), color='red', label='Imaginary Capsize', linestyle='--')
    plt.plot(vel, np.zeros_like(vel), 'k-', label='_nolegend_', linewidth=2)
    # plot the real parts of the eigenvalues
    plt.plot(vel, np.real(wea['evals']), color='blue', label='Real Weave')
    plt.plot(vel, np.real(cap['evals']), color='red', label='Real Capsize')
    plt.plot(vel, np.real(cas['evals']), color='green', label='Real Caster')
    plt.ylim((-10, 10))
    plt.xlim((0, 10))
    plt.title('{name}\nEigenvalues vs Speed'.format(name=name))
    plt.xlabel('Speed [m/s]')
    plt.ylabel('Real and Imaginary Parts of the Eigenvalue [1/s]')
    plt.savefig('plots/' + ''.join(name.split()) + 'EigPlot.png')
    # plot root loci
    plt.figure(nBk + i)
    for j in range(len(evals[0, :])):
        plt.scatter(evals[:,j].real, evals[:,j].imag, s=2, c=vel,
                cmap=plt.cm.gist_rainbow, edgecolors='none')
    plt.colorbar()
    plt.grid()
    plt.axis('equal')
    plt.title('{name}\nEigenvalues vs Speed'.format(name=name))
    plt.savefig('plots/' + ''.join(name.split()) + 'RootLoci.png')
    # plot all bikes on the same plot
    plt.figure(2*nBk + 1)
    plt.plot(vel, np.abs(np.imag(wea['evals'])), color=colors[i], label='_nolegend_', linestyle='--')
    plt.plot(vel, np.abs(np.imag(cap['evals'])), color=colors[i], label='_nolegend_', linestyle='--')
    plt.plot(vel, np.zeros_like(vel), 'k-', label='_nolegend_', linewidth=3)
    plt.plot(vel, np.real(wea['evals']), color=colors[i], label='_nolegend_')
    plt.plot(vel, np.real(cap['evals']), color=colors[i], label=data['bikes'][i])
    plt.plot(vel, np.real(cas['evals']), color=colors[i], label='_nolegend_')
# plot the bike names on the eigenvalue plot
plt.legend()
plt.ylim((-10, 10))
plt.xlim((0, 10))
plt.title('Eigenvalues vs Speed')
plt.xlabel('Speed [m/s]')
plt.ylabel('Real and Imaginary Parts of the Eigenvalue [1/s]')
try:
    plt.savefig('plots/bike_eig.png')
except:
    pass
# make a plot comparing the critical speeds of each bike
#critFig = plt.figure(num=2)
#plt.clf()
#bike = np.arange(len(vd))
#plt.plot(vd, bike, '|', markersize=50)
#plt.plot(vc, bike, '|', markersize=50, linewidth=6)
#plt.plot(vw, bike, '|', markersize=50, linewidth=6)
#plt.plot(vc - vw, bike)
#plt.legend([r'$v_d$', r'$v_c$', r'$v_w$', 'stable speed range'])
#plt.yticks(np.arange(8), tuple(data['bikes']))
plt.show()
