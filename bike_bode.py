import numpy as np
import pickle
import matplotlib.pyplot as plt

from BenchmarkBikeTools import *

# load in the base data file
f = open('data/data.p', 'r')
data = pickle.load(f)
f.close()

nBk = len(data['bikes'])

colors = ['k', 'r', 'b', 'g', 'y', 'm', 'c', 'orange']

Tdel2phi = plt.figure(num=1)
Tdel2del = plt.figure(num=2)
Tphi2phi = plt.figure(num=3)
Tphi2del = plt.figure(num=4)

dir = 'data/bikeRiderCanonical/'
for i, name in enumerate(data['bikes']):
    fname = ''.join(name.split()) + 'RiderCan.p'
    file = open(dir + fname)
    can = pickle.load(file)
    file.close()
    # make some bode plots
    A, B = abMatrix(can['M'], can['C1'], can['K0'], can['K2'], 1., 9.81)
    # y is [phidot, deldot, phi, del]
    C = np.eye(A.shape[0])
    freq = np.logspace(0, 2, 5000)
    plt.figure(1)
    bode(ABCD=(A, B[:, 1], C[2], 0.), w=freq, fig=Tdel2phi)
    plt.figure(2)
    bode(ABCD=(A, B[:, 1], C[3], 0.), w=freq, fig=Tdel2del)
    plt.figure(3)
    bode(ABCD=(A, B[:, 0], C[2], 0.), w=freq, fig=Tphi2phi)
    plt.figure(4)
    bode(ABCD=(A, B[:, 0], C[3], 0.), w=freq, fig=Tphi2del)
#for i, line in enumerate(Tdel2phi.ax1.lines):
#    plt.setp(line, color=colors[i])
#    plt.setp(Tdel2phi.ax2.lines[i], color=colors[i])
# plot the bike names on the eigenvalue plot
plt.show()

