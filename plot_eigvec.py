from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pickle

from benchmark_bike_tools import bike_eig, sort_modes

f = open('data/bikeRiderCanonical/BianchiPistaRiderCan.p')
can = pickle.load(f)
f.close()

vel = np.linspace(0, 20, num=500)
evals, evecs = bike_eig(can['M'], can['C1'], can['K0'], can['K2'], vel, 9.81)
wea, cap, cas = sort_modes(evals, evecs)

colors = ['blue', 'red', 'green', 'orange']

fig = plt.figure()
ax = Axes3D(fig)
for i, row in enumerate(wea['evecs'][:, :, 1]):
    for k, component in enumerate(row[:2]):
        point1 = np.array([0., 0., vel[i]])
        point2 = np.array([np.abs(np.real(component)),
            np.abs(np.imag(component)), vel[i]])
        ax.plot(np.hstack((point1[0], point2[0])), np.hstack((point1[1],
            point2[1])), zs=vel[i], color=colors[k])
            #ax.plot(np.abs(np.real(wea['evecs'][:, :, 1])),
                #np.abs(np.imag(wea['evecs'][:, :, 1])), zs=vel)
plt.show()
