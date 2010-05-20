import numpy as np
from BenchmarkBikeTools import *
d = {'M': np.array([[102.7, 1.7], [1.7, 0.4]]), 'C1': np.array([[0., 26.7],
    [-0.5, 1.1]]), 'K0': np.array([[-89.2, -1.7], [-1.7, -0.7]]), 'K2':
    np.array([[0., 74.1], [0., 1.6]])}
v = np.linspace(0, 10, num=1000)
evals, evecs = bike_eig(d['M'], d['C1'], d['K0'], d['K2'], v, 9.81)
newEvals = sort_modes(evals, evecs)
