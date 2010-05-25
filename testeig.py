import numpy as np
from matplotlib.pyplot import figure, show
from BenchmarkBikeTools import *
from control_tools import *
d = {'M': np.array([[102.7, 1.7], [1.7, 0.4]]), 'C1': np.array([[0., 26.7],
    [-0.5, 1.1]]), 'K0': np.array([[-89.2, -1.7], [-1.7, -0.7]]), 'K2':
    np.array([[0., 74.1], [0., 1.6]])}
v = np.linspace(0, 10, num=1000)
evals, evecs = bike_eig(d['M'], d['C1'], d['K0'], d['K2'], v, 9.81)
weave, capsize, caster = sort_modes(evals, evecs)
vd, vw, vc = critical_speeds(v, weave['evals'], capsize['evals'])
A, B = abMatrix(d['M'], d['C1'], d['K0'], d['K2'], 10., 9.81)
# the abMatrix function outputs for this state space:
# x = [phi_dot, delta_dot, phi, delta]
# u = [Tphi, Tdelta]
Aul = A[2:, 2:]
Aur = A[2:, :2]
All = A[:2, 2:]
Alr = A[:2, :2]
AFlip = np.vstack((np.hstack((Aul, Aur)), np.hstack((All, Alr))))
BFlip = np.vstack([B[2:, :], B[:2, :]])
# the bode function works with the following state space:
# x = [phi, delta, phi_dot, delta_dot]
# u = [Tphi, Tdelta]
C = np.array([1., 0., 0., 0.])
D = 0.
w = np.logspace(0, 2, 5000)
f = figure(num=1)
mag, phase, fig = bode(ABCD=(AFlip, BFlip[:, 0], C, D), w=w, fig=f)
show()
