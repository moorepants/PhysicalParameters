import pickle as p

from BenchmarkBikeTools import *

# load in the base data file
f = open('data/data.p', 'r')
data = p.load(f)
f.close()

# write the parameter files
for i, name in enumerate(data['bikes']):
    dir = 'bikeParameters/'
    fname = ''.join(name.split())
    M, C1, K0, K2, param = bmp2cm(dir + fname + 'Par.txt')
    A, B = abMatrix(M, C1, K0, K2, param['v'], param['g'])
    dir = 'bikeCanonical/'
    file = open(dir + fname + 'Can.txt', 'w')
    for mat in ['M','C1', 'K0', 'K2', 'A', 'B']:
        if mat == 'A' or mat == 'B':
            file.write(mat + ' (v = ' + str(param['v']) + ')\n')
        else:
            file.write(mat + '\n')
        file.write(str(eval(mat)) + '\n')
    file.close()
    file = open(dir + fname + 'Can.p', 'w')
    p.dump({'M':M, 'C1':C1, 'K0':K0, 'K2':K2, 'A':A, 'B':B, 'v':param['v']},
            file)
    file.close()
par_n = {}
for k, v in par.items():
    if type(v[i]) == type(par['rF'][0]) or type(v[i]) == type(par['mF'][0]):
        par_n[k] = u.nominal_values(v)
    else:
        par_n[k] = par[k]
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
Tdel2phi = plt.figure(num=4)
Tdel2del = plt.figure(num=5)
Tphi2phi = plt.figure(num=6)
Tphi2del = plt.figure(num=7)
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
    # y is [phidot, deldot, phi, del]
    C = np.eye(A.shape[0])
    freq = np.logspace(0, 2, 5000)
    plt.figure(4)
    bode(ABCD=(A, B[:, 1], C[2], 0.), w=freq, fig=Tdel2phi)
    plt.figure(5)
    bode(ABCD=(A, B[:, 1], C[3], 0.), w=freq, fig=Tdel2del)
    plt.figure(6)
    bode(ABCD=(A, B[:, 0], C[2], 0.), w=freq, fig=Tphi2phi)
    plt.figure(7)
    bode(ABCD=(A, B[:, 0], C[3], 0.), w=freq, fig=Tphi2del)
#for i, line in enumerate(Tdel2phi.ax1.lines):
#    plt.setp(line, color=colors[i])
#    plt.setp(Tdel2phi.ax2.lines[i], color=colors[i])
# plot the bike names on the eigenvalue plot
plt.figure(3)
plt.legend()
# make a plot comparing the critical speeds of each bike
critFig = plt.figure(num=8)
bike = np.arange(len(vd))
plt.plot(vd, bike, '|', markersize=50)
plt.plot(vc, bike, '|', markersize=50, linewidth=6)
plt.plot(vw, bike, '|', markersize=50, linewidth=6)
plt.plot(vc - vw, bike)
plt.legend([r'$v_d$', r'$v_c$', r'$v_w$', 'stable speed range'])
plt.yticks(np.arange(8), tuple(bikeNames))
plt.show()
