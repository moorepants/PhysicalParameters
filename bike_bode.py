colors = ['k', 'r', 'b', 'g', 'y', 'm', 'c', 'orange']
vd = np.zeros(nBk)
vw = np.zeros(nBk)
vc = np.zeros(nBk)
eigFig = plt.figure(num=3)
Tdel2phi = plt.figure(num=4)
Tdel2del = plt.figure(num=5)
Tphi2phi = plt.figure(num=6)
Tphi2del = plt.figure(num=7)
for i, name in enumerate(bikeNames):
    dir = 'bikeRiderParameters/'
    fname = ''.join(name.split()) + 'RiderPar.txt'
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

