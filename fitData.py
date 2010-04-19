import scipy.io.matlab.mio as mio
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import os
import pickle as p

dirs, subdirs, filenames = list(os.walk('data/pendDat'))[0]
file = open('period.txt', 'w')
filenames.sort()
period = {}
#for name in ['StratosFrameCompoundFirst2.mat']:
for name in filenames:
    pendDat = {}
    mio.loadmat('data/pendDat/' + name, mdict=pendDat)
    y = pendDat['data'].ravel()
    x = np.linspace(0, 30, num=len(y))
    # plot the original data
    plt.plot(x, y, '.')
    # decaying oscillating exponential function
    fitfunc = lambda p, t: p[0] + np.exp(-p[3]*p[4]*t)*(p[1]*np.cos(p[4]*np.sqrt(1-p[3]**2)*t) + p[2]*np.sin(p[4]*np.sqrt(1-p[3]**2)*t))
    # initial guesses
    p0 = np.array([1.35, -.75, -.5, 0.01, 3.93])
    # create the error function
    errfunc = lambda p, t, y: fitfunc(p, t) - y
    # minimize the error function
    p1, success = op.leastsq(errfunc, p0[:], args=(x, y))
    # plot the fitted curve
    lscurve = fitfunc(p1, x)
    plt.plot(x, lscurve, 'k')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    #plt.show()
    plt.savefig('data/pendDat/graphs/' + name[:-4] + '.png')
    plt.close()
    SSR = np.sum((lscurve - np.mean(y))**2)
    SST = np.sum((y - np.mean(y))**2)
    rsq = SSR/SST
    if rsq <= 0.99:
        rsq = str(rsq) + '*'
    else:
        pass
    wo = p1[4]
    f = wo/2./np.pi
    T = 1./f
    note = pendDat['notes']
    line = name + ',' + str(T) + ',' + str(rsq) + ',' + str(note) + '\n'
    file.write(line)
    print line
    if name[:-5] in period.keys():
        period[name[:-5]].append(T)
    else:
        period[name[:-5]] = [T]
file.close()
file = open('period.p', 'w')
p.dump(period, file)
file.close()
