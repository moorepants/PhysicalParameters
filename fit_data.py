import scipy.io.matlab.mio as mio
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import os
import pickle as p
from scipy.optimize import approx_fprime
from uncertainties import ufloat
from benchmark_bike_tools import fit_goodness, jac_fitfunc, plot_osfit

dirs, subdirs, filenames = list(os.walk('data/pendDat'))[0]
file = open('data/period.txt', 'w')
filenames.sort()
period = {}
for name in ['YellowRevForkTorsionalFirst1.mat']:
#for name in ['YellowFwheelCompoundFirst1.mat']:
#for name in ['StratosFrameCompoundFirst2.mat']:
#for name in filenames:
    pendDat = {}
    mio.loadmat('data/pendDat/' + name, mdict=pendDat)
    y = pendDat['data'].ravel()
    time = float(pendDat['duration'][0])
    x = np.linspace(0, time, num=len(y))
    plt.figure(1)
    # plot the original data
    plt.plot(x, y, '.')
    # decaying oscillating exponential function
    fitfunc = lambda p, t: p[0] + np.exp(-p[3]*p[4]*t)*(p[1]*np.sin(p[4]*np.sqrt(1-p[3]**2)*t) + p[2]*np.cos(p[4]*np.sqrt(1-p[3]**2)*t))
    # initial guesses
    p0 = np.array([1.35, -.5, -.75, 0.01, 3.93])
    # create the error function
    errfunc = lambda p, t, y: fitfunc(p, t) - y
    # minimize the error function
    p1, success = op.leastsq(errfunc, p0[:], args=(x, y))
    # plot the fitted curve
    lscurve = fitfunc(p1, x)
    plt.plot(x, lscurve, 'k-')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [V]')
    #plt.show()
    plt.savefig('data/pendDat/graphs/' + name[:-4] + '.png')
    plt.close()
    rsq, SSE, SST, SSR = fit_goodness(y, lscurve)
    sigma = np.sqrt(SSE/(len(y)-len(p0))) # DoF for non-lin fit may be different
    # calculate the jacobian
    L = jac_fitfunc(p1, x)
    # the Hessian
    H = np.dot(L.T, L)
    # the covariance matrix
    U = sigma**2.*np.linalg.inv(H)
    # the standard deviations
    sigp = np.sqrt(U.diagonal())
    # add a star in the R value is low
    if rsq <= 0.99:
        rsq = str(rsq) + '*'
    else:
        pass
    # frequency and period
    wo = ufloat((p1[4], sigp[4]))
    zeta = ufloat((p1[3], sigp[3]))
    wd = (1. - zeta**2.)**(1./2.)*wo
    f = wd/2./np.pi
    T = 1./f
    fig = plt.figure(2)
    plot_osfit(x, y, lscurve, p1, rsq, T, fig=fig)
    plt.savefig('data/pendDat/graphs/atest.png')
    plt.close()
    # include the notes for the experiment
    note = pendDat['notes']
    line = name + ',' + str(T) + ',' + str(rsq) + ',' + str(sigma) + ',' + str(note) + '\n'
    file.write(line)
    print line
    # if the filename is already in the period dictionary...
    if name[:-5] in period.keys():
        # append the period to the list
        period[name[:-5]].append(T)
    # else if the filename isn't in the period dictionary...
    else:
        # start a new list
        period[name[:-5]] = [T]
file.close()
file = open('data/period.p', 'w')
p.dump(period, file)
file.close()
