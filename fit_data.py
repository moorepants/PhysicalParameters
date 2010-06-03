import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import os
import pickle as p
from uncertainties import ufloat

from benchmark_bike_tools import fit_goodness, jac_fitfunc, plot_osfit

dirs, subdirs, filenames = list(os.walk('data/pendDat/p'))[0]
file = open('data/period.txt', 'w')
filenames.sort()
period = {}
#for name in ['YellowRevForkTorsionalFirst1.p']:
#for name in ['YellowFwheelCompoundFirst1.p']:
#for name in ['StratosFrameCompoundFirst2.p']:
for name in filenames:
    df = open('data/pendDat/p/' + name)
    pendDat = p.load(df)
    df.close()
    y = pendDat['data'].ravel()
    time = pendDat['duration']
    x = np.linspace(0, time, num=len(y))
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
    rsq, SSE, SST, SSR = fit_goodness(y, lscurve)
    sigma = np.sqrt(SSE/(len(y)-len(p0)))
    # calculate the jacobian
    L = jac_fitfunc(p1, x)
    # the Hessian
    H = np.dot(L.T, L)
    # the covariance matrix
    U = sigma**2.*np.linalg.inv(H)
    # the standard deviations
    sigp = np.sqrt(U.diagonal())
    # frequency and period
    wo = ufloat((p1[4], sigp[4]))
    zeta = ufloat((p1[3], sigp[3]))
    wd = (1. - zeta**2.)**(1./2.)*wo
    f = wd/2./np.pi
    T = 1./f
    fig = plt.figure(1)
    plot_osfit(x, y, lscurve, p1, rsq, T, fig=fig)
    plt.savefig('data/pendDat/graphs/' + name[:-2] + '.png')
    plt.close()
    # add a star in the R value is low
    if rsq <= 0.99:
        rsq = str(rsq) + '*'
    else:
        pass
    # include the notes for the experiment
    try:
        note = pendDat['notes']
    except:
        note = ''
    line = name + ',' + str(T) + ',' + str(rsq) + ',' + str(sigma) + ',' + str(note) + '\n'
    file.write(line)
    print line
    # if the filename is already in the period dictionary...
    if name[:-3] in period.keys():
        # append the period to the list
        period[name[:-3]].append(T)
    # else if the filename isn't in the period dictionary...
    else:
        # start a new list
        period[name[:-3]] = [T]
file.close()
file = open('data/period.p', 'w')
p.dump(period, file)
file.close()
