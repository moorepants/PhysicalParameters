import scipy.io.matlab.mio as mio
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import os
import pickle as p
from scipy.optimize import approx_fprime
from uncertainties import num_with_uncert

def fit_goodness(ym, yp):
    '''
    Calculate the goodness of fit.

    Parameters:
    ----------
    ym : vector of measured values
    yp : vector of predicted values

    Returns:
    --------
    rsq: r squared value of the fit
    SSE: error sum of squares
    SST: total sum of squares
    SSR: regression sum of squares

    '''
    from numpy import sum, mean
    SSR = sum((yp - mean(ym))**2)
    SST = sum((ym - mean(ym))**2)
    SSE = SST - SSR
    rsq = SSR/SST
    return rsq, SSE, SST, SSR

def jac_fitfunc(p, t):
    '''
    Calculate the Jacobian of a decaying oscillation function.

    Uses the analytical formulations of the partial derivatives.

    Parameters:
    -----------
    p : the five parameters of the equation
    t : time

    Returns:
    --------
    jac : the partial of the vector function with respect to the parameters
    vector. A 5 x N matrix where N is the number of time steps.

    '''
    jac = np.zeros((len(p), len(t)))
    e = np.exp(-p[3]*p[4]*t)
    dampsq = np.sqrt(1 - p[3]**2)
    s = np.sin(dampsq*p[4]*t)
    c = np.cos(dampsq*p[4]*t)
    jac[0] = np.ones_like(t)
    jac[1] = e*s
    jac[2] = e*c
    jac[3] = -p[4]*t*e*(p[1]*s + p[2]*c) + e*(-p[1]*p[3]*p[4]*t/dampsq*c
            + p[2]*p[3]*p[4]*t/dampsq*s)
    jac[4] = -p[3]*t*e*(p[1]*s + p[2]*c) + e*dampsq*t*(p[1]*c - p[2]*s)
    return jac.T

dirs, subdirs, filenames = list(os.walk('data/pendDat'))[0]
file = open('period.txt', 'w')
filenames.sort()
period = {}
for name in ['YellowRevForkTorsionalFirst1.mat']:
#for name in ['StratosFrameCompoundFirst2.mat']:
#for name in filenames:
    pendDat = {}
    mio.loadmat('data/pendDat/' + name, mdict=pendDat)
    y = pendDat['data'].ravel()
    x = np.linspace(0, 30, num=len(y))
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
    plt.plot(x, lscurve, 'k')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
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
    wo = num_with_uncert((p1[4], sigp[4]))
    zeta = num_with_uncert((p1[3], sigp[3]))
    wd = (1. - zeta**2.)**(1./2.)*wo
    f = wd/2./np.pi
    T = 1./f
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
file = open('period.p', 'w')
p.dump(period, file)
file.close()
