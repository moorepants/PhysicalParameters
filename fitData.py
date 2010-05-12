import scipy.io.matlab.mio as mio
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import os
import pickle as p
from scipy.optimize import approx_fprime

dirs, subdirs, filenames = list(os.walk('data/pendDat'))[0]
file = open('period.txt', 'w')
filenames.sort()
period = {}
#for name in ['YellowRevForkTorsionalFirst1.mat']:
for name in ['StratosFrameCompoundFirst2.mat']:
#for name in filenames:
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
    sigma = np.sqrt((SST-SSR)/(len(y)-len(p0))) # DoF for non-lin fit may be different
    # calculate the Jacobian
    def f(x):
        '''
        Exponential decay

        Parameters:
        -----------
        x[0] : shift
        x[1] : cosine constant
        x[2] : sine constant
        x[3] : damping ratio
        x[4] : frequency
        x[5] : t

        Returns:
        --------
        y : f(x)

        '''
        return x[0] + np.exp(-x[3]*x[4]*x[5])*(x[1]*np.cos(x[4]*np.sqrt(1-x[3]**2)*x[5]) + x[2]*np.sin(x[4]*np.sqrt(1-x[3]**2)*x[5]))
    L = np.zeros((len(x), len(p1))) # time steps x parameters
    for i in range(L.shape[0]): # for each time step
        for j in range(L.shape[1]): # for each parameter
            dx = np.zeros(len(p1))
            dx[j] = 1e-5
            perturb = np.hstack((p1 + dx, x[i]))
            L[i, j] = (f(perturb) - f(np.hstack((p1, x[i]))))/dx[j]
    H = np.dot(L.T, L)
    U = sigma**2.*np.linalg.inv(H)
    # add a star in the R value is low
    if rsq <= 0.99:
        rsq = str(rsq) + '*'
    else:
        pass
    # frequency and period
    wo = p1[4]
    f = wo/2./np.pi
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
