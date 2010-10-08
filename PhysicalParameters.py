import pickle
from re import findall, sub
from os import system, walk
from copy import copy
from numpy.linalg import inv, eig, lstsq
from numpy import sin, cos, vstack, array, zeros, dot, diag
from numpy import exp, sqrt, ones_like, sum, mean, pi, unwrap
from numpy import argmin, abs, real, imag, zeros_like, max
from numpy import hsplit, finfo, eye, hstack, log10, arctan2
from numpy import poly1d, linspace, shape, ones, arange
from uncertainties import ufloat
from uncertainties.unumpy.ulinalg import inv as uinv
from uncertainties.unumpy import nominal_values, std_devs, uarray
from uncertainties.unumpy import matrix as umatrix
from uncertainties.unumpy.core import wrap_array_func
from uncertainties.unumpy import sin as usin
from uncertainties.unumpy import cos as ucos
from uncertainties.unumpy import tan as utan
from scipy.optimize import leastsq
from matplotlib.pyplot import figure, plot, xlabel, ylabel, title, legend, gca
from matplotlib.pyplot import axes, xlim, setp, close, savefig, subplot, Circle
from matplotlib.pyplot import axis, ylim, xticks, show

def fit_data():

    dirs, subdirs, filenames = list(walk('data/pendDat/p'))[0]
    f = open('data/period.txt', 'w')
    filenames.sort()
    period = {}
    #for name in ['YellowRevForkTorsionalFirst1.p']:
    #for name in ['YellowFwheelCompoundFirst1.p']:
    #for name in ['StratosFrameCompoundFirst2.p']:
    for name in filenames:
        df = open('data/pendDat/p/' + name)
        pendDat = pickle.load(df)
        df.close()
        y = pendDat['data'].ravel()
        time = pendDat['duration']
        x = linspace(0, time, num=len(y))
        # decaying oscillating exponential function
        fitfunc = lambda p, t: p[0] + exp(-p[3]*p[4]*t)*(p[1]*sin(p[4]*sqrt(1-p[3]**2)*t) + p[2]*cos(p[4]*sqrt(1-p[3]**2)*t))
        # initial guesses
        p0 = array([1.35, -.5, -.75, 0.01, 3.93])
        # create the error function
        errfunc = lambda p, t, y: fitfunc(p, t) - y
        # minimize the error function
        p1, success = leastsq(errfunc, p0[:], args=(x, y))
        # plot the fitted curve
        lscurve = fitfunc(p1, x)
        rsq, SSE, SST, SSR = fit_goodness(y, lscurve)
        sigma = sqrt(SSE/(len(y)-len(p0)))
        # calculate the jacobian
        L = jac_fitfunc(p1, x)
        # the Hessian
        H = dot(L.T, L)
        # the covariance matrix
        U = sigma**2.*inv(H)
        # the standard deviations
        sigp = sqrt(U.diagonal())
        # frequency and period
        wo = ufloat((p1[4], sigp[4]))
        zeta = ufloat((p1[3], sigp[3]))
        wd = (1. - zeta**2.)**(1./2.)*wo
        f = wd/2./pi
        T = 1./f
        fig = plt.figure(1)
        plot_osfit(x, y, lscurve, p1, rsq, T, fig=fig)
        savefig('data/pendDat/graphs/' + name[:-2] + '.png')
        close()
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
    f.close()
    f = open('data/period.p', 'w')
    pickle.dump(period, f)
    f.close()

def tor_com():

    f = open('data/period.p', 'r')
    period = pickle.load(f)
    f.close()

    tor = zeros((12, 8), dtype='object')
    com = zeros((4, 8), dtype='object')
    # list of bike names (only first letter is capatilized)
    bN = ['Browser', 'Browserins', 'Crescendo', 'Fisher', 'Pista', 'Stratos',
            'Yellow', 'Yellowrev']
    # list of the orientation angles
    fst = ['First', 'Second', 'Third']
    # list of the bicycle parts
    fffr = ['Frame', 'Fork', 'Fwheel', 'Rwheel']
    # list of type of pendulums
    tc = ['Torsional', 'Compound']
    # average the periods
    for k, v in period.items():
        # substitute names so the camel case function works
        km = sub('BrowserIns', 'Browserins', k)
        km = sub('YellowRev', 'Yellowrev', km)
        desc = space_out_camel_case(km).split()
        print desc
        if desc[0] == 'Rod':
            rodPeriod = mean(v)
        else:
            # if torsional, put it in the torsional matrix
            if desc[2] == tc[0]:
                r = fffr.index(desc[1])*3 + fst.index(desc[3])
                c = bN.index(desc[0])
                tor[r, c] = mean(v)
                print 'Added torsional to [', r, ',', c, ']'
            # if compound, put in in the compound matrix
            elif desc[2] == tc[1]:
                com[fffr.index(desc[1]), bN.index(desc[0])] = mean(v)
    avgPer = {}
    avgPer['tor'] = tor
    avgPer['com'] = com
    avgPer['rodPer'] = rodPeriod
    f = open('data/avgPer.p', 'w')
    pickle.dump(avgPer, f)
    f.close()

def calc_parameters():
    # load the data file
    f = open('data/udata.p', 'r')
    ddU = pickle.load(f)
    f.close()

    # the number of different bikes
    nBk = len(ddU['bikes'])

    # make a list of the bikes' names
    bikeNames = ddU['bikes']

    # calculate all the benchmark parameters
    par = {}

    # calculate the wheel radii
    par['rR'] = ddU['rearWheelDist']/2./pi/ddU['rearWheelRot']
    par['rF'] = ddU['frontWheelDist']/2./pi/ddU['frontWheelRot']

    # steer axis tilt in radians
    par['lambda'] = pi/180.*(90. - ddU['headTubeAngle'])

    # calculate the front wheel trail
    forkOffset = ddU['forkOffset']
    par['c'] = (par['rF']*usin(par['lambda'])
                  - forkOffset)/ucos(par['lambda'])

    # wheelbase
    par['w'] = ddU['wheelbase']

    # calculate the dees
    par['d1'] = ucos(par['lambda'])*(par['c']+par['w']-par['rR']*utan(par['lambda']))
    par['d3'] = -ucos(par['lambda'])*(par['c']-par['rF']*utan(par['lambda']))

    # calculate the frame rotation angle
    # alpha is the angle between the negative z pendulum (horizontal) and the
    # positive (up) steer axis, rotation about positive y
    alphaFrame = ddU['frameAngle']
    # beta is the angle between the x bike frame and the x pendulum frame, rotation
    # about positive y
    betaFrame = par['lambda'] - alphaFrame*pi/180

    # calculate the slope of the CoM line
    frameM = -utan(betaFrame)

    # calculate the z-intercept of the CoM line
    # frameMassDist is positive according to the pendulum ref frame
    frameMassDist = ddU['frameMassDist']
    cb = ucos(betaFrame)
    frameB = -frameMassDist/cb - par['rR']

    # calculate the fork rotation angle
    betaFork = par['lambda'] - ddU['forkAngle']*pi/180.

    # calculate the slope of the fork CoM line
    forkM = -utan(betaFork)

    # calculate the z-intercept of the CoM line
    forkMassDist = ddU['forkMassDist']
    cb = ucos(betaFork)
    tb = utan(betaFork)
    forkB = - par['rF'] - forkMassDist/cb + par['w']*tb

    # plot the CoM lines
    comFig = figure(num=1)
    # intialize the matrices for the center of mass locations
    frameCoM = zeros((2, shape(frameM)[1]), dtype='object')
    forkCoM = zeros((2, shape(forkM)[1]), dtype='object')
    # for each of the bikes...
    for i in range(shape(frameM)[1]):
        comb = array([[0, 1], [0, 2], [1, 2]])
        # calculate the frame center of mass position
        # initialize the matrix to store the line intersections
        lineX = zeros((3, 2), dtype='object')
        # for each line intersection...
        for j, row in enumerate(comb):
            a = umatrix(vstack([-frameM[row, i], ones((2))]).T)
            b = frameB[row, i]
            lineX[j] = dot(a.I, b)
        frameCoM[:, i] = mean(lineX, axis=0)
        # calculate the fork center of mass position
        # reinitialize the matrix to store the line intersections
        lineX = zeros((3, 2), dtype='object')
        # for each line intersection...
        for j, row in enumerate(comb):
            a = umatrix(vstack([-forkM[row, i], ones((2))]).T)
            b = forkB[row, i]
            lineX[j] = dot(a.I, b)
        forkCoM[:, i] = mean(lineX, axis=0)
        # make a subplot for this bike
        subplot(2, 4, i + 1)
        # plot the rear wheel
        c = Circle((0, par['rR'][i].nominal_value), radius=par['rR'][i].nominal_value)
        gca().add_patch(c)
        # plot the front wheel
        c = Circle((par['w'][i].nominal_value, par['rF'][i].nominal_value), radius=par['rF'][i].nominal_value)
        gca().add_patch(c)
        # plot the lines (pendulum axes)
        x = linspace(-par['rR'][i].nominal_value, par['w'][i].nominal_value + par['rF'][i].nominal_value, 2)
        # for each line...
        for j in range(len(frameM)):
            framey = -frameM[j, i].nominal_value*x - frameB[j, i].nominal_value
            forky = -forkM[j, i].nominal_value*x - forkB[j, i].nominal_value
            plot(x,framey, 'r')
            plot(x,forky, 'g')
        # plot the ground line
        plot(x, zeros_like(x), 'k')
        # plot the fundamental bike
        deex = zeros(4)
        deez = zeros(4)
        deex[0] = 0.
        deex[1] = (par['d1'][i]*ucos(par['lambda'][i])).nominal_value
        deex[2] = (par['w'][i]-par['d3'][i]*ucos(par['lambda'][i])).nominal_value
        deex[3] = par['w'][i].nominal_value
        deez[0] = -par['rR'][i].nominal_value
        deez[1] = -(par['rR'][i]+par['d1'][i]*usin(par['lambda'][i])).nominal_value
        deez[2] = -(par['rF'][i]-par['d3'][i]*usin(par['lambda'][i])).nominal_value
        deez[3] = -par['rF'][i].nominal_value
        plot(deex, -deez, 'k')
        # plot the centers of mass
        plot(frameCoM[0, i].nominal_value, -frameCoM[1, i].nominal_value, 'k+', markersize=12)
        plot(forkCoM[0, i].nominal_value, -forkCoM[1, i].nominal_value, 'k+', markersize=12)
        axis('equal')
        ylim((0, 1))
        title(bikeNames[i])
    par['xB'] = frameCoM[0, :]
    par['zB'] = frameCoM[1, :]
    par['xH'] = forkCoM[0, :]
    par['zH'] = forkCoM[1, :]

    # load the average period data
    f = open('data/avgPer.p', 'r')
    avgPer = pickle.load(f)
    f.close()
    # torsional, compound and rod periods
    tor = avgPer['tor']
    com = avgPer['com']
    # the yellow bikes have the same frame
    tor[:3, 7] = tor[:3, 6]
    com[0, 7] = com[0, 6]
    # the browser's have the same fork
    tor[3:6, 0] = tor[3:6, 1]
    com[1, 0] = com[1, 1]
    # the browsers have the same front and rear wheels
    tor[6, 1] = tor[6, 0]
    tor[9, 1] = tor[9, 0]
    com[2, 1] = com[2, 0]
    com[3, 1] = com[3, 0]
    # the yellow bikes have the same front and rear wheels
    tor[6, 7] = tor[6, 6]
    tor[9, 7] = tor[9, 6]
    com[2, 7] = com[2, 6]
    com[3, 7] = com[3, 6]

    tRod = avgPer['rodPer']
    # calculate the stiffness of the torsional pendulum
    iRod = tube_inertia(ddU['lRod'], ddU['mRod'], ddU['rRod'], 0.)[1]
    k = tor_stiffness(iRod, tRod)

    # masses
    par['mR'] = ddU['rearWheelMass']
    par['mF'] = ddU['frontWheelMass']
    par['mB'] = ddU['frameMass']
    par['mH'] = ddU['forkMass']

    # calculate the wheel y inertias
    par['g'] = 9.81*ones(ddU['forkMass'].shape, dtype=float)
    par['IRyy'] = com_inertia(par['mR'], par['g'], ddU['rWheelPendLength'], com[3, :])
    par['IFyy'] = com_inertia(par['mF'], par['g'], ddU['fWheelPendLength'], com[2, :])

    # calculate the wheel x/z inertias
    par['IRxx'] = tor_inertia(k, tor[6, :])
    par['IFxx'] = tor_inertia(k, tor[9, :])

    # calculate the y inertias for the frame and fork
    framePendLength = (frameCoM[0, :]**2 + (frameCoM[1, :] + par['rR'])**2)**(0.5)
    par['IByy'] = com_inertia(par['mB'], par['g'], framePendLength, com[0, :])
    forkPendLength = ((forkCoM[0, :] - par['w'])**2 + (forkCoM[1, ] + par['rF'])**2)**(0.5)
    par['IHyy'] = com_inertia(par['mH'], par['g'], forkPendLength, com[1, :])

    # calculate the fork in-plane moments of inertia
    Ipend = tor_inertia(k, tor)
    par['IHxx'] = []
    par['IHxz'] = []
    par['IHzz'] = []
    for i, row in enumerate(Ipend[3:6, :].T):
        Imat = inertia_components_uncert(row, betaFork[:, i])
        par['IHxx'].append(Imat[0, 0])
        par['IHxz'].append(Imat[0, 1])
        par['IHzz'].append(Imat[1, 1])
    par['IHxx'] = array(par['IHxx'])
    par['IHxz'] = array(par['IHxz'])
    par['IHzz'] = array(par['IHzz'])

    # calculate the frame in-plane moments of inertia
    par['IBxx'] = []
    par['IBxz'] = []
    par['IBzz'] = []
    for i, row in enumerate(Ipend[:3, :].T):
        Imat = inertia_components_uncert(row, betaFrame[:, i])
        par['IBxx'].append(Imat[0, 0])
        par['IBxz'].append(Imat[0, 1])
        par['IBzz'].append(Imat[1, 1])
    par['IBxx'] = array(par['IBxx'])
    par['IBxz'] = array(par['IBxz'])
    par['IBzz'] = array(par['IBzz'])
    par['v'] = ones_like(par['g'])

    # make a dictionary with only the nominal values
    par_n = {}
    for k, v in par.items():
        if type(v[0]) == type(par['rF'][0]) or type(v[0]) == type(par['mF'][0]):
            par_n[k] = nominal_values(v)
        else:
            par_n[k] = par[k]

    # plot all the parameters to look for crazy numbers
    parFig = figure(num=2)
    i = 1
    xt = ['B', 'BI', 'C', 'F', 'P', 'S', 'Y', 'YR']
    for k, v in par_n.items():
        subplot(5, 6, i)
        plot(v, '-D', markersize=14)
        title(k)
        xticks(arange(8), tuple(xt))
        i += 1
    show()

    # write the parameter files
    for i, name in enumerate(bikeNames):
        dir = 'data/bikeParameters/'
        fname = ''.join(name.split())
        f = open(dir + fname + 'Par.txt', 'w')
        for k, v in par.items():
            if type(v[i]) == type(par['rF'][0]) or type(v[i]) == type(par['mF'][0]):
                line = k + ',' + str(v[i].nominal_value) + ',' + str(v[i].std_dev()) + '\n'
            else:
                line = k + ',' + str(v[i]) + ',' + '0.0' + '\n'
            f.write(line)
        f.close()

    # pickle the parameters too
    f = open('data/par.p', 'w')
    pickle.dump(par, f)
    f.close()

def ab_to_mck(A, B):
    '''
    Returns the spring-mass-damper matrices for a general system.

    Parameters:
    -----------
    A : ndarray, shape(n,n)
    B : ndarray, shape(n/2,n)

    Returns:
    --------
    M : ndarray, shape(n/2,n/2)
    C : ndarray, shape(n/2,n/2)
    K : ndarray, shape(n/2/n/2)

    This function converts the linear set of equations x' = Ax + Bu to the
    canonical form Mq'' + Cq' + Kq = f. The states, x, must be ordered
    sequentially with the configuration variables proceeding the rates (i.e.
    x = [x1, ..., xn, x1', ..., xn'] and q = [x1, ..., xn]).

    '''
    numRow, numCol = A.shape
    M = inv(B[numRow/2:, :])
    C = -dot(M, A[numRow/2:, numCol/2:])
    K = -dot(M, A[numRow/2:, :numCol/2])

    return M, C, K

def ueig2(uA):

    ueig0 = wrap_array_func(lambda m: numpy.linalg.eig(m)[0])
    ueig1 = wrap_array_func(lambda m: numpy.linalg.eig(m)[1])

    def ueig(m):
        """
        Version of numpy.linalg.eig that works on matrices that contain
        numbers with uncertainties.
        """
        return (ueig0(m), ueig1(m))

    return ueig(uA)

def ueig(uA):
    '''
    Returns eigenvalues and eigenvectors with uncertainties.

    Parameters:
    -----------
    uA : ndarry, shape(n x n)

    Returns:
    --------
    uw : ndarry, shape = (n,)
    uv : ndarray, shape = (n, n)

    uses both numpy.linalg.eig and the uncertainties package to calculate eigen
    values and eigenvectors with uncertainties

    '''

    # separate the nominal values from the uncertainties
    sA = std_devs(uA)

    # create the covariance matrix for A
    CA = diag(sA.flatten())

    # pull out the nominal A
    A = nominal_values(uA)

    # the nominal eigenvalues and eigenvectors
    w, v = eig(A)
    print 'w=', w

    # FA is the jacobian
    FAw = zeros((w.shape[0], A.flatten().shape[0]))
    FAv = zeros((v.flatten().shape[0], A.flatten().shape[0]))

    # pw is the perturbed eigenvectors used in the FA calc
    pw = zeros((w.shape[0], A.flatten().shape[0]), dtype=complex)
    pv = zeros((w.shape[0]**2, A.flatten().shape[0]), dtype=complex)

    # calculate the perturbed eigenvalues for each A entry
    for i, a in enumerate(A.flatten()):
        # set the differentiation step
        if a == 0.:
            delta = 1e-8
        else:
            delta = sqrt(finfo(float).eps)*a

        # make a copy of A
        pA = copy(A).flatten()

        # perturb the entry
        pA[i] = pA[i] + delta

        # back to matrix
        pA = hsplit(pA, A.shape[0])
        print 'A', A
        print 'pA', pA

        # calculate the eigenvalues
        print eig(pA)[0]
        pw[:, i], tpv = eig(pA)
        print 'perturbed eig', pw[:, i]
        print 'nom eig', w
        print 'delta', delta
        FAw[:, i] = (pw[:, i] - w)/delta
        print "FAw column=", FAw[:, i]
        pv[:, i] = tpv.flatten('F')
        FAv[:, i] = (pv[:, i] - v.flatten('F'))/delta

    # calculate the covariance matrix for the eigenvalues
    Cw = dot(dot(FAw, CA), FAw.T)
    Cv = dot(dot(FAv, CA), FAv.T)
    print FAw

    # build the eigenvalues with uncertainties
    uw = uarray((w, diag(Cw)))
    uv = uarray((v.flatten('F'), diag(Cv)))
    uv = vstack(hsplit(uv, w.shape[0])).T
    return uw, uv

def replace_values(directory, template, newfile, replacers):
    '''
    Replaces variables with values in a text file.

    Parameters:
    -----------
    directory: string that locates the template file
    template: string text file
    newfile: string  text file
    replacers: dictionary

    Returns:
    --------

    This looks for |varible_name| and replaces it with the value stored in
    variable_name. It also looks for |variable_name?| and replaces it with the
    uncertainty of the value of variable_name.

    '''
    print replacers

    # open the template file
    f = open(directory + template, 'r')
    # open the new file
    fn = open(directory + newfile, 'w')
    for line in f:
        print 'This is the template line:\n', line
        # find all of the matches in the line
        test = findall('\|(\w*.)\|', line)
        print 'these are all the matches', test
        print 'search for nom: ', findall('\|(\w*)(?!\?)\|', line)
        print 'search for un: ',  findall('\|(\w*\?)\|', line)
        # if there are matches
        if test:
            # go through each match and make a substitution
            for match in test:
                print "replace this: ", match
                # if there is a '_' then the match is an array entry
                if '_' in match:
                    ques = None
                    try:
                        var, row, col, ques = match.split('_')
                    except:
                        var, row, col = match.split('_')
                    valStr = "replacers['" + var + "']" + '[' + row + ', ' + col + ']'
                    #print valStr
                    value = eval(valStr)
                    print 'value =', value
                    # if the match has a question mark it is an uncertainty value
                    if ques:
                        try:
                            replacement = uround(value).split('+/-')[1]
                            print "with this:", replacement
                            line = sub('\|(\w*\?)\|', replacement, line, count=1)
                        except: # there is no uncertainty
                            replacement = '0.0'
                            print "with this:", replacement
                            line = sub('\|(\w*\?)\|', replacement, line, count=1)
                    else:
                        try:
                            replacement = uround(value).split('+/-')[0]
                            print "with this:", replacement
                            line = sub('\|(\w*(?!\?))\|', replacement, line, count=1)
                        except: # there is no uncertainty
                            replacement = str(value)
                            print "with this:", replacement
                            line = sub('\|(\w*(?!\?))\|', replacement, line, count=1)
                    del var, row, col
                # else the match is just a scalar
                else:
                    # if the match has a question mark it is an uncertainty value
                    if match[-1] == '?':
                        try:
                            line = sub('\|(\w*\?)\|',
                                    uround(replacers[match[:-1]]).split('+/-')[1], line, count=1)
                            print line
                        except:
                            pass
                    else:
                        try:
                            line = sub('\|(\w*(?!\?))\|',
                                    uround(replacers[match]).split('+/-')[0], line, count=1)
                            print line
                        except:
                            line = sub('\|(\w*(?!\?))\|', str(replacers[match][i]), line, count=1)
        print line
        fn.write(line)
    f.close()
    fn.close()
    system('pdflatex -output-directory=' + directory + ' ' + newfile)
    system('rm ' + directory + '*.aux')
    system('rm ' + directory + '*.log')

def uround(value):
    '''Round values according to their uncertainity

    Parameters:
    -----------
    value: float with uncertainty

    Returns:
    --------
    s: string that is properly rounded

    2.4563752289999+/-0.0003797273827

    becomes

    2.4564+/-0.0004

    This doesn't work for weird cases like large uncertainties.
    '''
    try:
        # grab the nominal value and the uncertainty
        nom = value.nominal_value
        uncert = value.std_dev()
        # convert the uncertainty to a string
        s = str(uncert)
        # find the first non-zero character
        for j, number in enumerate(s):
            if number == '0' or number == '.':
                pass
            else:
                digit = j
                break
        newUncert = round(uncert, digit-1)
        newNom = round(nom, len(str(newUncert)) - 2)
        newValue = ufloat((newNom, newUncert))
        diff = len(str(newUncert)) - len(str(newNom))
        if diff > 0:
            s = str(newNom) + int(diff)*'0' + '+/-' +str(newUncert)
        else:
            s = str(newValue)
    except:
        s = str(value)
    return s

def plot_osfit(t, ym, yf, p, rsq, T, fig=None):
    '''Plot fitted data over the measured

    Parameters:
    -----------
    t : ndarray (n,)
        Measurement time in seconds
    ym : ndarray (n,)
        The measured voltage
    yf : ndarray (n,)
    p : ndarray (5,)
        The fit parameters for the decaying osicallation fucntion
    rsq : float
        The r squared value of y (the fit)
    T : float
        The period

    Returns:
    --------
    fig : the figure

    '''
    if fig:
        fig = fig
    else:
        fig = figure(2)
    ax1 = axes([0.1, 0.1, 0.8, 0.7])
    ax1.plot(t, ym, '.', markersize=2)
    plot(t, yf, 'k-')
    xlabel('Time [s]')
    ylabel('Amplitude [V]')
    equation = r'$f(t)={0:1.2f}+e^{{-({3:1.3f})({4:1.1f})t}}\left[{1:1.2f}\sin{{\sqrt{{1-{3:1.3f}^2}}{4:1.1f}t}}+{2:1.2f}\cos{{\sqrt{{1-{3:1.3f}^2}}{4:1.1f}t}}\right]$'.format(p[0], p[1], p[2], p[3], p[4])
    rsquare = '$r^2={0:1.3f}$'.format(rsq)
    period = '$T={0} s$'.format(T)
    title(equation + '\n' + rsquare + ', ' + period)
    legend(['Measured', 'Fit'])
    #xlim((0, 1))
    return fig

def space_out_camel_case(s):
        """Adds spaces to a camel case string.  Failure to space out string
        returns the original string.
        >>> space_out_camel_case('DMLSServicesOtherBSTextLLC')
        'DMLS Services Other BS Text LLC'
        """
        return sub('((?=[A-Z][a-z])|(?<=[a-z])(?=[A-Z]))', ' ', s).strip()

def bode(ABCD=None, numden=None, w=None, fig=None, n=None, label=None,
        title=None, color=None):
    """Bode plot.

    Takes the system A, B, C, D matrices of the state space system.

    Need to implement transfer function num/den functionality.

    Returns magnitude and phase vectors, and figure object.
    """
    if fig == None:
        fig = figure()

    mag = zeros(len(w))
    phase = zeros(len(w))
    fig.yprops = dict(rotation=90,
                  horizontalalignment='right',
                  verticalalignment='center',
                  x=-0.01)

    fig.axprops = {}
    fig.ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4], **fig.axprops)
    fig.ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4], **fig.axprops)

    if (ABCD):
        A, B, C, D = ABCD
        I = eye(A.shape[0])
        for i, f in enumerate(w):
            sImA_inv = inv(1j*f*I - A)
            G = dot(dot(C, sImA_inv), B) + D
            mag[i] = 20.*log10(abs(G))
            phase[i] = arctan2(imag(G), real(G))
        phase = 180./np.pi*unwrap(phase)
    elif (numden):
        n = poly1d(numden[0])
        d = poly1d(numden[1])
        Gjw = n(1j*w)/d(1j*w)
        mag = 20.*log10(abs(Gjw))
        phase = 180./pi*unwrap(arctan2(imag(Gjw), real(Gjw)))

    fig.ax1.semilogx(w, mag, label=label)
    if title:
        fig.ax1.set_title(title)
    fig.ax2.semilogx(w, phase, label=label)


    fig.axprops['sharex'] = fig.axprops['sharey'] = fig.ax1
    fig.ax1.grid(b=True)
    fig.ax2.grid(b=True)

    setp(fig.ax1.get_xticklabels(), visible=False)
    setp(fig.ax1.get_yticklabels(), visible=True)
    setp(fig.ax2.get_yticklabels(), visible=True)
    fig.ax1.set_ylabel('Magnitude [dB]', **fig.yprops)
    fig.ax2.set_ylabel('Phase [deg]', **fig.yprops)
    fig.ax2.set_xlabel('Frequency [rad/s]')
    if label:
        fig.ax1.legend()

    if color:
        setp(fig.ax1.lines, color=color)
        setp(fig.ax2.lines, color=color)

    return mag, phase, fig

def sort_modes(evals, evecs):
    '''
    Sort eigenvalues and eigenvectors into weave, capsize, caster modes.

    Parameters
    ----------
    evals : ndarray, shape (n, 4)
        eigenvalues
    evecs : ndarray, shape (n, 4, 4)
        eigenvectors

    Returns
    -------
    weave['evals'] : ndarray, shape (n, 2)
        The eigen value pair associated with the weave mode.
    weave['evecs'] : ndarray, shape (n, 4, 2)
        The associated eigenvectors of the weave mode.
    capsize['evals'] : ndarray, shape (n,)
        The real eigenvalue associated with the capsize mode.
    capsize['evecs'] : ndarray, shape(n, 4, 1)
        The associated eigenvectors of the capsize mode.
    caster['evals'] : ndarray, shape (n,)
        The real eigenvalue associated with the caster mode.
    caster['evecs'] : ndarray, shape(n, 4, 1)
        The associated eigenvectors of the caster mode.

    This only works on the standard bicycle eigenvalues, not necessarily on any
    general eigenvalues for the bike model (e.g. there isn't always a distinct weave,
    capsize and caster). Some type of check using the derivative of the curves
    could make it more robust.
    '''
    evalsorg = zeros_like(evals)
    evecsorg = zeros_like(evecs)
    # set the first row to be the same
    evalsorg[0] = evals[0]
    evecsorg[0] = evecs[0]
    # for each speed
    for i, speed in enumerate(evals):
        if i == evals.shape[0] - 1:
            break
        # for each current eigenvalue
        used = []
        for j, e in enumerate(speed):
            try:
                x, y = real(evalsorg[i, j].nominal_value), imag(evalsorg[i, j].nominal_value)
            except:
                x, y = real(evalsorg[i, j]), imag(evalsorg[i, j])
            # for each eigenvalue at the next speed
            dist = zeros(4)
            for k, eignext in enumerate(evals[i + 1]):
                try:
                    xn, yn = real(eignext.nominal_value), imag(eignext.nominal_value)
                except:
                    xn, yn = real(eignext), imag(eignext)
                # distance between points in the real/imag plane
                dist[k] = abs(((xn - x)**2 + (yn - y)**2)**0.5)
            if argmin(dist) in used:
                # set the already used indice higher
                dist[argmin(dist)] = max(dist) + 1.
            else:
                pass
            evalsorg[i + 1, j] = evals[i + 1, argmin(dist)]
            evecsorg[i + 1, :, j] = evecs[i + 1, :, argmin(dist)]
            # keep track of the indices we've used
            used.append(argmin(dist))
    weave = {'evals' : evalsorg[:, 2:], 'evecs' : evecsorg[:, :, 2:]}
    capsize = {'evals' : evalsorg[:, 1], 'evecs' : evecsorg[:, :, 1]}
    caster = {'evals' : evalsorg[:, 0], 'evecs' : evecsorg[:, :, 0]}
    return weave, capsize, caster

def critical_speeds(v, weave, capsize):
    '''
    Return critical speeds of the benchmark bicycle.

    Parameters
    ----------
    v : ndarray, shape (n,)
        Speed
    weave : ndarray, shape (n, 2)
        The weave eignevalue pair
    capsize : ndarray, shape (n, 1)

    Returns
    -------
    vd : float
        The speed at which the weave mode goes from two real eigenvalues to a
        complex pair.
    vw : float
        The speed at which the weave mode becomes stable.
    vc : float
        The speed at which the capsize mode becomes unstable.

    '''
    vw = v[argmin(abs(real(weave[:, 0])))]
    vc = v[argmin(abs(real(capsize)))]
    m = max(abs(imag(weave[:, 0])))
    w = zeros_like(imag(weave[:, 0]))
    for i, eig in enumerate(abs(imag(weave[:, 0]))):
        if eig == 0.:
            w[i] = m + 1.
        else:
            w[i] = eig
    vd = v[argmin(w)]
    return vd, vw, vc

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
    t : time vector

    Returns:
    --------
    jac : The jacobian, the partial of the vector function with respect to the
    parameters vector. A 5 x N matrix where N is the number of time steps.

    '''
    jac = zeros((len(p), len(t)))
    e = exp(-p[3]*p[4]*t)
    dampsq = sqrt(1 - p[3]**2)
    s = sin(dampsq*p[4]*t)
    c = cos(dampsq*p[4]*t)
    jac[0] = ones_like(t)
    jac[1] = e*s
    jac[2] = e*c
    jac[3] = -p[4]*t*e*(p[1]*s + p[2]*c) + e*(-p[1]*p[3]*p[4]*t/dampsq*c
            + p[2]*p[3]*p[4]*t/dampsq*s)
    jac[4] = -p[3]*t*e*(p[1]*s + p[2]*c) + e*dampsq*t*(p[1]*c - p[2]*s)
    return jac.T

def bike_eig(M, C1, K0, K2, v, g):
    '''
    Return eigenvalues and eigenvectors of the benchmark bicycle.

    Parameters:
    -----------
    M : ndarray, shape (2, 2)
        mass matrix
    C1 : ndarray, shape (2, 2)
        damping like matrix
    K0 : ndarray, shape (2, 2)
        stiffness matrix proportional to gravity
    K2 : ndarray, shape (2, 2)
        stiffness matrix proportional to the square of velocity
    v : ndarray, shape (n,)
        an array of speeds for which to calculate eigenvalues
    g : float
        local acceleration due to gravity in meters per seconds squared

    Returns:
    --------
    evals : ndarray, shape (n, 4)
        eigenvalues
    evecs : ndarray, shape (n, 4, 4)
        eigenvectors

    '''
    m, n = 2*M.shape[0], v.shape[0]
    evals = zeros((n, m), dtype='complex128')
    evecs = zeros((n, m, m), dtype='complex128')
    for i, speed in enumerate(v):
        A, B = abMatrix(M, C1, K0, K2, speed, g)
        w, vec = eig(nominal_values(A))
        evals[i] = w
        evecs[i] = vec
    return evals, evecs

def bmp2cm(filename):
    '''Return the benchmark canonical matrices from the bicycle parameters. Formulated from Meijaard et al. 2007.

    Parameters:
    -----------
    filename: file
        is a text file with the 27 parameters listed in csv format. One
        parameter per line in the form: 'lamba,10/pi' or 'v,5'. Use the
        benchmark paper's units!
    Returns:
    --------
        M is the mass matrix
        C1 is the damping like matrix that is proportional to v
        K0 is the stiffness matrix proportional to gravity
        K2 is the stiffness matrix proportional to v**2
        p is a dictionary of the parameters. the keys are the variable names

    This function handles parameters with uncertanties.

        '''
    f = open(filename, 'r')
    p = {}
    # parse the text file
    for i, line in enumerate(f):
        list1 = line[:-1].split(',')
        # if there is an uncertainty value try to make a ufloat
        try:
            p[list1[0]] = ufloat((eval(list1[1]), eval(list1[2])))
        # else keep it as a float
        except:
            p[list1[0]] = eval(list1[1])
    mT = p['mR'] + p['mB'] + p['mH'] + p['mF']
    xT = (p['xB']*p['mB'] + p['xH']*p['mH'] + p['w']*p['mF'])/mT
    zT = (-p['rR']*p['mR'] + p['zB']*p['mB'] + p['zH']*p['mH'] - p['rF']*p['mF'])/mT
    ITxx = p['IRxx'] + p['IBxx'] + p['IHxx'] + p['IFxx'] +p['mR']*p['rR']**2 + p['mB']*p['zB']**2 + p['mH']*p['zH']**2 + p['mF']*p['rF']**2
    ITxz = p['IBxz'] + p['IHxz'] - p['mB']*p['xB']*p['zB'] - p['mH']*p['xH']*p['zH'] + p['mF']*p['w']*p['rF']
    p['IRzz'] = p['IRxx']
    p['IFzz'] = p['IFxx']
    ITzz = p['IRzz'] + p['IBzz'] + p['IHzz'] + p['IFzz'] + p['mB']*p['xB']**2 + p['mH']*p['xH']**2 + p['mF']*p['w']**2
    mA = p['mH'] + p['mF']
    xA = (p['xH']*p['mH'] + p['w']*p['mF'])/mA
    zA = (p['zH']*p['mH'] - p['rF']*p['mF'])/mA
    IAxx = p['IHxx'] + p['IFxx'] + p['mH']*(p['zH'] - zA)**2 + p['mF']*(p['rF'] + zA)**2
    IAxz = p['IHxz'] - p['mH']*(p['xH'] - xA)*(p['zH'] - zA) + p['mF']*(p['w'] - xA)*(p['rF'] + zA)
    IAzz = p['IHzz'] + p['IFzz'] + p['mH']*(p['xH'] - xA)**2 + p['mF']*(p['w'] - xA)**2
    uA = (xA - p['w'] - p['c'])*cos(p['lambda']) - zA*sin(p['lambda'])
    IAll = mA*uA**2 + IAxx*sin(p['lambda'])**2 + 2*IAxz*sin(p['lambda'])*cos(p['lambda']) + IAzz*cos(p['lambda'])**2
    IAlx = -mA*uA*zA + IAxx*sin(p['lambda']) + IAxz*cos(p['lambda'])
    IAlz = mA*uA*xA + IAxz*sin(p['lambda']) + IAzz*cos(p['lambda'])
    mu = p['c']/p['w']*cos(p['lambda'])
    SR = p['IRyy']/p['rR']
    SF = p['IFyy']/p['rF']
    ST = SR + SF
    SA = mA*uA + mu*mT*xT
    Mpp = ITxx
    Mpd = IAlx + mu*ITxz
    Mdp = Mpd
    Mdd = IAll + 2*mu*IAlz + mu**2*ITzz
    M = array([[Mpp, Mpd], [Mdp, Mdd]])
    K0pp = mT*zT # this value only reports to 13 digit precision it seems?
    K0pd = -SA
    K0dp = K0pd
    K0dd = -SA*sin(p['lambda'])
    K0 = array([[K0pp, K0pd], [K0dp, K0dd]])
    K2pp = 0.
    K2pd = (ST - mT*zT)/p['w']*cos(p['lambda'])
    K2dp = 0.
    K2dd = (SA + SF*sin(p['lambda']))/p['w']*cos(p['lambda'])
    K2 = array([[K2pp, K2pd], [K2dp, K2dd]])
    C1pp = 0.
    C1pd = mu*ST + SF*cos(p['lambda']) + ITxz/p['w']*cos(p['lambda']) - mu*mT*zT
    C1dp = -(mu*ST + SF*cos(p['lambda']))
    C1dd = IAlz/p['w']*cos(p['lambda']) + mu*(SA + ITzz/p['w']*cos(p['lambda']))
    C1 = array([[C1pp, C1pd], [C1dp, C1dd]])
    return M, C1, K0, K2, p

def abMatrix(M, C1, K0, K2, v, g):
    '''Calculate the A and B matrices for the benchmark bicycle

    Parameters:
    -----------
        M : the mass matrix
        C1 : the damping like matrix that is proportional to v
        K0 : the stiffness matrix proportional to gravity
        K2 : the stiffness matrix proportional to v**2
        v : speed
        g : acceleration due to gravity
    Returns:
        A : system dynamic matrix
        B : control matrix

    The states are [roll rate, steer rate, roll angle, steer angle]
    '''

    a11 = -v*C1
    a12 = -(g*K0 + v**2*K2)
    a21 = eye(2)
    a22 = zeros((2, 2))
    A = vstack((dot(uinv(M), hstack((a11, a12))), hstack((a21, a22))))
    B = vstack((uinv(M), zeros((2, 2))))
    return A, B

def tor_inertia(k, T):
    '''Calculate the moment of interia for an ideal torsional pendulm

    Parameters:
    -----------
    k: torsional stiffness
    T: period

    Returns:
    --------
    I: moment of inertia

    '''

    I = k*T**2./4./pi**2.

    return I

def com_inertia(m, g, l, T):
    '''Calculate the moment of inertia for an object hung as a compound
    pendulum

    Parameters:
    -----------
    m: mass
    g: gravity
    l: length
    T: period

    Returns:
    --------
    I: moment of interia

    '''

    I = (T/2./pi)**2.*m*g*l - m*l**2.

    return I

def tube_inertia(l, m, ro, ri):
    '''Calculate the moment of inertia for a tube (or rod) where the x axis is
    aligned with the tube's axis

    l: length
    m: mass
    ro: outer radius
    ri: inner radius
    Ix: moment of inertia about tube axis
    Iy, Iz: moment of inertia about normal axis

    '''
    Ix = m/2.*(ro**2 + ri**2)
    Iy = m/12.*(3*ro**2 + 3*ri**2 + l**2)
    Iz = Iy
    return array([Ix, Iy, Iz])

def tor_stiffness(I, T):
    '''Calculate the stiffness of a torsional pendulum with a known moment of
    inertia

    Parameters
    ----------
    I : moment of inertia
    T : period

    Returns
    -------
    k : stiffness

    '''
    k = 4.*I*pi**2/T**2
    return k

def inertia_components(I, alpha):
    '''Calculate the 2D orthongonal inertia tensor

    When at least three moments of inertia and their axes orientations are
    known relative to a common inertial frame, the moments of inertia relative
    the frame are computed.

    Parameters
    ----------

    I : A vector of at least three moments of inertia about various axes

    alpha : A vector of orientation angles (positive rotation) relative to the
    known frame corresponding to the moments of inertia

    Returns
    -------

    Inew : An inertia tensor

    '''
    sa = sin(alpha)
    ca = cos(alpha)
    A = vstack((ca**2, -2*sa*ca, sa**2)).T
    Iorth = lstsq(A, I)[0]
    Inew = array([[Iorth[0], Iorth[1]], [Iorth[1], Iorth[2]]])
    return Inew

def inertia_components_uncert(I, alpha):
    '''Calculate the 2D orthongonal inertia tensor

    When at least three moments of inertia and their axes orientations are
    known relative to a common inertial frame, the moments of inertia relative
    the frame are computed.

    Parameters
    ----------

    I : A vector of at least three moments of inertia

    alpha : A vector of orientation angles corresponding to the moments of
            inertia

    Returns
    -------

    Inew : An inertia tensor

    '''
    sa = usin(alpha)
    ca = ucos(alpha)
    A = umatrix(vstack((ca**2, -2*sa*ca, sa**2)).T)
    Iorth = dot(A.I, I)
    Iorth = array([Iorth[0, 0], Iorth[0, 1], Iorth[0, 2]], dtype='object')
    Inew = array([[Iorth[0], Iorth[1]], [Iorth[1], Iorth[2]]])
    return Inew

def parallel_axis(Ic, m, d):
    '''Parallel axis thereom. Takes the moment of inertia about the rigid
    body's center of mass and translates it to a new reference frame that is
    the distance, d, from the center of mass.'''
    a = d[0]
    b = d[1]
    c = d[2]
    dMat = zeros((3, 3))
    dMat[0] = array([b**2 + c**2, -a*b, -a*c])
    dMat[1] = array([-a*b, c**2 + a**2, -b*c])
    dMat[2] = array([-a*c, -b*c, a**2 + b**2])
    return Ic + m*dMat

def trail(rF, lam, fo):
    '''Caluculate the trail and mechanical trail

    Parameters:
    -----------
    rF: float
        The front wheel radius
    lam: float
        The steer axis tilt (pi/2 - headtube angle). The angle between the
        headtube and a vertical line.
    fo: float
        The fork offset

    Returns:
    --------
    c: float
        Trail
    cm: float
        Mechanical Trail

    '''

    # trail
    c = (rF*sin(lam) - fo)/cos(lam)
    # mechanical trail
    cm = c*cos(lam)
    return c, cm
