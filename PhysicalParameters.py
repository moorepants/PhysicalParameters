import pickle
import re
import os
from copy import copy
from math import pi
import numpy as np
from uncertainties import ufloat, unumpy, umath
from uncertainties.unumpy.core import wrap_array_func
from scipy.optimize import leastsq
from scipy.io import savemat
import matplotlib.pyplot as plt

def make_tables():

    # load in the base data file
    f = open('data/data.p', 'r')
    data = pickle.load(f)
    f.close()

    # load in the parameter data file
    f = open('data/par.p', 'r')
    par = pickle.load(f)
    f.close()

    direct = 'tables/'
    for i, name in enumerate(data['bikes']):
        fname = ''.join(name.split())
        # open the new file
        f = open(direct + 'ParameterTable.tex', 'r')
        fn = open(direct + fname + 'Par.tex', 'w')
        for line in f:
            #print line
            # find all of the matches in the line
            test = re.findall('\|(\w*.)\|', line)
            #print 'search for u: ',  re.findall('\|(\w*\?)\|', line)
            #print 'search for nom: ', re.findall('\|(\w*)(?!\?)\|', line)
            # if there are matches
            if test:
                #print 'Found this!\n', test
                # go through each match and make a substitution
                for match in test:
                    #print "replace this: ", match
                    # if the match has a question mark it is an uncertainty value
                    if match[-1] == '?':
                        try:
                            line = re.sub('\|(\w*\?)\|',
                                    uround(par[match[:-1]][i]).split('+/-')[1], line, count=1)
                            #print line
                        except:
                            pass
                    # else if the match is the bicycle name
                    elif match == 'bikename':
                        line = re.sub('\|bikename\|', name.upper(), line)
                    else:
                        try:
                            line = re.sub('\|(\w*(?!\?))\|',
                                    uround(par[match][i]).split('+/-')[0], line, count=1)
                            #print line
                        except:
                            line = re.sub('\|(\w*(?!\?))\|', str(par[match][i]), line, count=1)
            #print line
            fn.write(line)
        f.close()
        fn.close()
        os.system('pdflatex -output-directory=tables ' + direct + fname + 'Par.tex')

    # make the master parameter table
    template = open('tables/MasterParTableTemplate.tex', 'r')
    final = open('tables/MasterParTable.tex', 'w')

    abbrev = ['B', 'B*', 'C', 'G', 'P', 'S', 'Y', 'Y*']
    for line in template:
        if line[0] == '%':
            varname, fline = line[1:].split('%')
            # remove the \n
            fline = fline[:-1]
            for i, bike in enumerate(data['bikes']):
                if varname == 'bike':
                    fline = fline + ' & \multicolumn{2}{c}{' + abbrev[i] + '}'
                else:
                    val, sig = uround(par[varname][i]).split('+/-')
                    fline = fline + ' & ' + val + ' & ' + sig
            fline = fline + r'\\' + '\n'
            final.write(fline)
        else:
            final.write(line)

    template.close()
    final.close()
    os.system('pdflatex -output-directory=tables ' + direct + 'MasterParTable.tex')


    # make the master canonical matrix table
    template = open('tables/MasterCanTableTemplate.tex', 'r')
    final = open('tables/MasterCanTable.tex', 'w')

    abbrev = ['B', 'B*', 'C', 'G', 'P', 'S', 'Y', 'Y*']
    for line in template:
        if line[0] == '%':
            varname, indice, fline = line[1:].split('%')
            # remove the \n
            fline = fline[:-1]
            for i, bike in enumerate(data['bikes']):
                fcan = open('data/bikeCanonical/' + ''.join(bike.split()) + 'Can.p')
                can = pickle.load(fcan)
                fcan.close()
                if varname == 'bike':
                    fline = fline + ' & \multicolumn{2}{c}{' + abbrev[i] + '}'
                else:
                    try:
                        val, sig = uround(can[varname][indice[0], indice[1]]).split('+/-')
                    except:
                        val = str(can[varname][indice[0], indice[1]])
                        sig = '0.0'
                    fline = fline + ' & ' + val + ' & ' + sig
            fline = fline + r'\\' + '\n'
            final.write(fline)
        else:
            final.write(line)

    template.close()
    final.close()
    os.system('pdflatex -output-directory=tables ' + direct + 'MasterCanTable.tex')

    os.system('rm tables/*.aux')
    os.system('rm tables/*.log')
    os.system('rm tables/*.out')
    os.system('rm tables/*.blg')
    os.system('rm tables/*.bbl')

def bike_bode_plots():

    # load in the base data file
    f = open('data/data.p', 'r')
    data = pickle.load(f)
    f.close()

    nBk = len(data['bikes'])

    colors = ['k', 'r', 'b', 'g', 'y', 'm', 'c', 'orange']

    Tdel2phi = plt.figure(num=1)
    Tdel2del = plt.figure(num=2)
    Tphi2phi = plt.figure(num=3)
    Tphi2del = plt.figure(num=4)

    dir = 'data/bikeRiderCanonical/'
    for i, name in enumerate(data['bikes']):
        fname = ''.join(name.split()) + 'RiderCan.p'
        file = open(dir + fname)
        can = pickle.load(file)
        file.close()
        # make some bode plots
        A, B = abMatrix(can['M'], can['C1'], can['K0'], can['K2'], 1., 9.81)
        # y is [phidot, deldot, phi, del]
        C = np.eye(A.shape[0])
        freq = np.logspace(0, 2, 5000)
        plt.figure(1)
        bode(ABCD=(A, B[:, 1], C[2], 0.), w=freq, fig=Tdel2phi)
        plt.figure(2)
        bode(ABCD=(A, B[:, 1], C[3], 0.), w=freq, fig=Tdel2del)
        plt.figure(3)
        bode(ABCD=(A, B[:, 0], C[2], 0.), w=freq, fig=Tphi2phi)
        plt.figure(4)
        bode(ABCD=(A, B[:, 0], C[3], 0.), w=freq, fig=Tphi2del)
    #for i, line in enumerate(Tdel2phi.ax1.lines):
    #    plt.setp(line, color=colors[i])
    #    plt.setp(Tdel2phi.ax2.lines[i], color=colors[i])
    # plot the bike names on the eigenvalue plot
    plt.show()

def bike_eig_plots():

    # choose the bike alone, bike with rigid legs or bike with rider
    # bike alone
    direct = 'data/bikeCanonical/'
    fend = 'Can.p'
    # bike with rider's legs
    #direct = 'data/bikeLegsCanonical/'
    #fend = 'LegsCan.p'
    # bike with rider
    #direct = 'data/bikeRiderCanonical/'
    #fend = 'RiderCan.p'


    # load in the base data file
    f = open('data/data.p', 'r')
    data = pickle.load(f)
    f.close()

    # add Jodi's thesis bike for comparison (only for bikeCanonical)
    #data['bikes'].append('Jodi Bike')

    nBk = len(data['bikes']) # number of bikes

    # one color for each bike
    colors = ['k', 'r', 'b', 'g', 'y', 'm', 'c', 'orange', 'red']

    vd = np.zeros(nBk)
    vw = np.zeros(nBk)
    vc = np.zeros(nBk)

    # figure properties
    figwidth = 3.25 # in inches
    goldenMean = (np.sqrt(5)-1.0)/2.0
    figsize = [figwidth, figwidth*goldenMean]
    params = {#'backend': 'ps',
        #'axes.labelsize': 10,
        #'text.fontsize': 10,
        'legend.fontsize': 10,
        #'xtick.labelsize': 8,
        #'ytick.labelsize': 8,
        #'text.usetex': True,
        #'figure.figsize': figsize
        }
    plt.rcParams.update(params)
    eigFig = plt.figure(num=nBk + 1)#, figsize=figsize)
    #plt.clf()
    #plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])

    # caluculate eigenvalues/vectors over this range
    vel = np.linspace(0, 10, num=1000)

    # color scale for root loci plot


    for i, name in enumerate(data['bikes']):
        fname = ''.join(name.split()) + fend
        f = open(direct + fname)
        can = pickle.load(f)
        f.close()
        for k, v in can.items():
            can[k] = unumpy.nominal_values(v)
        evals, evecs = bike_eig(can['M'], can['C1'], can['K0'], can['K2'], vel, 9.81)
        wea, cap, cas = sort_modes(evals, evecs)
        vd[i], vw[i], vc[i] = critical_speeds(vel, wea['evals'], cap['evals'])
        # plot individual plot
        plt.figure(i)
        plt.plot(vel, np.abs(np.imag(wea['evals'])), color='blue', label='Imaginary Weave', linestyle='--')
        plt.plot(vel, np.abs(np.imag(cap['evals'])), color='red', label='Imaginary Capsize', linestyle='--')
        plt.plot(vel, np.zeros_like(vel), 'k-', label='_nolegend_', linewidth=2)
        # plot the real parts of the eigenvalues
        plt.plot(vel, np.real(wea['evals']), color='blue', label='Real Weave')
        plt.plot(vel, np.real(cap['evals']), color='red', label='Real Capsize')
        plt.plot(vel, np.real(cas['evals']), color='green', label='Real Caster')
        plt.ylim((-10, 10))
        plt.xlim((0, 10))
        plt.title('{name}\nEigenvalues vs Speed'.format(name=name))
        plt.xlabel('Speed [m/s]')
        plt.ylabel('Real and Imaginary Parts of the Eigenvalue [1/s]')
        plt.savefig('plots/' + ''.join(name.split()) + 'EigPlot.png')
        # plot root loci
        plt.figure(nBk + i)
        for j in range(len(evals[0, :])):
            plt.scatter(evals[:,j].real, evals[:,j].imag, s=2, c=vel,
                    cmap=plt.cm.gist_rainbow, edgecolors='none')
        plt.colorbar()
        plt.grid()
        plt.axis('equal')
        plt.title('{name}\nEigenvalues vs Speed'.format(name=name))
        plt.savefig('plots/' + ''.join(name.split()) + 'RootLoci.png')
        # plot all bikes on the same plot
        plt.figure(2*nBk + 1)
        plt.plot(vel, np.abs(np.imag(wea['evals'])), color=colors[i], label='_nolegend_', linestyle='--')
        plt.plot(vel, np.abs(np.imag(cap['evals'])), color=colors[i], label='_nolegend_', linestyle='--')
        plt.plot(vel, np.zeros_like(vel), 'k-', label='_nolegend_', linewidth=3)
        plt.plot(vel, np.real(wea['evals']), color=colors[i], label='_nolegend_')
        plt.plot(vel, np.real(cap['evals']), color=colors[i], label=data['bikes'][i])
        plt.plot(vel, np.real(cas['evals']), color=colors[i], label='_nolegend_')
    # plot the bike names on the eigenvalue plot
    plt.legend()
    plt.ylim((-10, 10))
    plt.xlim((0, 10))
    plt.title('Eigenvalues vs Speed')
    plt.xlabel('Speed [m/s]')
    plt.ylabel('Real and Imaginary Parts of the Eigenvalue [1/s]')
    try:
        plt.savefig('plots/bike_eig.png')
    except:
        pass
    # make a plot comparing the critical speeds of each bike
    #critFig = plt.figure(num=2)
    #plt.clf()
    #bike = np.arange(len(vd))
    #plt.plot(vd, bike, '|', markersize=50)
    #plt.plot(vc, bike, '|', markersize=50, linewidth=6)
    #plt.plot(vw, bike, '|', markersize=50, linewidth=6)
    #plt.plot(vc - vw, bike)
    #plt.legend([r'$v_d$', r'$v_c$', r'$v_w$', 'stable speed range'])
    #plt.yticks(np.arange(8), tuple(data['bikes']))
    plt.show()

def hunch_angle():

    def uvec(x):
        xhat = x/np.linalg.norm(x)
        return xhat
    # Victor on the Stratos
    # name the four points from the inkscape superimposed lines
    a = np.array([190.716, 1052.483])
    b = np.array([457.633, 533.479])
    c = np.array([-312.107, -371.112])
    d = np.array([645.204, -352.409])
    # calculate the two vectors
    v1 = a - b
    v2 = c -d
    # calculate the angle between the two vectors
    theta = np.arccos(np.dot(uvec(v1), uvec(v2)))
    print "Victor on the Stratos =", np.rad2deg(theta)
    # Victor on the Browser
    # name the four points from the inkscape superimposed lines
    a = np.array([406.081, 989.949])
    b = np.array([512.147, 453.558])
    c = np.array([-334.360, -175.767])
    d = np.array([705.086, -125.259])
    # calculate the two vectors
    v1 = a - b
    v2 = c -d
    # calculate the angle between the two vectors
    theta = np.arccos(np.dot(uvec(v1), uvec(v2)))
    print "Victor on the Browser =", np.rad2deg(theta)

def fit_data():

    dirs, subdirs, filenames = list(os.walk('data/pendDat/p'))[0]
    periodfile = open('data/period.txt', 'w')
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
        x = np.linspace(0, time, num=len(y))
        # decaying oscillating exponential function
        fitfunc = lambda p, t: p[0] + np.exp(-p[3]*p[4]*t)*(p[1]*np.sin(p[4]*np.sqrt(1-p[3]**2)*t) + p[2]*np.cos(p[4]*np.sqrt(1-p[3]**2)*t))
        # initial guesses
        p0 = np.array([1.35, -.5, -.75, 0.01, 3.93])
        # create the error function
        errfunc = lambda p, t, y: fitfunc(p, t) - y
        # minimize the error function
        p1, success = leastsq(errfunc, p0[:], args=(x, y))
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
        f = wd/2./pi
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
        periodfile.write(line)
        print line
        # if the filename is already in the period dictionary...
        if name[:-3] in period.keys():
            # append the period to the list
            period[name[:-3]].append(T)
        # else if the filename isn't in the period dictionary...
        else:
            # start a new list
            period[name[:-3]] = [T]
    periodfile.close()
    f = open('data/period.p', 'w')
    pickle.dump(period, f)
    f.close()

def tor_com():

    f = open('data/period.p', 'r')
    period = pickle.load(f)
    f.close()

    tor = np.zeros((12, 8), dtype='object')
    com = np.zeros((4, 8), dtype='object')
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
        km = re.sub('BrowserIns', 'Browserins', k)
        km = re.sub('YellowRev', 'Yellowrev', km)
        desc = space_out_camel_case(km).split()
        print desc
        if desc[0] == 'Rod':
            rodPeriod = np.mean(v)
        else:
            # if torsional, put it in the torsional matrix
            if desc[2] == tc[0]:
                r = fffr.index(desc[1])*3 + fst.index(desc[3])
                c = bN.index(desc[0])
                tor[r, c] = np.mean(v)
                print 'Added torsional to [', r, ',', c, ']'
            # if compound, put in in the compound matrix
            elif desc[2] == tc[1]:
                com[fffr.index(desc[1]), bN.index(desc[0])] = np.mean(v)
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
    par['c'] = (par['rF']*unumpy.sin(par['lambda'])
                  - forkOffset)/unumpy.cos(par['lambda'])

    # wheelbase
    par['w'] = ddU['wheelbase']

    # calculate the dees
    par['d1'] = unumpy.cos(par['lambda'])*(par['c']+par['w']-par['rR']*unumpy.tan(par['lambda']))
    par['d3'] = -unumpy.cos(par['lambda'])*(par['c']-par['rF']*unumpy.tan(par['lambda']))

    # calculate the frame rotation angle
    # alpha is the angle between the negative z pendulum (horizontal) and the
    # positive (up) steer axis, rotation about positive y
    alphaFrame = ddU['frameAngle']
    # beta is the angle between the x bike frame and the x pendulum frame, rotation
    # about positive y
    betaFrame = par['lambda'] - alphaFrame*pi/180

    # calculate the slope of the CoM line
    frameM = -unumpy.tan(betaFrame)

    # calculate the z-intercept of the CoM line
    # frameMassDist is positive according to the pendulum ref frame
    frameMassDist = ddU['frameMassDist']
    cb = unumpy.cos(betaFrame)
    frameB = -frameMassDist/cb - par['rR']

    # calculate the fork rotation angle
    betaFork = par['lambda'] - ddU['forkAngle']*pi/180.

    # calculate the slope of the fork CoM line
    forkM = -unumpy.tan(betaFork)

    # calculate the z-intercept of the CoM line
    forkMassDist = ddU['forkMassDist']
    cb = unumpy.cos(betaFork)
    tb = unumpy.tan(betaFork)
    forkB = - par['rF'] - forkMassDist/cb + par['w']*tb

    # plot the CoM lines
    comFig = plt.figure(num=1)
    # intialize the matrices for the center of mass locations
    frameCoM = np.zeros((2, np.shape(frameM)[1]), dtype='object')
    forkCoM = np.zeros((2, np.shape(forkM)[1]), dtype='object')
    # for each of the bikes...
    for i in range(np.shape(frameM)[1]):
        comb = np.array([[0, 1], [0, 2], [1, 2]])
        # calculate the frame center of mass position
        # initialize the matrix to store the line intersections
        lineX = np.zeros((3, 2), dtype='object')
        # for each line intersection...
        for j, row in enumerate(comb):
            a = unumpy.matrix(np.vstack([-frameM[row, i], np.ones((2))]).T)
            b = frameB[row, i]
            lineX[j] = np.dot(a.I, b)
        frameCoM[:, i] = np.mean(lineX, axis=0)
        # calculate the fork center of mass position
        # reinitialize the matrix to store the line intersections
        lineX = np.zeros((3, 2), dtype='object')
        # for each line intersection...
        for j, row in enumerate(comb):
            a = unumpy.matrix(np.vstack([-forkM[row, i], np.ones((2))]).T)
            b = forkB[row, i]
            lineX[j] = np.dot(a.I, b)
        forkCoM[:, i] = np.mean(lineX, axis=0)
        # make a subplot for this bike
        plt.subplot(2, 4, i + 1)
        # plot the rear wheel
        c = plt.Circle((0, par['rR'][i].nominal_value), radius=par['rR'][i].nominal_value)
        plt.gca().add_patch(c)
        # plot the front wheel
        c = plt.Circle((par['w'][i].nominal_value, par['rF'][i].nominal_value), radius=par['rF'][i].nominal_value)
        plt.gca().add_patch(c)
        # plot the lines (pendulum axes)
        x = np.linspace(-par['rR'][i].nominal_value, par['w'][i].nominal_value + par['rF'][i].nominal_value, 2)
        # for each line...
        for j in range(len(frameM)):
            framey = -frameM[j, i].nominal_value*x - frameB[j, i].nominal_value
            forky = -forkM[j, i].nominal_value*x - forkB[j, i].nominal_value
            plt.plot(x,framey, 'r')
            plt.plot(x,forky, 'g')
        # plot the ground line
        plt.plot(x, np.zeros_like(x), 'k')
        # plot the fundamental bike
        deex = np.zeros(4)
        deez = np.zeros(4)
        deex[0] = 0.
        deex[1] = (par['d1'][i]*unumpy.cos(par['lambda'][i])).nominal_value
        deex[2] = (par['w'][i]-par['d3'][i]*unumpy.cos(par['lambda'][i])).nominal_value
        deex[3] = par['w'][i].nominal_value
        deez[0] = -par['rR'][i].nominal_value
        deez[1] = -(par['rR'][i]+par['d1'][i]*unumpy.sin(par['lambda'][i])).nominal_value
        deez[2] = -(par['rF'][i]-par['d3'][i]*unumpy.sin(par['lambda'][i])).nominal_value
        deez[3] = -par['rF'][i].nominal_value
        plt.plot(deex, -deez, 'k')
        # plot the centers of mass
        plt.plot(frameCoM[0, i].nominal_value, -frameCoM[1, i].nominal_value, 'k+', markersize=12)
        plt.plot(forkCoM[0, i].nominal_value, -forkCoM[1, i].nominal_value, 'k+', markersize=12)
        plt.axis('equal')
        plt.ylim((0, 1))
        plt.title(bikeNames[i])
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
    par['g'] = 9.81*np.ones(ddU['forkMass'].shape, dtype=float)
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
    par['IHxx'] = np.array(par['IHxx'])
    par['IHxz'] = np.array(par['IHxz'])
    par['IHzz'] = np.array(par['IHzz'])

    # calculate the frame in-plane moments of inertia
    par['IBxx'] = []
    par['IBxz'] = []
    par['IBzz'] = []
    for i, row in enumerate(Ipend[:3, :].T):
        Imat = inertia_components_uncert(row, betaFrame[:, i])
        par['IBxx'].append(Imat[0, 0])
        par['IBxz'].append(Imat[0, 1])
        par['IBzz'].append(Imat[1, 1])
    par['IBxx'] = np.array(par['IBxx'])
    par['IBxz'] = np.array(par['IBxz'])
    par['IBzz'] = np.array(par['IBzz'])
    par['v'] = np.ones_like(par['g'])

    # make a dictionary with only the nominal values
    par_n = {}
    for k, v in par.items():
        if type(v[0]) == type(par['rF'][0]) or type(v[0]) == type(par['mF'][0]):
            par_n[k] = unumpy.nominal_values(v)
        else:
            par_n[k] = par[k]

    # plot all the parameters to look for crazy numbers
    parFig = plt.figure(num=2)
    i = 1
    xt = ['B', 'BI', 'C', 'F', 'P', 'S', 'Y', 'YR']
    for k, v in par_n.items():
        plt.subplot(5, 6, i)
        plt.plot(v, '-D', markersize=14)
        plt.title(k)
        plt.xticks(np.arange(8), tuple(xt))
        i += 1
    plt.show()

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

def calc_canon():
    '''
    Calculates the A, B, M, C1, K0, and K2 matrices and saves them to pickled
    dictionaries and text files.

    '''

    # load in the base data file
    f = open('data/data.p', 'r')
    data = pickle.load(f)
    f.close()

    data['bikes'].append('Jodi Bike')

    # write the parameter files
    for i, name in enumerate(data['bikes']):
        directory = 'data/bikeParameters/'
        fname = ''.join(name.split())
        M, C1, K0, K2, param = bmp2cm(directory + fname + 'Par.txt')
        A, B = abMatrix(M, C1, K0, K2, param['v'], param['g'])
        directory = 'data/bikeCanonical/'
        f = open(directory + fname + 'Can.txt', 'w')
        for mat in ['M','C1', 'K0', 'K2', 'A', 'B']:
            if mat == 'A' or mat == 'B':
                f.write(mat + ' (v = ' + str(param['v']) + ')\n')
            else:
                f.write(mat + '\n')
            f.write(str(eval(mat)) + '\n')
        f.close()
        f = open(directory + fname + 'Can.p', 'w')
        canon = {'M':M, 'C1':C1, 'K0':K0, 'K2':K2, 'A':A, 'B':B, 'v':param['v']}
        pickle.dump(canon, f)
        f.close()
        # build some pretty tex files
        replace_values(directory, 'CanonTemplate.tex', fname + 'Can.tex', canon)

def calc_jason_on_bikes():

    # load in the base data file
    f = open('data/data.p', 'r')
    data = pickle.load(f)
    f.close()

    # load in the parameter file
    f = open('data/par.p', 'r')
    par = pickle.load(f)
    f.close()

    nBk = len(data['bikes'])

    # remove the uncertainties
    par_n = {}
    for k, v in par.items():
        if type(v[0]) == type(par['rF'][0]) or type(v[0]) == type(par['mF'][0]):
            par_n[k] = unumpy.nominal_values(v)
        else:
            par_n[k] = par[k]

    # Jason's parameters (sitting on the Batavus Browser)
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

    f = open('data/parWithRider.p', 'w')
    pickle.dump(par_n, f)
    f.close()

    speeds = [2.5, 4., 5., 5.8, 7.5, 12.0]

    # write the parameter files
    for i, name in enumerate(data['bikes']):
        direct = 'data/bikeRiderParameters/'
        fname = ''.join(name.split())
        f = open(direct + fname  + 'RiderPar.txt', 'w')
        for k, v in par_n.items():
            line = k + ',' + str(v[i]) + '\n'
            f.write(line)
        f.close()
        M, C1, K0, K2, param = bmp2cm(direct + fname + 'RiderPar.txt')
        direct = 'data/bikeRiderCanonical/'
        f = open(direct + fname + 'RiderCan.txt', 'w')
        for mat in ['M','C1', 'K0', 'K2']:
            f.write(mat + '\n')
            f.write(str(eval(mat)) + '\n')
        f.write("The states are [roll rate, steer rate, roll angle, steer angle]\n")
        for v in speeds:
            A, B = abMatrix(M, C1, K0, K2, v, param['g'])
            for mat in ['A', 'B']:
                f.write(mat + ' (v = ' + str(v) + ')\n')
                f.write(str(eval(mat)) + '\n')
        f.close()
        mdict = {'M':M, 'C1':C1, 'K0':K0, 'K2':K2}
        f = open(direct + fname + 'RiderCan.p', 'w')
        pickle.dump(mdict, f)
        f.close()
        savemat(direct + fname + 'RiderCan.mat', mdict)

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
    M = np.linalg.inv(B[numRow/2:, :])
    C = -np.dot(M, A[numRow/2:, numCol/2:])
    K = -np.dot(M, A[numRow/2:, :numCol/2])

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
    sA = unumpy.std_devs(uA)

    # create the covariance matrix for A
    CA = np.diag(sA.flatten())

    # pull out the nominal A
    A = unumpy.nominal_values(uA)

    # the nominal eigenvalues and eigenvectors
    w, v = np.linalg.eig(A)
    print 'w=', w

    # FA is the jacobian
    FAw = np.zeros((w.shape[0], A.flatten().shape[0]))
    FAv = np.zeros((v.flatten().shape[0], A.flatten().shape[0]))

    # pw is the perturbed eigenvectors used in the FA calc
    pw = np.zeros((w.shape[0], A.flatten().shape[0]), dtype=complex)
    pv = np.zeros((w.shape[0]**2, A.flatten().shape[0]), dtype=complex)

    # calculate the perturbed eigenvalues for each A entry
    for i, a in enumerate(A.flatten()):
        # set the differentiation step
        if a == 0.:
            delta = 1e-8
        else:
            delta = np.sqrt(np.finfo(float).eps)*a

        # make a copy of A
        pA = copy(A).flatten()

        # perturb the entry
        pA[i] = pA[i] + delta

        # back to matrix
        pA = np.hsplit(pA, A.shape[0])
        print 'A', A
        print 'pA', pA

        # calculate the eigenvalues
        print np.linalg.eig(pA)[0]
        pw[:, i], tpv = np.linalg.eig(pA)
        print 'perturbed eig', pw[:, i]
        print 'nom eig', w
        print 'delta', delta
        FAw[:, i] = (pw[:, i] - w)/delta
        print "FAw column=", FAw[:, i]
        pv[:, i] = tpv.flatten('F')
        FAv[:, i] = (pv[:, i] - v.flatten('F'))/delta

    # calculate the covariance matrix for the eigenvalues
    Cw = np.dot(np.dot(FAw, CA), FAw.T)
    Cv = np.dot(np.dot(FAv, CA), FAv.T)
    print FAw

    # build the eigenvalues with uncertainties
    uw = unumpy.uarray((w, np.diag(Cw)))
    uv = unumpy.uarray((v.flatten('F'), np.diag(Cv)))
    uv = np.vstack(np.hsplit(uv, w.shape[0])).T
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
        test = re.findall('\|(\w*.)\|', line)
        print 'these are all the matches', test
        print 'search for nom: ', re.findall('\|(\w*)(?!\?)\|', line)
        print 'search for un: ',  re.findall('\|(\w*\?)\|', line)
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
                            line = re.sub('\|(\w*\?)\|', replacement, line, count=1)
                        except: # there is no uncertainty
                            replacement = '0.0'
                            print "with this:", replacement
                            line = re.sub('\|(\w*\?)\|', replacement, line, count=1)
                    else:
                        try:
                            replacement = uround(value).split('+/-')[0]
                            print "with this:", replacement
                            line = re.sub('\|(\w*(?!\?))\|', replacement, line, count=1)
                        except: # there is no uncertainty
                            replacement = str(value)
                            print "with this:", replacement
                            line = re.sub('\|(\w*(?!\?))\|', replacement, line, count=1)
                    del var, row, col
                # else the match is just a scalar
                else:
                    # if the match has a question mark it is an uncertainty value
                    if match[-1] == '?':
                        try:
                            line = re.sub('\|(\w*\?)\|',
                                    uround(replacers[match[:-1]]).split('+/-')[1], line, count=1)
                            print line
                        except:
                            pass
                    else:
                        try:
                            line = re.sub('\|(\w*(?!\?))\|',
                                    uround(replacers[match]).split('+/-')[0], line, count=1)
                            print line
                        except:
                            line = re.sub('\|(\w*(?!\?))\|', str(replacers[match][i]), line, count=1)
        print line
        fn.write(line)
    f.close()
    fn.close()
    os.system('pdflatex -output-directory=' + directory + ' ' + newfile)
    os.system('rm ' + directory + '*.aux')
    os.system('rm ' + directory + '*.log')

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
        fig = plt.figure(2)
    ax1 = plt.axes([0.1, 0.1, 0.8, 0.7])
    ax1.plot(t, ym, '.', markersize=2)
    plt.plot(t, yf, 'k-')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [V]')
    equation = r'$f(t)={0:1.2f}+e^{{-({3:1.3f})({4:1.1f})t}}\left[{1:1.2f}\sin{{\sqrt{{1-{3:1.3f}^2}}{4:1.1f}t}}+{2:1.2f}\cos{{\sqrt{{1-{3:1.3f}^2}}{4:1.1f}t}}\right]$'.format(p[0], p[1], p[2], p[3], p[4])
    rsquare = '$r^2={0:1.3f}$'.format(rsq)
    period = '$T={0} s$'.format(T)
    plt.title(equation + '\n' + rsquare + ', ' + period)
    plt.legend(['Measured', 'Fit'])
    #plt.xlim((0, 1))
    return fig

def space_out_camel_case(s):
        """Adds spaces to a camel case string.  Failure to space out string
        returns the original string.
        >>> space_out_camel_case('DMLSServicesOtherBSTextLLC')
        'DMLS Services Other BS Text LLC'
        """
        return re.sub('((?=[A-Z][a-z])|(?<=[a-z])(?=[A-Z]))', ' ', s).strip()

def bode(ABCD=None, numden=None, w=None, fig=None, n=None, label=None,
        title=None, color=None):
    """Bode plot.

    Takes the system A, B, C, D matrices of the state space system.

    Need to implement transfer function num/den functionality.

    Returns magnitude and phase vectors, and figure object.
    """
    if fig == None:
        fig = plt.figure()

    mag = np.zeros(len(w))
    phase = np.zeros(len(w))
    fig.yprops = dict(rotation=90,
                  horizontalalignment='right',
                  verticalalignment='center',
                  x=-0.01)

    fig.axprops = {}
    fig.ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4], **fig.axprops)
    fig.ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4], **fig.axprops)

    if (ABCD):
        A, B, C, D = ABCD
        I = np.eye(A.shape[0])
        for i, f in enumerate(w):
            sImA_inv = np.linalg.inv(1j*f*I - A)
            G = np.dot(np.dot(C, sImA_inv), B) + D
            mag[i] = 20.*np.log10(np.abs(G))
            phase[i] = np.arctan2(np.imag(G), np.real(G))
        phase = 180./np.pi*np.unwrap(phase)
    elif (numden):
        n = np.poly1d(numden[0])
        d = np.poly1d(numden[1])
        Gjw = n(1j*w)/d(1j*w)
        mag = 20.*np.log10(np.abs(Gjw))
        phase = 180./pi*np.unwrap(np.arctan2(np.imag(Gjw), np.real(Gjw)))

    fig.ax1.semilogx(w, mag, label=label)
    if title:
        fig.ax1.set_title(title)
    fig.ax2.semilogx(w, phase, label=label)


    fig.axprops['sharex'] = fig.axprops['sharey'] = fig.ax1
    fig.ax1.grid(b=True)
    fig.ax2.grid(b=True)

    plt.setp(fig.ax1.get_xticklabels(), visible=False)
    plt.setp(fig.ax1.get_yticklabels(), visible=True)
    plt.setp(fig.ax2.get_yticklabels(), visible=True)
    fig.ax1.set_ylabel('Magnitude [dB]', **fig.yprops)
    fig.ax2.set_ylabel('Phase [deg]', **fig.yprops)
    fig.ax2.set_xlabel('Frequency [rad/s]')
    if label:
        fig.ax1.legend()

    if color:
        plt.setp(fig.ax1.lines, color=color)
        plt.setp(fig.ax2.lines, color=color)

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
    capsize and caster). Some type of check unsing the derivative of the curves
    could make it more robust.
    '''
    evalsorg = np.zeros_like(evals)
    evecsorg = np.zeros_like(evecs)
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
                x, y = np.real(evalsorg[i, j].nominal_value), np.imag(evalsorg[i, j].nominal_value)
            except:
                x, y = np.real(evalsorg[i, j]), np.imag(evalsorg[i, j])
            # for each eigenvalue at the next speed
            dist = np.zeros(4)
            for k, eignext in enumerate(evals[i + 1]):
                try:
                    xn, yn = np.real(eignext.nominal_value), np.imag(eignext.nominal_value)
                except:
                    xn, yn = np.real(eignext), np.imag(eignext)
                # distance between points in the real/imag plane
                dist[k] = np.abs(((xn - x)**2 + (yn - y)**2)**0.5)
            if np.argmin(dist) in used:
                # set the already used indice higher
                dist[np.argmin(dist)] = np.max(dist) + 1.
            else:
                pass
            evalsorg[i + 1, j] = evals[i + 1, np.argmin(dist)]
            evecsorg[i + 1, :, j] = evecs[i + 1, :, np.argmin(dist)]
            # keep track of the indices we've used
            used.append(np.argmin(dist))
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
    vw = v[np.argmin(np.abs(np.real(weave[:, 0])))]
    vc = v[np.argmin(np.abs(np.real(capsize)))]
    m = np.max(np.abs(np.imag(weave[:, 0])))
    w = np.zeros_like(np.imag(weave[:, 0]))
    for i, eig in enumerate(np.abs(np.imag(weave[:, 0]))):
        if eig == 0.:
            w[i] = m + 1.
        else:
            w[i] = eig
    vd = v[np.argmin(w)]
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
    SSR = np.sum((yp - np.mean(ym))**2)
    SST = np.sum((ym - np.mean(ym))**2)
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
    evals = np.zeros((n, m), dtype='complex128')
    evecs = np.zeros((n, m, m), dtype='complex128')
    for i, speed in enumerate(v):
        A, B = abMatrix(M, C1, K0, K2, speed, g)
        w, vec = np.linalg.eig(unumpy.nominal_values(A))
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
    uA = (xA - p['w'] - p['c'])*umath.cos(p['lambda']) - zA*umath.sin(p['lambda'])
    IAll = mA*uA**2 + IAxx*umath.sin(p['lambda'])**2 + 2*IAxz*umath.sin(p['lambda'])*umath.cos(p['lambda']) + IAzz*umath.cos(p['lambda'])**2
    IAlx = -mA*uA*zA + IAxx*umath.sin(p['lambda']) + IAxz*umath.cos(p['lambda'])
    IAlz = mA*uA*xA + IAxz*umath.sin(p['lambda']) + IAzz*umath.cos(p['lambda'])
    mu = p['c']/p['w']*umath.cos(p['lambda'])
    SR = p['IRyy']/p['rR']
    SF = p['IFyy']/p['rF']
    ST = SR + SF
    SA = mA*uA + mu*mT*xT
    Mpp = ITxx
    Mpd = IAlx + mu*ITxz
    Mdp = Mpd
    Mdd = IAll + 2*mu*IAlz + mu**2*ITzz
    M = np.array([[Mpp, Mpd], [Mdp, Mdd]])
    K0pp = mT*zT # this value only reports to 13 digit precision it seems?
    K0pd = -SA
    K0dp = K0pd
    K0dd = -SA*umath.sin(p['lambda'])
    K0 = np.array([[K0pp, K0pd], [K0dp, K0dd]])
    K2pp = 0.
    K2pd = (ST - mT*zT)/p['w']*umath.cos(p['lambda'])
    K2dp = 0.
    K2dd = (SA + SF*umath.sin(p['lambda']))/p['w']*umath.cos(p['lambda'])
    K2 = np.array([[K2pp, K2pd], [K2dp, K2dd]])
    C1pp = 0.
    C1pd = mu*ST + SF*umath.cos(p['lambda']) + ITxz/p['w']*umath.cos(p['lambda']) - mu*mT*zT
    C1dp = -(mu*ST + SF*umath.cos(p['lambda']))
    C1dd = IAlz/p['w']*umath.cos(p['lambda']) + mu*(SA + ITzz/p['w']*umath.cos(p['lambda']))
    C1 = np.array([[C1pp, C1pd], [C1dp, C1dd]])
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
    a21 = np.eye(2)
    a22 = np.zeros((2, 2))
    A = np.vstack((np.dot(unumpy.ulinalg.inv(M), np.hstack((a11, a12))), np.hstack((a21, a22))))
    B = np.vstack((unumpy.ulinalg.inv(M), np.zeros((2, 2))))
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
    return np.array([Ix, Iy, Iz])

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
    sa = np.sin(alpha)
    ca = np.cos(alpha)
    A = np.vstack((ca**2, -2*sa*ca, sa**2)).T
    Iorth = np.linalg.lstsq(A, I)[0]
    Inew = np.array([[Iorth[0], Iorth[1]], [Iorth[1], Iorth[2]]])
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
    sa = unumpy.sin(alpha)
    ca = unumpy.cos(alpha)
    A = unumpy.matrix(np.vstack((ca**2, -2*sa*ca, sa**2)).T)
    Iorth = np.dot(A.I, I)
    Iorth = np.array([Iorth[0, 0], Iorth[0, 1], Iorth[0, 2]], dtype='object')
    Inew = np.array([[Iorth[0], Iorth[1]], [Iorth[1], Iorth[2]]])
    return Inew

def parallel_axis(Ic, m, d):
    '''Parallel axis thereom. Takes the moment of inertia about the rigid
    body's center of mass and translates it to a new reference frame that is
    the distance, d, from the center of mass.'''
    a = d[0]
    b = d[1]
    c = d[2]
    dMat = np.zeros((3, 3))
    dMat[0] = np.array([b**2 + c**2, -a*b, -a*c])
    dMat[1] = np.array([-a*b, c**2 + a**2, -b*c])
    dMat[2] = np.array([-a*c, -b*c, a**2 + b**2])
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
    c = (rF*np.sin(lam) - fo)/np.cos(lam)
    # mechanical trail
    cm = c*np.cos(lam)
    return c, cm
