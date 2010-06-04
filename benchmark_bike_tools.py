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
    import re
    import os

    from benchmark_bike_tools import uround

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
            print 'Found this!\n', test
            # go through each match and make a substitution
            for match in test:
                print "replace this: ", match
                # if there is a '_' then the match is an array entry
                if '_' in match:
                    ques = None
                    try:
                        var, row, col, ques = match.split('_') # this returns the variable
                    except:
                        var, row, col = match.split('_') # this returns the variable

                    # if the match has a question mark it is an uncertainty value
                    if ques:
                        try:
                            line = re.sub('\|(\w*\?)\|',
                                    uround(eval('replacers[' + var + ']' + loc[:-1])).split('+/-')[1], line, count=1)
                            print line
                        except: # there is no uncertainty
                            pass
                    else:
                        try:
                            line = re.sub('\|(\w*(?!\?))\|',
                                    uround(eval('replacers[' + var + ']' + loc)).split('+/-')[0], line, count=1)
                            print line
                        except: # there is no uncertainty
                            line = re.sub('\|(\w*(?!\?))\|', str(eval('replacers[' + var + ']' + loc)), line, count=1)
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
    # os.system('pdflatex -output-directory=' + directory + ' ' + newfile)
    # os.system('rm ' + directory + '*.aux')
    # os.system('rm ' + directory + '*.log')

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
    from uncertainties import ufloat
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
    from matplotlib.pyplot import figure, plot, xlabel, ylabel, title, legend, axes, xlim
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
        import re
        return re.sub('((?=[A-Z][a-z])|(?<=[a-z])(?=[A-Z]))', ' ', s)

def bode(ABCD=None, numden=None, w=None, fig=None, n=None, label=None,
        title=None, color=None):
    """Bode plot.

    Takes the system A, B, C, D matrices of the state space system.

    Need to implement transfer function num/den functionality.

    Returns magnitude and phase vectors, and figure object.
    """
    import numpy as np
    import scipy as sp
    import matplotlib.pyplot as plt
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
        phase = 180./np.pi*np.unwrap(np.arctan2(np.imag(Gjw), np.real(Gjw)))

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
    capsize and caster). Some type of check using the derivative of the curves
    could make it more robust.
    '''
    from numpy import abs, zeros_like, imag, real, argmin, sqrt, zeros
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
        for j, eig in enumerate(speed):
            x, y = real(evalsorg[i, j]), imag(evalsorg[i, j])
            # for each eigenvalue at the next speed
            dist = zeros(4)
            for k, eignext in enumerate(evals[i + 1]):
                xn, yn = real(eignext), imag(eignext)
                # distance between points in the real/imag plane
                dist[k] = abs(sqrt((xn - x)**2 + (yn - y)**2))
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
    from numpy import argmin, abs, real, imag, zeros_like, max
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
    t : time vector

    Returns:
    --------
    jac : The jacobian, the partial of the vector function with respect to the
    parameters vector. A 5 x N matrix where N is the number of time steps.

    '''
    from numpy import zeros, exp, sqrt, sin, cos, ones_like
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
    from numpy.linalg import eig
    from numpy import zeros
    m, n = 2*M.shape[0], v.shape[0]
    evals = zeros((n, m), dtype='complex128')
    evecs = zeros((n, m, m), dtype='complex128')
    for i, speed in enumerate(v):
        A, B = abMatrix(M, C1, K0, K2, speed, g)
        w, vec = eig(A)
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
    from numpy import pi, array
    from uncertainties import ufloat
    from uncertainties.umath import sin, cos
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
    from numpy import eye, zeros, vstack, hstack, dot
    from numpy.linalg import inv
    a11 = -v*C1
    a12 = -(g*K0 + v**2*K2)
    a21 = eye(2)
    a22 = zeros((2, 2))
    A = vstack((dot(inv(M), hstack((a11, a12))), hstack((a21, a22))))
    B = vstack((inv(M), zeros((2, 2))))
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
    from math import pi

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
    from math import pi

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
    from numpy import array
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
    from math import pi
    k = 4.*I*pi**2/T**2
    return k

def inertia_components(I, alpha):
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
    from numpy import sin, cos, vstack, array
    from numpy.linalg import lstsq
    sa = sin(alpha)
    ca = cos(alpha)
    A = vstack((ca**2, 2*sa*ca, sa**2)).T
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
    from numpy import vstack, array, dot
    from uncertainties import unumpy
    sa = unumpy.sin(alpha)
    ca = unumpy.cos(alpha)
    A = unumpy.matrix(vstack((ca**2, 2*sa*ca, sa**2)).T)
    Iorth = dot(A.I, I)
    Iorth = array([Iorth[0, 0], Iorth[0, 1], Iorth[0, 2]], dtype='object')
    Inew = array([[Iorth[0], Iorth[1]], [Iorth[1], Iorth[2]]])
    return Inew

def parallel_axis(Ic, m, d):
    '''Parallel axis thereom. Takes the moment of inertia about the rigid
    body's center of mass and translates it to a new reference frame that is
    the distance, d, from the center of mass.'''
    from numpy import array, zeros
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
        The steer axis tilt (pi/2 - headtube angle)
    fo: float
        The fork offset

    Returns:
    --------
    c: float
        Trail
    c: float
        Mechanical Trail

    '''
    from math import sin, cos

    c = (rF*sin(lam) - fo)/cos(lam)
    cm = c*cos(lam)
    return c, cm
