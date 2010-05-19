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

    Input:
        filename is a text file with the 27 parameters listed in csv format. One
        parameter per line in the form: 'lamba,10/pi' or 'v,5'. Use the
        benchmark paper's units!
    Output:
        M is the mass matrix
        C1 is the damping like matrix that is proportional to v
        K0 is the stiffness matrix proportional to gravity
        K2 is the stiffness matrix proportional to v**2
        p is a dictionary of the parameters. the keys are the variable names

        '''
    from numpy import pi, cos, sin, array
    f = open(filename, 'r')
    p = {}
    # parse the text file
    for i, line in enumerate(f):
        list1 = line[:-2].split(',')
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
    K2pp = 0
    K2pd = (ST - mT*zT)/p['w']*cos(p['lambda'])
    K2dp = 0
    K2dd = (SA + SF*sin(p['lambda']))/p['w']*cos(p['lambda'])
    K2 = array([[K2pp, K2pd], [K2dp, K2dd]])
    C1pp = 0
    C1pd = mu*ST + SF*cos(p['lambda']) + ITxz/p['w']*cos(p['lambda']) - mu*mT*zT
    C1dp = -(mu*ST + SF*cos(p['lambda']))
    C1dd = IAlz/p['w']*cos(p['lambda']) + mu*(SA + ITzz/p['w']*cos(p['lambda']))
    C1 = array([[C1pp, C1pd], [C1dp, C1dd]])
    return M, C1, K0, K2, p

def abMatrix(M, C1, K0, K2, v, g):
    '''Calculate the A and B matrices for the benchmark bicycle

    Input:
        M is the mass matrix
        C1 is the damping like matrix that is proportional to v
        K0 is the stiffness matrix proportional to gravity
        K2 is the stiffness matrix proportional to v**2
        v : speed
        g : acceleration due to gravity
    Returns:
        A system dynamic matrix
        B control matrix
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
    from numpy import sin, cos, vstack, array, matrix, dot, zeros_like
    from numpy.linalg import lstsq
    from uncertainties import umath
    sa = zeros_like(alpha)
    ca = zeros_like(alpha)
    for i in range(len(alpha)):
        sa[i] = umath.sin(alpha[i])
        ca[i] = umath.cos(alpha[i])
    A = matrix(vstack((ca**2, 2*sa*ca, sa**2)).T)
    Iorth = dot(dot(dot(A.T, A).I, A.T), I)
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
