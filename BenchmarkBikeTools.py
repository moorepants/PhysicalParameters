def bmp2cm(filename):
    ''' Converts the benchmark bicycle parameters to the canonical matrices'''
    import numpy as np
    f = open(filename, 'r')
    p = {}
    for i, line in enumerate(f):
        list = line[:-1].split(',')
        pi = np.pi
        p[list[0]] = eval(list[1])
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
    uA = (xA - p['w'] - p['c'])*np.cos(p['lambda']) - zA*np.sin(p['lambda'])
    IAll = mA*uA**2 + IAxx*np.sin(p['lambda'])**2 + 2*IAxz*np.sin(p['lambda'])*np.cos(p['lambda']) + IAzz*np.cos(p['lambda'])**2
    IAlx = -mA*uA*zA + IAxx*np.sin(p['lambda']) + IAxz*np.cos(p['lambda'])
    IAlz = mA*uA*xA + IAxz*np.sin(p['lambda']) + IAzz*np.cos(p['lambda'])
    mu = p['c']/p['w']*np.cos(p['lambda'])
    SR = p['IRyy']/p['rR']
    SF = p['IFyy']/p['rF']
    ST = SR + SF
    SA = mA*uA + mu*mT*xT
    Mpp = ITxx
    Mpd = IAlx + mu*ITxz
    Mdp = Mpd
    Mdd = IAll + 2*mu*IAlz + mu**2*ITzz
    M = np.array([[Mpp, Mpd], [Mdp, Mdd]])
    K0pp = mT*zT # this value only reports to 13 digit precision?
    K0pd = -SA
    K0dp = K0pd
    K0dd = -SA*np.sin(p['lambda'])
    K0 = np.array([[K0pp, K0pd], [K0dp, K0dd]])
    K2pp = 0
    K2pd = (ST - mT*zT)/p['w']*np.cos(p['lambda'])
    K2dp = 0
    K2dd = (SA + SF*np.sin(p['lambda']))/p['w']*np.cos(p['lambda'])
    K2 = np.array([[K2pp, K2pd], [K2dp, K2dd]])
    C1pp = 0
    C1pd = mu*ST + SF*np.cos(p['lambda']) + ITxz/p['w']*np.cos(p['lambda']) - mu*mT*zT
    C1dp = -(mu*ST + SF*np.cos(p['lambda']))
    C1dd = IAlz/p['w']*np.cos(p['lambda']) + mu*(SA + ITzz/p['w']*np.cos(p['lambda']))
    C1 = np.array([[C1pp, C1pd], [C1dp, C1dd]])
    return M, K0, K2, C1, p

def aMatrix(M, K0, K2, C1, p):
    '''Calculates the A matrix from the canonical matrices for the benchmark
    bicycle'''
    from numpy import eye, zeros, vstack, hstack, dot
    from numpy.linalg import inv
    a11 = -p['v']*C1
    a12 = -(p['g']*K0 + p['v']**2*K2)
    a21 = eye(2)
    a22 = zeros((2, 2))
    A = vstack((dot(inv(M), hstack((a11, a12))), hstack((a21, a22))))
    return A

def tor_inertia(k, T):
    '''Calculates the moment of interia for an ideal torsional pendulm'''
    from math import pi
    I = k*T**2/4./pi**2
    return I

def com_inertia(m, g, l, T):
    '''Calculates the moment of inertia for an object hung from a compound
    pendulum'''
    from math import pi
    I = (T/2./pi)**2*m*g*l - m*l**2
    return I

def tube_inertia(l, m, ro, ri):
    '''Calculates the moment of inertia for a tube (or rod) where the x axis is
    aligned with the tube's axis'''
    from numpy import array
    Ix = m/2.*(ro**2 + ri**2)
    Iy = m/12.*(3*ro**2 + 3*ri**2 + l**2) 
    Iz = Iy
    return array([Ix, Iy, Iz])

def tor_stiffness(I, T):
    '''Calculates the stiffness of a torsional pendulum with a known moment of
    inertia'''
    from math import pi
    k = 4.*I*pi**2/T**2
    return k

def inertia_components(I, alpha):
    '''Calculates the 2D orthongonal inertia tensor when at least three moments
    of inertia and their axis orientations are specified'''
    from numpy import sin, cos, vstack, array
    from numpy.linalg import lstsq
    sa = sin(alpha)
    ca = cos(alpha)
    A = vstack((ca**2, 2*sa*ca, sa**2)).T
    Iorth = lstsq(A, I)[0]
    Inew = array([[Iorth[0], -Iorth[1]], [-Iorth[1], Iorth[2]]])
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
