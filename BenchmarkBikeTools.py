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
    import numpy as np
    a11 = -p['v']*C1
    a12 = -(p['g']*K0 + p['v']**2*K2)
    a21 = np.eye(2)
    a22 = np.zeros((2, 2))
    A = np.vstack((np.dot(np.linalg.inv(M), np.hstack((a11, a12))), np.hstack((a21, a22))))
    return A
