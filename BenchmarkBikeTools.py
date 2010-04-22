def bmp2cm(filename):
    ''' Converts the benchmark bicycle parameters to the canonical matrices'''
    import numpy as np
    f = open(filename, 'r')
    p = {}
    p['var'] = []
    p['val'] = []
    for i, line in enumerate(f):
        list = line[:-1].split(',')
        pi = np.pi
        p[list[0]] = eval(list[1])
        #p['var'].append(list[0])
        #try:
        #    p['val'].append(float(list[1]))
        #except ValueError:
        #    pi = np.pi
        #    p['val'].append(eval(list[1]))
    w = p['w']
    c = p['c']
    l = p['lambda']
    g = p['g']
    v = p['v']
    rR = p['rR']
    mR = p['mR']
    IRxx = p['IRxx']
    IRyy = p['IRyy']
    xB = p['xB']
    zB = p['zB']
    mB = p['mB']
    IBxx = p['IBxx']
    IByy = p['IByy']
    IBzz = p['IBzz']
    IBxz = p['IBxz']
    xH = p['xH']
    zH = p['zH']
    mH = p['mH']
    IHxx = p['IHxx']
    IHyy = p['IHyy']
    IHzz = p['IHzz']
    IHxz = p['IHxz']
    rF = p['rF']
    mF = p['mF']
    IFxx = p['IFxx']
    IFyy = p['IFyy']
    mT = mR + mB + mH + mF
    xT = (xB*mB + xH*mH + w*mF)/mT
    zT = (-rR*mR + zB*mB + zH*mH - rF*mF)/mT
    ITxx = IRxx + IBxx + IHxx + IFxx +mR*rR**2 + mb*zB**2 + mH*zH**2 + mF*rF**2
    ITxz = IBxz + IHxz - mB*xB*zB - mH*xH*zH + mF*w*rF
    IRzz = IRxx
    IFzz = IFxx
    ITzz = IRzz + IBzz + IHzz + IFzz + mB*xB**2 + mH*xH**2 + mF*w**2
    mA = mH + mF
    xA = (xH*mH + w*mF)/mA
    zA = (zH*mH - rF*mH)/mA
    IAxx = IHxx + IFxx + mH*(zH - zA)**2 + mF*(rF + zA)**2
    IAxz = IHxz - mH*(xH - xA)*(zH - zA) + mF*(w - xA)*(rF + zA)
    IAzz = IHzz + IFzz + mH*(xH - xA)**2 + mF*(w - xA)**2
    uA = (xA - w - c)*np.cos(l) - zA*np.sin(l)
    IAll = mA*uA**2 + IAxx*np.sin(l)**2 + 2*IAxz*np.sin(l)*np.cos(l) + IAzz*np.cos(l)**2
    IAlx = -mA*uA*zA + IAxx*np.sin(l) + IAxz*np.cos(l)
    IAlz = mA*uA*xA + IAxz*np.sin(l) + IAzz*np.cos(l)
    mu = c/w*cos(l)
    SR = IRyy/rR
    SF = IFyy/rF
    ST = SR + SF
    SA = mA*uA + mu*mT*xT
    Mpp = ITxx
    Mpd = IAlx + mu*ITxz
    Mdp = Mpd
    Mdd = IAll + 2*mu*IAlz + mu**2*ITzz
    M = np.array([[Mpp, Mpd], [Mdp, Mdd]])
    K0pp = mT*zT
    K0pd = -SA
    K0dp = K0pd
    K0dd = -SA*np.sin(l)
    K0 = np.array([[K0pp, K0pd], [K0dp, K0dd]])
    K2pp = 0
    K2pd = (ST - mT*zT)/w*np.cos(l)
    K2dp = 0
    K2dd = (SA + SF*np.sin(l))/w*np.cos(l)
    K2 = np.array([[K2pp, K2pd], [K2dp, K2dd]])
    C1pp = 0
    C1pd = mu*ST + SF*np.cos(l) + ITxz/w*np.cos(l) - mu*mT*zT
    C1dp = -(mu*ST + SF*np.cos(l))
    C1dd = IAlz/w*np.cos(l) + mu*(SA + ITzz/w*np.cos(l))
    C1 = np.array([[C1pp, C1pd], [C1dp, C1dd]])
    return M
