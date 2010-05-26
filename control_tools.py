import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

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
