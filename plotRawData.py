import os
import scipy.io.matlab.mio as mio
import matplotlib.pyplot as plt

# make a list of the folder contents
dirs, subdirs, filenames = list(os.walk('data/pendDat'))[0]
print filenames
i = 0
while i < len(filenames):
    d = {}
    mio.loadmat('data/pendDat/' + filenames[i], mdict=d)
    print 'Plotting', filenames[i]
    plt.plot(d['data'])
    plt.title(filenames[i])
    plt.show()
    raw_input()
    plt.close()
    i += 1
