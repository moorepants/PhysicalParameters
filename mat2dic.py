import pickle
import scipy.io.matlab.mio as mio
import uncertainties as u

# load the main data file into a dictionary
d = {}
mio.loadmat('data/data.mat', mdict=d)

# the number of different bikes
nBk = len(d['bikes'])

# make a list of the bikes' names
bikeNames = []
for bike in d['bikes']:
    # get rid of the weird matlab unicoding
    bikeNames.append(bike[0][0].encode('ascii'))

# clean up the matlab imports
d['bikes'] = bikeNames
del(d['__globals__'], d['__header__'], d['__version__'])
for k, v in d.items():
    if np.shape(v)[0] == 1:
        d[k] = v[0]

# pickle the data dictionary
f = open('data/data.p', 'w')
pickle.dump(d, f)
f.close()

# make a dictionary for the measurement standard deviations
dU = {}
f = open('data/MeasUncert.txt', 'r')
for line in f:
    l = line.split(',')
    dU[l[0]] = eval(l[1])
f.close()

# add the uncertainties to the data
ddU = {}
for k, v in dU.items():
    for pair in zip(d[k].flatten(), np.ones_like(d[k].flatten())*v):
        if k in ddU.keys():
            ddU[k].append(u.ufloat((float(pair[0]), pair[1])))
        else:
            ddU[k] = []
            ddU[k].append(u.ufloat((float(pair[0]), pair[1])))
    ddU[k] = np.array(ddU[k])
    if ddU[k].shape[0] > 8:
        ddU[k] = ddU[k].reshape((ddU[k].shape[0]/8, -1))

# pickle the data with the uncertainties
f = open('data/udata.p', 'w')
pickle.dump(ddU, f)
f.close()
