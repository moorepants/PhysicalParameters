import pickle
import re
import uncertainties as un

# load in the base data file
f = open('data/data.p', 'r')
data = pickle.load(f)
f.close()

nBk = len(data['bikes'])

# load in the parameter data file
f = open('data/par.p', 'r')
par = pickle.load(f)
f.close()

for i, name in enumerate(data['bikes']):
    fname = ''.join(name.split()) + 'RiderCan.p'
    file = open(dir + fname)
    can = pickle.load(file)
    file.close()

