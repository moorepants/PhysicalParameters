import pickle as p
from uncertainties.unumpy import nominal_values as nom

from benchmark_bike_tools import *

# load in the base data file
f = open('data/data.p', 'r')
data = p.load(f)
f.close()

# write the parameter files
for i, name in enumerate(data['bikes']):
    dir = 'data/bikeParameters/'
    fname = ''.join(name.split())
    M, C1, K0, K2, param = bmp2cm(dir + fname + 'Par.txt')
    try:
        A, B = abMatrix(nom(M), nom(C1), nom(K0), nom(K2), nom(param['v']),
                nom(param['g']))
    except:
        A, B = abMatrix(M, C1, K0, K2, param['v'], param['g'])
    dir = 'data/bikeCanonical/'
    file = open(dir + fname + 'Can.txt', 'w')
    for mat in ['M','C1', 'K0', 'K2', 'A', 'B']:
        if mat == 'A' or mat == 'B':
            file.write(mat + ' (v = ' + str(param['v']) + ')\n')
        else:
            file.write(mat + '\n')
        file.write(str(eval(mat)) + '\n')
    file.close()
    file = open(dir + fname + 'Can.p', 'w')
    p.dump({'M':M, 'C1':C1, 'K0':K0, 'K2':K2, 'A':A, 'B':B, 'v':param['v']},
            file)
    file.close()
