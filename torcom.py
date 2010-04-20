import pickle as p
import numpy as np
import re

f = open('period.p', 'r')
period = p.load(f)
f.close()
def space_out_camel_case(s):
        """Adds spaces to a camel case string.  Failure to space out string
        returns the original string.
        >>> space_out_camel_case('DMLSServicesOtherBSTextLLC')
        'DMLS Services Other BS Text LLC'
        """
        return re.sub('((?=[A-Z][a-z])|(?<=[a-z])(?=[A-Z]))', ' ', s)
tor = np.zeros((12, 8))
com = np.zeros((4, 8))
bN = ['Browser', 'Browserins', 'Crescendo', 'Fisher', 'Pista', 'Stratos',
        'Yellow', 'Yellowrev']
fst = ['First', 'Second', 'Third']
fffr = ['Frame', 'Fork', 'Fwheel', 'Rwheel']
tc = ['Torsional', 'Compound']
# average the periods
for k, v in period.items():
    km = re.sub('BrowserIns', 'Browserins', k)
    km = re.sub('YellowRev', 'Yellowrev', km)
    desc = space_out_camel_case(km).split()
    print desc
    if desc[0] == 'Rod':
        rodPeriod = np.mean(v)
    else:
        # if torsional, put it in the torsional matrix
        if desc[2] == tc[0]:
            r = fffr.index(desc[1])*3 + fst.index(desc[3])
            c = bN.index(desc[0])
            tor[r, c] = np.mean(v)
            print 'Added torsional to [', r, ',', c, ']'
        # if compound, put in in the compound matrix
        elif desc[2] == tc[1]:
            com[fffr.index(desc[1]), bN.index(desc[0])] = np.mean(v)
avgPer = {}
avgPer['tor'] = tor
avgPer['com'] = com
avgPer['rodPer'] = rodPeriod
f = open('avgPer.p', 'w')
p.dump(avgPer, f)
f.close()
