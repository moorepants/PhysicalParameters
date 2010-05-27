import pickle
import re
import os
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

direct = 'ParTables'
for i, name in enumerate(data['bikes']):
    fname = ''.join(name.split()) 
    # open the new file
    f = open(direct + '/ParameterTable.tex', 'r')
    fn = open(direct + '/' + fname + 'RiderPar.tex', 'w') 
    for line in f:
        print line
        # find all of the matches in the line
        test = re.findall('\|(\w*.)\|', line)
        print 'search for u: ',  re.findall('\|(\w*\?)\|', line)
        print 'search for nom: ', re.findall('\|(\w*)(?!\?)\|', line)
        # if there are matches
        if test:
            print 'Found this!\n', test
            # go through each match and make a substitution
            for match in test:
                print "replace this: ", match
                if match[-1] == '?':
                    try:
                        line = re.sub('\|(\w*\?)\|', '%-6.3f' % par[match[:-1]][i].std_dev(), line,
                                count=1)
                        print line
                    except:
                        pass
                else:
                    try:
                        line = re.sub('\|(\w*(?!\?))\|', '%.3f' % par[match][i].nominal_value, line, count=1)
                        print line
                    except:
                        pass
        print line
        fn.write(line)
    f.close()
    fn.close()
    os.system('pdflatex -output-directory=ParTables ' + direct + '/' + fname + 'RiderPar.tex')
    os.system('rm ParTables/*.aux')
    os.system('rm ParTables/*.log')
