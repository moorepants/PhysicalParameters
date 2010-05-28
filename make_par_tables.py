import pickle
import re
import os
import uncertainties as un

# load in the base data file
f = open('data/data.p', 'r')
data = pickle.load(f)
f.close()

# load in the parameter data file
f = open('data/par.p', 'r')
par = pickle.load(f)
f.close()

direct = 'parTables/'
for i, name in enumerate(data['bikes']):
    fname = ''.join(name.split())
    # open the new file
    f = open(direct + 'ParameterTable.tex', 'r')
    fn = open(direct + fname + 'Par.tex', 'w')
    for line in f:
        #print line
        # find all of the matches in the line
        test = re.findall('\|(\w*.)\|', line)
        #print 'search for u: ',  re.findall('\|(\w*\?)\|', line)
        #print 'search for nom: ', re.findall('\|(\w*)(?!\?)\|', line)
        # if there are matches
        if test:
            #print 'Found this!\n', test
            # go through each match and make a substitution
            for match in test:
                #print "replace this: ", match
                if match[-1] == '?':
                    try:
                        line = re.sub('\|(\w*\?)\|',
                                str(par[match[:-1]][i].std_dev()), line, count=1)
                        #print line
                    except:
                        pass
                elif match == 'bikename':
                    line = re.sub('\|bikename\|', name.upper(), line)
                else:
                    try:
                        line = re.sub('\|(\w*(?!\?))\|',
                                str(par[match][i].nominal_value), line, count=1)
                        #print line
                    except:
                        line = re.sub('\|(\w*(?!\?))\|', str(par[match][i]), line, count=1)
        #print line
        fn.write(line)
    f.close()
    fn.close()
    os.system('pdflatex -output-directory=parTables ' + direct + fname + 'Par.tex')
    os.system('rm parTables/*.aux')
    os.system('rm parTables/*.log')
