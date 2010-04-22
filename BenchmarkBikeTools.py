def bmp2cm(filename):
    ''' Converts the benchmark bicycle parameters to the canonical matrices'''
    f = open(filename, 'r')
    parameters = {}
    parameters['var'] = []
    parameters['val'] = []
    for i, line in enumerate(f):
        list = line[:-1].split(',')
        parameters['var'].append(list[0])
        try:
            parameters['val'].append(float(list[1]))
        except:


    return parameters
