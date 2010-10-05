import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
import uncertainties as u
import uncertainties.umath as umath
import uncertainties.unumpy as unumpy
from math import pi

from benchmark_bike_tools import *

class bicycle():
    # these are the various parameter sets
    ptypes = ['Benchmark', 'Sharp', 'Moore', 'Peterson']

    def __new__(shortname):
        # is there a data directory for this bicycle? if not, tell to put some
        # fucking data in the folder so we have something to work with!
        try os.path.isdir('data/' + shortname):
            pass
        except:
            print "Are you nuts?! Make a directory with basic data for your bicycle in
            data/shortname, where 'shortname' is the capitalized one word name
            of your bicycle. Then I can actually created a bicycle object."

    def __init__(shortname):
        '''
        shortname: string
            shortname of your bicicleta, one word, first letter is capped
        '''
        self.shortname = shortname
        self.directory = ('data/' + shortname)

        # are there any parameters already listed? grab any parameters and
        # store them

        self.params = {}
        try:
            for fname in os.listdir(self.directory):
                for typ in self.ptypes
                    # name of parameter file
                    fnamep = self.shortname + typ + '.p'
                    fnametxt = self.shortname + typ + '.txt'
                    if fname == fnamep:
                        # grab the .p file first if there is one
                        f = open(self.directory + fname, 'r')
                        self.params[typ] = pickle.load(f)
                        # set this flag so that
                        pfile = True
                        f.close()
                    elif fname == fnametxt and pfile != True:
                        f = open(fname, 'r')
                        self.params[typ] = {}
                        # parse the text file
                        for i, line in enumerate(f):
                            list1 = line[:-1].split(',')
                        # if there is an uncertainty value try to make
                        # a ufloat
                        try:
                            p[list1[0]] =
                            ufloat((eval(list1[1]),
                            eval(list1[2])))
                            # else keep it
                            # as a float
                        except:
                            p[list1[0]] = eval(list1[1])

        except: # get the measured data

            # load the data file
            f = open('data/udata.p', 'r')
            ddU = pickle.load(f)
            f.close()

            # the number of different bikes
            nBk = len(ddU['bikes'])

            # make a list of the bikes' names
            bikeNames = ddU['bikes']

            # calculate all the benchmark parameters
            par = {}

            # calculate the wheel radii
            par['rR'] = ddU['rearWheelDist']/2./pi/ddU['rearWheelRot']
            par['rF'] = ddU['frontWheelDist']/2./pi/ddU['frontWheelRot']

            # steer axis tilt in radians
            par['lambda'] = pi/180.*(90. - ddU['headTubeAngle'])

            # calculate the front wheel trail
            forkOffset = ddU['forkOffset']
            par['c'] = (par['rF']*unumpy.sin(par['lambda'])
                          - forkOffset)/unumpy.cos(par['lambda'])

            # wheelbase
            par['w'] = ddU['wheelbase']

            # calculate the dees
            par['d1'] = unumpy.cos(par['lambda'])*(par['c']+par['w']-par['rR']*unumpy.tan(par['lambda']))
            par['d3'] = -unumpy.cos(par['lambda'])*(par['c']-par['rF']*unumpy.tan(par['lambda']))

            # calculate the frame rotation angle
            # alpha is the angle between the negative z pendulum (horizontal) and the
            # positive (up) steer axis, rotation about positive y
            alphaFrame = ddU['frameAngle']
            # beta is the angle between the x bike frame and the x pendulum frame, rotation
            # about positive y
            betaFrame = par['lambda'] - alphaFrame*pi/180

            # calculate the slope of the CoM line
            frameM = -unumpy.tan(betaFrame)

            # calculate the z-intercept of the CoM line
            # frameMassDist is positive according to the pendulum ref frame
            frameMassDist = ddU['frameMassDist']
            cb = unumpy.cos(betaFrame)
            frameB = -frameMassDist/cb - par['rR']

            # calculate the fork rotation angle
            betaFork = par['lambda'] - ddU['forkAngle']*np.pi/180.

            # calculate the slope of the fork CoM line
            forkM = -unumpy.tan(betaFork)

            # calculate the z-intercept of the CoM line
            forkMassDist = ddU['forkMassDist']
            cb = unumpy.cos(betaFork)
            tb = unumpy.tan(betaFork)
            forkB = - par['rF'] - forkMassDist/cb + par['w']*tb
                
