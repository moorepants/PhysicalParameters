from numpy import *

def uvec(x):
    xhat = x/linalg.norm(x)
    return xhat
# Victor on the Stratos
# name the four points from the inkscape superimposed lines
a = array([190.716, 1052.483])
b = array([457.633, 533.479])
c = array([-312.107, -371.112])
d = array([645.204, -352.409])
# calculate the two vectors
v1 = a - b
v2 = c -d
# calculate the angle between the two vectors
theta = arccos(dot(uvec(v1), uvec(v2)))
print "Victor on the Stratos =", rad2deg(theta)
# Victor on the Browser
# name the four points from the inkscape superimposed lines
a = array([406.081, 989.949])
b = array([512.147, 453.558])
c = array([-334.360, -175.767])
d = array([705.086, -125.259])
# calculate the two vectors
v1 = a - b
v2 = c -d
# calculate the angle between the two vectors
theta = arccos(dot(uvec(v1), uvec(v2)))
print "Victor on the Browser =", rad2deg(theta)

