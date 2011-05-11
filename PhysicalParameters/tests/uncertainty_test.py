'''test code for inverse and least squares with the uncertainties package'''
import uncertainties as u
import numpy as np

a = np.array([[2, 1], [3, -1]])
b = np.array([9, 16])

x = np.linalg.solve(a, b)
print "numpy solve", x
x = u.numpy.linalg.solve(a, b)
print "unc solve", x
x = np.dot(np.linalg.inv(a), b)
print "numpy inv", x
x = u.numpy.dot(u.numpy.linalg.inv(a), b)
print "unc inv", x
x = np.linalg.lstsq(a, b)[0]
print "numpy lstsq", x
x = u.numpy.linalg.lstsq(a, b)[0]
print "unc lstsq", x

print "now with uncertainties!"
au = np.matrix(u.array_u(([[2, 1], [3, -1]], [[0.2, 0.1], [0.3, 0.1]])))
bu = u.array_u(([9, 16], [0.1, 0.1]))

xu = np.dot(au.I, bu)
print "numpy inv", xu   # Prints the result with uncertainties!

xu_lst = np.dot(np.dot(np.dot(au.T, au).I, au.T), bu)
print xu_lst
