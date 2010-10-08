import numpy as np
import uncertainties as un
import uncertainties.unumpy as unp
from benchmark_bike_tools import ueig, ueig2

A = np.array([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
sA = np.array([[0.1, 0.2, 0.001], [0.1, 0.3, 0.4], [0.01, 0.01, 0.01]])
w, v = np.linalg.eig(A)
uA = unp.uarray((A, sA))
uw, uv = ueig2(uA)

# try a complex eigenvalue case
A = np.array([[  -0.44761596,   -4.55542354,    8.39223438, -281.6400387 ],
              [  27.22927912,  -38.95457009,    8.10466039, -406.06944732],
              [   1.,            0.,            0.,            0.        ],
              [   0.,            1.,            0.,            0.        ]])
sA = np.array([[  0.01, 0.5,    0.2,  1.0 ],
               [  0.5,  0.5,    0.01, 1.0 ],
               [  0.,   0.,     0.,   0.  ],
               [  0.,   0.,     0.,   0.  ]])
w, v = np.linalg.eig(A)
uA = unp.uarray((A, sA))
uw, uv = ueig2(uA)
