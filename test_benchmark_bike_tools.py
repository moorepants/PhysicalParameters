from numpy import array
from numpy.linalg import eig
from uncertainties.numpy import uarray, nominal_values
from benchmark_bike_tools import ueig
def test_nom_ueig():
    sA = array([[1, 2], [3, 4]])
    A = array([[0.1, 0.2], [0.1, 0.3]])
    w, v = eig(A)
    uA = uarray((A, sA))
    uw, uv = ueig(A)
    assert nominal_values(uw) == w && nominal_values(uv) == v


