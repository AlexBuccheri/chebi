import scipy
import numpy as np
import pytest

from src.math_utils import construct_hermitian_matrix

from src.chebyshev_filtering.bound_estimators import bound_estimator


# Vector is not normalised, but the routine will take care of it
@pytest.fixture()
def arbitrary_vector() -> np.ndarray:
    v = np.array([-0.01559802-0.65376293j,  0.96589114+0.28524583j, -0.06326701+0.73671063j,
                   1.45009867-0.71205413j,  0.11344137+1.55939688j, -0.24354162-0.50079438j,
                  -1.19209895+0.79164884j, -0.64086740-0.08909565j, -0.47020116-1.59935008j,
                  -0.85327884+0.9131559j,   0.52813317-0.30832328j,  0.35150628+0.87180037j,
                  -0.00332531-1.031583j,   -0.00959612-0.31731432j, -0.11096189+0.55634474j,
                   0.53748494+0.51878254j,  0.74138261-0.57831131j, -0.08332997-0.7943208j,
                  -0.92478226-0.34391581j,  0.30477588+1.19448957j], dtype=np.cdouble)
    return v


def test_k_step_lanczos(arbitrary_vector):

    # Matrix dimensions
    n = arbitrary_vector.size

    # An arbitrary, Hermitian matrix
    h_mat = construct_hermitian_matrix(n)

    # Eigenvalues from direct solver
    w = scipy.linalg.eigh(h_mat, eigvals_only=True)
    assert np.isclose(np.amax(w), 4349.85677, atol=1.e-8), "Maximal eigenvalue from direct eigenvalue solver"

    # Approximate max eigenvalue, for k = [1, 10], using k-step Lanczos
    k_max = 10
    upper_bound = np.empty(shape=k_max)
    for k in range(1, k_max + 1):
        upper_bound[k-1] = bound_estimator(h_mat, arbitrary_vector, k=k)

    # Note, the index is k - 1 due to python
    assert np.isclose(upper_bound[0], 85.97157054573876, atol=1.e-8), ' k = 1 is clearly a bad choice'

    # Approximate max eigenvalue is > max eigenvalue, for k > 1
    assert np.isclose(upper_bound[1], 5224.174308483503, atol=1.e-8), 'k = 2'
    assert np.isclose(upper_bound[2], 4349.856765348369, atol=1.e-8), 'k = 3'
    assert np.isclose(upper_bound[3], 6493.329975538969, atol=1.e-8), 'k = 4'

    assert (upper_bound[4] / np.amax(w) - 1) * 100 < 5, 'Approximate result is within 5% of max_lambda(H)'

    assert np.isclose(upper_bound[4], 4501.741362327415, atol=1.e-8), 'k = 5'
    assert np.isclose(upper_bound[5], 4349.856765348396, atol=1.e-8), 'k = 6'
    assert np.isclose(upper_bound[6], 4542.35434538459, atol=1.e-8), 'k = 7'
    assert np.isclose(upper_bound[7], 4512.427781540934, atol=1.e-8), 'k = 8'
    assert np.isclose(upper_bound[8], 4349.856765349421, atol=1.e-8), 'k = 9'
    assert np.isclose(upper_bound[9], 4374.370170140614, atol=1.e-8), 'k = 10'
