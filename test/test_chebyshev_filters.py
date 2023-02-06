import scipy
import numpy as np

from src.math_utils import construct_hermitian_matrix, sort_eigenpairs
from src.chebyshev_filtering.chebyshev_filters import eigensolver_with_chebyshev_filtering
from src.rayleigh_ritz import enforce_sign_consistency


def test_eigensolver_with_chebyshev_filtering():

    # Hamiltonian dimensions
    n = 30
    # Some arbitrary H with many eigenvalues ~ 0 but 6 above it.
    h = construct_hermitian_matrix(n)
    # Massage the diagonal to get something more useful
    h[1, 1] += 4
    h[2, 2] += 4
    h[3, 3] += 4
    h[17, 17] += 1
    h[18, 18] += 2
    h[19, 19] += 3

    # Directly diagonalise to get a reference result
    eigenvalues, eigenvectors = scipy.linalg.eigh(h)
    ref_eigenvalues = np.array([-1.11568275e+03, -1.05589986e-12, -7.52817202e-13, -2.83841622e-14,
                                -2.63441010e-14, -2.56091502e-14, -2.52813653e-14, -2.49754743e-14,
                                -2.45032049e-14, -2.07440939e-14, -1.13652793e-14, -9.65275658e-15,
                                -7.88850069e-15,  3.17904758e-16,  7.79772328e-16,  5.18602292e-15,
                                 9.32016356e-15,  9.94402853e-15,  1.17700854e-14,  2.58151365e-14,
                                 3.13074022e-14,  5.23956935e-14,  6.24786058e-13,  9.55349592e-01,
                                 1.91050048e+00,  2.76125141e+00,  2.89804096e+00,  3.99901858e+00,
                                 4.00000000e+00,  1.46321586e+04])
    assert np.allclose(eigenvalues, ref_eigenvalues, atol=1.e-8)

    # Arbitrary definition. All eigenvalues <= 0
    n_occupied = 24

    # Iterative solver one would use for first SCF step
    w_lanczos, v_lanczos_unsorted = scipy.sparse.linalg.eigsh(h, k=n_occupied, which='SA')
    enforce_sign_consistency(v_lanczos_unsorted)
    w_lanczos, v_lanczos = sort_eigenpairs(w_lanczos, v_lanczos_unsorted)

    assert np.allclose(w_lanczos, ref_eigenvalues[0:n_occupied], atol=1.e-8)

    # The lower bound of the unwanted spectrum should be a value corresponding to the energy related
    # to the highest occupied state. However, this can be set to an energy that is greater than the energy
    # of the highest occupied state.
    lower_bound = 0.5

    # Solve the eigenvalue problem with Chebyshev filtering
    w_ch, v_ch = eigensolver_with_chebyshev_filtering(h, v_lanczos, lower_bound)
    assert np.allclose(w_ch, ref_eigenvalues[0:n_occupied], atol=1.e-8)

    # TODO(Alex) Need some metric to test that the eigenvectors I get are
    # (approximately) correct. Maybe they converge m and k in `eigensolver_with_chebyshev_filtering`
    # Maybe also look at the charge density
    # for i in range(0, n_occupied):
    #     print( v_ch[:, i] - v_lanczos[:, i])
