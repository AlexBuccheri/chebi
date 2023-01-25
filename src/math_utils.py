"""Matrix utilities.
"""
import numpy as np
import scipy


def construct_hermitian_matrix(n: int) -> np.ndarray:
    """ Construct a Hermitian matrix.

    Could use random numbers and fix the seed.
    I am using [1, n^2], then reshaping. This means the elements
    get larger w.r.t. the array indices.

    :param n: Matrix dimensions
    :return: H: Hermitian matrix
    """
    # Arbitrary scaling factor to set imaginary values to some
    # percentage of the real values
    scale = 0.1

    # Construct vector from 1 to n**2, then reshape to 2D
    M = np.arange(1, n * n + 1, step=1, dtype=np.cdouble).reshape(n, n)
    # Add imaginary values
    M.imag = scale * np.real(M)

    # Hermitian definition
    H = 0.5 * (M + M.T.conj())
    assert scipy.linalg.ishermitian(H), "H not Hermitian"

    return H
