"""Matrix utilities.
"""
import numpy as np
import scipy


def unit_vector(u: np.ndarray, order=2) -> np.ndarray:
    """ Normalise vector.

    :param u: Vector
    :param order: Order of the norm
    :return: Unit vector
    """
    return u / np.linalg.norm(u, ord=order)


def random_complex_vector(n: int, normalise=False) -> np.ndarray:
    """ Complex vector generated with a random seed.

    :param n: Length of vector
    :param normalise: Return unit vector
    :return: v: Random complex vector
    """
    rng = np.random.default_rng()
    v = np.empty(shape=n, dtype=np.cdouble)
    v.real = rng.standard_normal(n)
    v.imag += rng.standard_normal(n)
    if normalise:
        v = np.linalg.norm(v)
    return v


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
