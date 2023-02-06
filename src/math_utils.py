"""Matrix utilities.
"""
import numpy as np
import scipy
from functools import singledispatch
from typing import Tuple


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


def regularised_hilbert_matrix(n: int) -> np.ndarray:
    """ Regularised Hilbert Matrix.

    Defined as a square matrix with entries being the unit fractions.
    Hilbert matrices are canonical examples of ill-conditioned matrices.

    https://en.wikipedia.org/wiki/Hilbert_matrix
    :param n: Dimensions.
    :return: H: Hilbert matrix
    """
    H = np.empty(shape=(n, n))
    for i in range(0, n):
        for j in range(0, n):
            H[i, j] = 1. /(i + j + 1)

    return H + 0.0001 * np.eye(n)


def is_orthogonal(A: np.ndarray, atol=1.e-8) -> bool:
    """ Is a square matrix orthogonal or unitary.

    :param A: Matrix, with vectors stored columnwise
    :param atol:
    :return: bool.
    """
    n_vectors = A.shape[1]
    # Unitary
    if np.iscomplex(A).all():
        eye = np.eye(n_vectors, dtype=np.cdouble)
        return np.allclose(A.conj().T @ A, eye, atol=atol)
    # Orthogonal
    else:
        eye = np.eye(n_vectors, dtype=np.double)
        return np.allclose(A.T @ A, eye, atol=atol)


def sort_eigenpairs(values, vectors) -> Tuple[np.ndarray, np.ndarray]:
    """ Sort eigenpairs consistently, according to ascending order
    of the eigenvalues.

    Assumes eigenvectors are stored columnwise.

    :return:
    """
    indices = np.argsort(values)
    return values[indices], vectors[:, indices]


@singledispatch
def construct_hermitian_matrix(n) -> np.ndarray:
    input_type = type(n)
    msg = f'construct_hermitian_matrix not implemented for input argument of type({input_type})'
    raise NotImplementedError(msg)


@construct_hermitian_matrix.register(int)
def _(n: int) -> np.ndarray:
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


# @construct_hermitian_matrix.register(np.ndarray)
# def construct_hermitian_matrix(eigenvalues):
#     """ Construct a Hermitian matrix with caller-defined eigenvalues.
#
#     H = V D V^-1 = V D V^dagger if V are defined to be orthonormal
#
#     H = 0.5(M + M^dagger)
#     0.5(M + M^dagger) = V D V^dagger
#     eigenvectors.
#
#     :param eigenvalues:
#     :return: H: Hermitian matrix
#     """
#     n = eigenvalues.size
#     d = np.diagflat(eigenvalues)
#     v = np.empty(shape=(n, n))
#     # Suboptimal index access but I want columns to be eigenvectors
#     for i in range(0, n):
#         v[:, i] = random_complex_vector(n, normalise=True)
#
#     # Check if the resultant vectors are all linearly independent
#     # If not, replace any that are
#
#     # Orthogonalise random eigenvectors. Probably use QR with numpy
#     v =
#
#
#     # If this is not Hermitian, it breaks the mocking
#
#     return H
