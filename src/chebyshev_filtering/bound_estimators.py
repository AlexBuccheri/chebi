""" Estimate largest eigenvalue of a Hermitian matrix.
"""
import warnings

import numpy as np
from typing import Tuple

from src.math_utils import random_complex_vector


def k_step_lanczos(A: np.ndarray, v: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """ k-step Lanczos.

    Based on algorithm 4.4. in "Self-consistent-field calculations using
    Chebyshev-filtered subspace iteration".

    Note, the safeguard for beta = 0 is not included,
    as this is not expected to occur for small k (<= 10).

    :param A: Hermitian matrix.
    :param v: Random complex vector of length A.shape[0].
    :param k: Number of steps (iterations) in Lanczos decomposition.
    :return: T and f: Tri-diagonal matrix and the residual vector of the
    kth iteration.
    """
    if k < 1:
        raise ValueError('k cannot be less than 1')
    if k > 10:
        raise warnings.warn('k > 10 may lead to beta=0 (which will crash the routine)')

    # Ensure input vector is normalised
    v = v / np.linalg.norm(v, ord=2)
    # Tri-diagonal matrix
    T = np.zeros(shape=(k, k), dtype=np.cdouble)
    # Residual vector
    f = np.matmul(A, v)

    alpha = np.vdot(f, v)
    f = f - alpha * v
    T[0, 0] = alpha

    for j in range(1, k):
        beta = np.linalg.norm(f, ord=2)
        v0 = np.copy(v)
        v = f / beta
        f = np.matmul(A, v)
        f = f - beta * v0
        alpha = np.vdot(f, v)
        f = f - alpha * v
        T[j, j] = alpha
        T[j - 1, j] = beta
        T[j, j - 1] = beta

    return T, f


def bound_estimator(A: np.ndarray, initial_v=None, k=10) -> float:
    """ Estimate the upper bound of sigma(H) by k-step Lanczos.

    Based on algorithm 4.4. in "Self-consistent-field calculations using
    Chebyshev-filtered subspace iteration".

    :param A: Hermitian matrix.
    :param initial_v: Optional, random complex vector of length A.shape[0].
    :param k: Number of steps (iterations) in Lanczos decomposition.
    :return: Upper bound for largest eigenvalue of matrix A.
    """
    # Generate random vector and convert to unit vector
    if initial_v is None:
        initial_v = random_complex_vector(A.shape[0])

    T, f = k_step_lanczos(A, initial_v, k)

    # numpy apparently can compute the l2 norm of a 2D matrix
    # which is equivalent to its spectral norm.
    # See https://en.wikipedia.org/wiki/Matrix_norm
    upper_bound = np.linalg.norm(T, ord=2) + np.linalg.norm(f, ord=2)
    return upper_bound
