""" Implementations of Chebyshev filters
"""
import numpy as np
from typing import Tuple


from src.math_utils import random_complex_vector, is_orthogonal
from src.chebyshev_filtering.bound_estimators import bound_estimator
from src.rayleigh_ritz import rayleigh_ritz_pairs, ritz_pair_accuracy


def center(a, b) -> float:
    return 0.5 * (a + b)


def half_width(a, b) -> float:
    return 0.5 * (b - a)


def sigma(a, c, e):
    """
    Confirm this is equivalent to -0.5 (a**2 + b**2)
    :param a:
    :param c:
    :param e:
    :return:
    """
    return e / (a - c)


# TODO(Alex) Need some test of this routine
def chebyshev_filter(H: np.ndarray, input_X: np.ndarray, m: int, a: float, b: float) -> np.ndarray:
    """ Filter vectors in X by an m-degree Chebyshev polynomial that dampens on the interval [a,b].

    Based on algorithm 4.3 of "Self-consistent-field calculations using Chebyshev-filtered subspace
    iteration".

    :param H: Hermitian matrix.
    :param input_X: Array of approximate eigenvectors, stored column-wise. REFACTOR LATER
    :param m: Degree of Chebyshev polynomial
    :param a: Lower bound of spectrum to filter (some value above the highest occupied state)
    :param b: Upper bound of spectrum to filter: Some value slightly > max_value(A).
    :return: Y: Output filtered vectors.
    """
    e = half_width(a, b)
    c = center(a, b)
    sigma = e / (a - c)
    sigma1 = sigma
    X = np.copy(input_X)
    Y = (H @ X - c * X) * sigma1 / e

    for i in range(2, m + 1):
        sigma2 = 1 / (2 / sigma1 - sigma)
        Y_new = 2 * (H @ Y - c * Y) * (sigma2 / e) - (sigma * sigma2 * X)
        X = Y
        Y = Y_new
        sigma = sigma2

    return Y


def eigensolver_with_chebyshev_filtering(ham: np.ndarray, phi: np.ndarray, lower_bound: float) -> Tuple[np.ndarray, np.ndarray]:
    """ Solve the standard eigenvalue problem using chebyshev filtering
    to obtain a subspace of the eigen spectrum.

    Routine Outline:
     * Estimate an upper bound
     * Filter the eigenvectors according to the subspace defined by [a, b]
     * Orthogonalise the filtered eigenvectors using QR factorisation
     * Apply the Rayleigh-Ritz method to solve for approximate eigenstates of H,
       by solving a subspace defined by the filtering.

    k and m set according to recommendations from:
    "Self-consistent-field calculations using Chebyshev-filtered subspace iteration".
    Recommended m = [8, 20]

    :return: Eigenvalues and eigenvectors of a subspace < a.
    """
    # Chebyshev polynomial degree
    m = 10
    # Number of Lanczos steps
    k = 6
    # Random starting vector
    v_i = random_complex_vector(ham.shape[0])

    a = lower_bound
    b = bound_estimator(ham, initial_v=v_i, k=k)
    phi = chebyshev_filter(ham, phi, m, a, b)
    phi, _ = np.linalg.qr(phi)
    assert is_orthogonal(phi, atol=1.e-8), 'Approximate eigenvectors have lost orthogonality'
    values, phi = rayleigh_ritz_pairs(ham, phi, consistent_sign=True)
    convergence = ritz_pair_accuracy(ham, values, phi)
    return values, phi
