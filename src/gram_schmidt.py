""" Gram-Schmidt Orthogonalisation
"""
import numpy as np


def projection_operator(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Projects the vector v orthogonally onto the line spanned by vector u.

    :param v: vector
    :param u: vector
    :return: v projected onto line span by u.
    """
    assert v.ndim == 1, "v must be a vector"
    assert u.ndim == 1, "u must be a vector"
    assert v.size == u.size, "Input vectors v and u must be the same size"

    # If u = 0, so is the projection
    if np.allclose(u, 0):
        return np.full_like(u, 0)

    vu = np.vdot(v, u)
    uu = np.vdot(u, u)
    proj_v_onto_u = (vu / uu) * u

    return proj_v_onto_u


def classical_gram_schmidt(V: np.ndarray) -> np.ndarray:
    """Classical Gram-Schmidt Orthogonalisation.

    ADD EQUATION

    Orthogonalise a set of vectors V.

    :param V: Set of vectors, stored row-wise

    :return: Matrix of orthogonal vectors.
    Vectors stored in rows.
    """
    n_vectors = V.shape[0]
    U = np.empty(shape=V.shape)
    proj_u = np.zeros(shape=V.shape[0])

    # First step. u0 = v0
    U[0, :] = V[0, :]

    for k in range(1, n_vectors):
        proj_u += projection_operator(U[k-1, :], V[k, :])
        U[k, :] = V[k, :] - proj_u[:]

    return U


