""" Gram-Schmidt Orthogonalisation
"""
import numpy as np


def unit_vector(u: np.ndarray) -> np.ndarray:
    return u / np.linalg.norm(u)


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


def classical_gram_schmidt(V: np.ndarray, normalise=True) -> np.ndarray:
    """Classical Gram-Schmidt Orthogonalisation.

    Vectors may not quite be orthogonal due to rounding
    errors.

    ADD EQUATION

    Orthogonalise a set of vectors V.

    :param V: Set of vectors, stored row-wise.

    :return: Matrix of orthnormal vectors. (check they're normalised)
    Vectors stored in rows.
    """
    n_vectors = V.shape[0]
    U = np.empty(shape=V.shape)
    proj_u = np.zeros(shape=V.shape[0])

    # First step. u0 = v0
    U[0, :] = np.copy(V[0, :])

    for k in range(1, n_vectors):
        proj_u += projection_operator(U[k-1, :], V[k, :])
        U[k, :] = V[k, :] - proj_u[:]

    if normalise:
        for k in range(0, n_vectors):
            U[k, :] = unit_vector(U[k, :])

    return U


def modified_gram_schimdt(V: np.ndarray) -> np.ndarray:
    """Modified Gram-Schmidt Orthogonalisation.

    This approach gives the same result as the original formula in exact arithmetic
    and introduces smaller errors in finite-precision arithmetic.

    Used this ref: https://laurenthoeltgen.name/post/gram-schmidt/
    Wiki's description is really unclear.

    TODO Flesh out the maths in the description

    :return:
    """
    n_vectors = V.shape[0]
    U = np.copy(V)

    for j in range(0, n_vectors):
        U[j, :] = unit_vector(U[j, :])
        for i in range(j+1, n_vectors):
            U[i, :] -= np.vdot(U[i, :], U[j, :]) * U[j, :]

    return U
