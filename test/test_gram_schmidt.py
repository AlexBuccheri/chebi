import numpy as np

from src.gram_schmidt import classical_gram_schmidt, modified_gram_schimdt





def test_gram_schmidt():
    """ Taken from Wiki:
    https://en.wikipedia.org/wiki/Gram–Schmidt_process#Example
    """
    # Two non-orthogonal vectors, stored row-wise
    V = np.array([[3, 1], [2, 2]])
    U = classical_gram_schmidt(V, normalise=False)

    assert np.allclose(U[0, :], V[0, :], atol=1.e-8), "u0 should always equal v0"
    assert np.allclose(U[1, :], np.array([-0.4, 1.2]), atol=1.e-8)

    # Check vectors are orthogonal
    inner_product = np.vdot(U[0, :], U[1, :])
    assert np.isclose(inner_product, 0., atol=1.e-8)

    assert not np.allclose(U.T @ U, np.eye(2), atol=1.e-8), "Vectors {u} are not orthonormal"


def test_gram_schmidt_with_normalisation():
    """ Taken from Wiki:
    https://en.wikipedia.org/wiki/Gram–Schmidt_process#Example
    """
    # Two non-orthogonal vectors, stored row-wise
    V = np.array([[3, 1], [2, 2]])
    U = classical_gram_schmidt(V)

    assert np.allclose(U[0, :], unit_vector(V[0, :]), atol=1.e-8), "u0 should equal hat{v0}"

    # Check vectors are orthonormal
    assert np.allclose(U.T @ U, np.eye(2), atol=1.e-8)


def test_modified_gram_schimdt():
    n = 6
    V = regularised_hilbert_matrix(n)
    U = modified_gram_schimdt(V)

    # Check vectors are orthonormal
    assert np.allclose(U.T @ U, np.eye(n), atol=1.e-8)
    print(U.T @ U)

    # Compare to classical method
    # Gives what looks like junk - makes me wonder if I evaluated correctly
    # U_c = classical_gram_schmidt(V, normalise=True)
    # print(U_c.T @ U_c)
