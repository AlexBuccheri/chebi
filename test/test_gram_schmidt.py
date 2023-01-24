import numpy as np

from src.gram_schmidt import classical_gram_schmidt


def test_gram_schmidt():
    """ Taken from Wiki:
    https://en.wikipedia.org/wiki/Gram–Schmidt_process#Example
    """
    # Two non-orthogonal vectors, stored row-wise
    V = np.array([[3, 1], [2, 2]])
    U = classical_gram_schmidt(V)

    assert np.allclose(U[0, :], V[0, :], atol=1.e-8), "u0 should always equal v0"
    assert np.allclose(U[1, :], np.array([-0.4, 1.2]), atol=1.e-8)

    # Check vectors are orthogonal
    inner_product = np.vdot(U[0, :], U[1, :])
    assert np.isclose(inner_product, 0., atol=1.e-8)
