import numpy as np
import scipy


from src.rayleigh_ritz import rayleigh_ritz_pairs, enforce_sign_consistency, ritz_pair_accuracy


def test_rayleigh_ritz_pairs():
    """
    Taken from https://en.wikipedia.org/wiki/Rayleigh–Ritz_method#Example
    Note, this diverges from the example because
        a) `eigh` mass-normalises the resultant vectors and
        b) the sign remains a degree of freedom, so I fix a convention.
    """
    A = np.array([[2., 0., 0.],
                  [0., 2., 1.],
                  [0., 1., 2.]])

    V = np.array([[0., 0.],
                  [1., 0.],
                  [0., 1.]])

    # Stored per column
    expected_eigenvectors = np.array([[0.,          1.,    0.],
                                      [-0.70710678,  0.,    0.70710678],
                                      [0.70710678, 0.,    0.70710678]])

    # Exact eigenstates of A
    w, v = scipy.linalg.eigh(A)
    enforce_sign_consistency(v)

    assert np.allclose(w, [1., 2., 3.], atol=1.e-8), 'All (exact) eigenvalues'
    assert np.allclose(v, expected_eigenvectors, atol=1.e-8), 'All (exact) eigenvectors'

    # Ritz approximation to A
    ritz_values, ritz_vectors = rayleigh_ritz_pairs(A, V)

    assert np.allclose(ritz_values, [1., 3.], atol=1.e-8), 'Ritz returns first and last eigenstates of A'
    assert np.allclose(ritz_vectors[:, 0], expected_eigenvectors[:, 0], atol=1.e-8)
    assert np.allclose(ritz_vectors[:, -1], expected_eigenvectors[:, -1], atol=1.e-8)


# TODO(Alex) Implement me
def test_ritz_pair_accuracy():
    pass
