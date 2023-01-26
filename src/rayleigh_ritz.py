"""Rayleigh–Ritz method
"""
import numpy as np
import scipy


def rayleigh_ritz_pairs(h: np.ndarray, phi: np.ndarray, consistent_sign=True):
    """ Approximate solution of the standard eigenvalue problem
    for a subspace of eigenstates.

    Approximate an eigenvalue problem Hx = wx for
    H.shape(n, n), using a smaller matrix of size m < n,
    generated from phi.shape = (n, m).

    See algorithm 4.1

    and the [wikipedia page](https://en.wikipedia.org/wiki/Rayleigh–Ritz_method)

    TODO Add maths here
    H_hat = phi^dagger H phi

    :param h: Hermitian or symmetric Hamiltonian
    :param phi: Matrix of approximate (?) eigenvectors, stored column-wise.
    Expect a tall/skinny matrix, such that m vectors < n elements.
    :return: Ritz pairs, which are approximations to the eigenvalues and
    eigenvectors of h, respectively.
    """
    # TODO Check to ensure it's Hermitian or symmetric, too

    if h.shape[0] != h.shape[1]:
        raise ValueError('Hamiltonian must be square')

    # Expect eigenvectors stored column-wise, and there to be fewer vectors
    # than the length of a given vector (i.e. fewer vectors than the basis size)
    if phi.shape[1] >= phi.shape[0]:
        raise ValueError('Expect vectors of phi to be stored column-wise, '
                         'such that phi.shape[1] < phi.shape[0]')

    # Eigenvector dimension is inconsistent with Hamiltonian dimensions
    if h.shape[1] != phi.shape[0]:
        raise ValueError('Number of rows in phi must equal number of columns '
                         'in H, for matrix multiplication to work')

    h_hat = np.matmul(phi.conj().T, h @ phi)

    # Solve the Ritz values and eigenvectors of h_hat
    ritz_values, q = scipy.linalg.eigh(h_hat)
    ritz_values.sort()

    # Rotate the basis
    ritz_vectors = phi @ q

    if consistent_sign:
        enforce_sign_consistency(ritz_vectors)

    return ritz_values, ritz_vectors


def enforce_sign_consistency(eigenvectors: np.ndarray) -> None:
    """ Enforce sign consistency of SCIPY eigenvectors.

    SCIPY's Eigh computes mass normalized eigenvectors but
    there is still one degree of freedom: the sign.

    Ensure the top value is positive for all eigenvectors.
    If the last element of a vector is negative, multiply through by -1.

    Thanks to ref https://boffi.github.io/dati_2018/hw02/2.pdf for
    this detail.

    :param eigenvectors: eigenvectors, stored column-wise.
    :return: Mutates input array.
    """
    for i, val in enumerate(eigenvectors[-1]):
        if val < 0:
            eigenvectors[:, i] *= -1


# TODO(Alex) Test this
def ritz_pair_accuracy(A, ritz_values, ritz_vectors):
    """
    || Ax_i - lambda_i x_i ||

    ALso look at ~ pg 79 of this: https://core.ac.uk/download/pdf/235413726.pdf
    :return:
    """
    # TODO Test of vector * matrix gives the correct behaviour
    # i.e. ritz_values @ ritz_vectors vs
    # for i in range(0, n):
    #   ritz_values[i] * ritz_vectors[:, i]
    np.linalg.norm(A @ ritz_vectors - ritz_values @ ritz_vectors, ord=1, axis=0)
