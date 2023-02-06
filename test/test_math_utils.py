import numpy as np


from src.math_utils import is_orthogonal, regularised_hilbert_matrix
from src.gram_schmidt import classical_gram_schmidt, modified_gram_schimdt



def test_qr_orthogonalisation():

    input_v = np.array([[3.0, 1.0], [2.0, 2.0]])
    Q, _ = np.linalg.qr(input_v)

    orthogonalised_v = np.array([[-0.83205029, -0.5547002],
                                 [-0.5547002,  0.83205029]])

    # These routines agree with each other but not QR, so I need to check which ref is valid
    # But is_orthogonal says true in each case... so maybe all is legit?
    gs_vectors = classical_gram_schmidt(input_v)
    print(gs_vectors)
    print(is_orthogonal(gs_vectors))
    mgs_vectors = modified_gram_schimdt(input_v)
    print(mgs_vectors)
    print(is_orthogonal(mgs_vectors))

    np.allclose(orthogonalised_v, Q, atol=1.e-8), 'Q from QR factorisation gives orthogonalised vectors'
    print(is_orthogonal(Q))

# def test_qr_orthogonalisation_3by3():
#     # test2 = np.array([[1.0, 1.0, 0.0], [1.0, 3.0, 1.0], [2.0, -1.0, 1.0]])
#     # Q2, _ = np.linalg.qr(test2)



