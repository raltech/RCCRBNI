import numpy as np
import os

'''
Score func calculates how well the produced dictionary is
Score is between 0 and the number of cells
'''

def relevance_score_func(A, dict_hat_count=None, relevance=None):
    import pdb; pdb.set_trace()
    return np.dot(dict_hat_count, relevance)


def scatter_matrix_score_func(A):
    # calculate the average variance associated with each action
    avg_var = np.mean(np.var(A, axis=1))

    # calculate the eigenvalues of scatter matrix
    eigvals = np.linalg.eigvals(np.transpose(A) @ A)

    return np.sum(eigvals < avg_var)

def span_score_func(A):
    # calculate the span of a set of A
    # :param A: list of A
    # :return: span of A
    span = np.linalg.matrix_rank(A)
    return span

def qr_rank_score_func(A, tol=1e-9, dict_hat_count=None):
    Q, R = np.linalg.qr(A)
    rank = np.sum(np.abs(np.diag(R)) > tol)
    row_indices = np.argsort(np.abs(np.diag(R)))[::-1][:rank]
    rank = np.sum(dict_hat_count[row_indices])
    return rank

def RREF_score_func(A, dict_hat_count=None):
    # Row-reduced echelon form (Gaussian elimination)
    A = A.astype(float)
    m, n = A.shape
    row_indices = []
    pivot_row = 0
    for col in range(n):
        pivot = np.argmax(np.abs(A[pivot_row:, col])) + pivot_row
        if np.abs(A[pivot, col]) < 1e-9:
            continue
        A[[pivot_row, pivot]] = A[[pivot, pivot_row]]
        row_indices.append(pivot)
        for i in range(pivot_row + 1, m):
            A[i] -= A[pivot_row] * A[i, col] / A[pivot_row, col]
        pivot_row += 1
    import pdb; pdb.set_trace()
    
    # need to convert row_indices to action indices
    rank = np.sum(dict_hat_count[row_indices])
    return rank

def cosine_sim_score_func(vectors):
    score = 0.0
    _, dim = vectors.shape
    # calculate the magnitude of each vector
    mag = np.linalg.norm(vectors, axis=1)

    # if the magnitude is 0, then change the magnitude to 1
    mag[mag == 0] = 1

    # calculate the cosine similarity between each vector and unit vector
    for d in range(dim):
        unit_vec = np.zeros(dim)
        unit_vec[d] = 1
        cos_sim = np.dot(vectors, unit_vec) / mag
        score += np.max(cos_sim)

    return score