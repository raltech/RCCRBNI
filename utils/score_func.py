import numpy as np
import os

'''
Score func calculates how well the produced dictionary is
Score is between 0 and the number of cells
'''

def scatter_matrix_score_func(vectors):
    # calculate the average variance associated with each action
    avg_var = np.mean(np.var(vectors, axis=1))

    # calculate the eigenvalues of scatter matrix
    eigvals = np.linalg.eigvals(np.transpose(vectors) @ vectors)

    return np.sum(eigvals < avg_var)

def span_score_func(vectors):
    # calculate the span of a set of vectors
    # :param vectors: list of vectors
    # :return: span of vectors
    span = np.linalg.matrix_rank(vectors)
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
    rank = np.sum(dict_hat_count[row_indices])
    return rank

def volume_score_func(vectors):
    dot_products = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            dot_products.append(np.abs(np.dot(vectors[i], vectors[j])))
    return sum(dot_products) / len(dot_products)

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

def mag_cosine_sim_score_func(vectors):
    score = 0.0
    _, dim = vectors.shape
    # calculate the magnitude of each vector
    mag = np.linalg.norm(vectors, axis=1)

    # calculate the magnitude difference between each vector and unit vector
    abs_mag_diff = np.abs(mag - 1)

    # if the magnitude is 0, then change the magnitude to 1
    mag[mag == 0] = 1

    # calculate the cosine similarity between each vector and unit vector
    for d in range(dim):
        unit_vec = np.zeros(dim)
        unit_vec[d] = 1
        cos_sim = np.dot(vectors, unit_vec) / mag
        score += np.max(cos_sim)*(1/np.exp(abs_mag_diff[np.argmax(cos_sim)]))

    return score