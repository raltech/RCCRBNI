import numpy as np
import os

'''
Score func calculates how well the produced dictionary is
Score is between 0 and the number of cells
'''
def span_score_func(vectors):
    # calculate the span of a set of vectors
    # :param vectors: list of vectors
    # :return: span of vectors
    span = np.linalg.matrix_rank(vectors)
    return span

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