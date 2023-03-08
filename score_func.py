import numpy as np

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
    # calculate the cosine similarity of a set of vectors
    # :param vectors: list of vectors
    # :return: cosine similarity of vectors
    sim = np.dot(vectors, vectors.T)
    return sim