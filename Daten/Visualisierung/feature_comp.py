import warnings

import numpy as np
from scipy.ndimage import label


class Paw(object):
    pass


def paw_recognition(matrix):
    paws = find_nonzero_clusters(matrix)
    if len(paws) < 0 or len(paws) > 4:
        warnings.warn('Dog has too few or many paws!')
    return len(paws), paws


def find_nonzero_clusters(matrix):  # values in matrix are considered features to find adjacent (also diag.) elems
    labeled_matrix, num_clusters = label(matrix, structure=adj)

    cluster_indices = []

    for cluster_label in range(1, num_clusters + 1):
        cluster_indices_mask = labeled_matrix == cluster_label
        cluster_indices_rows, cluster_indices_cols = np.where(cluster_indices_mask)

        cluster_indices.append((cluster_indices_rows[0], cluster_indices_cols[0]))

    return cluster_indices


adj = np.ones((3, 3))  # structure element defining connection rules between features

# for testing
tmp = np.array([[0, 0, 0, 0, 0, 1.1, 0, 0, 0],
                [0., 0., 2.3, 0, 4., 7.4, 3.4, 0., 0.],
                [0, 0, 0, 0, 6.2, 9.7, 4., 0., 0.],
                [0, 1.1, 5.7, 6.8, 3.4, 4., 2.3, 1.7, 0.6],
                [0, 4.5, 5.7, 1.7, 0., 0., 2.3, 7.4, 2.8],
                [0.6, 7.9, 9.1, 1.1, 1.7, 1.1, 2.3, 5.1, 1.7],
                [0, 3.4, 4., 2.3, 6.2, 4.5, 0.6, 0., 0.],
                [0, 0., 0, 0, 0, 6.2, 0.6, 0., 0.],
                [0, 0., 0, 3.4, 0, 1.7, 0., 0., 0.]])

matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

if __name__ == '__main__':
    print(adj)
    # paw_recognition(tmp, 0)
