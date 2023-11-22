import traceback
import warnings
from pprint import pprint

import numpy as np
from scipy.ndimage import label


class Dog(object):
    pass

    def __init__(self, front_left, front_right, back_left, back_right):
        self.fl = front_left
        self.fr = front_right
        self.bl = back_left
        self.br = back_right


class Paw(object):
    def __init__(self, start_index, front, right, area, ax_ind, ground=False):
        self.start_index = start_index
        # self.front = front
        # self.r = right
        self.area = area
        self.ground = ground
        self.ax_ind = ax_ind

    def lift(self):
        self.ground = False
        self.area = [[]]

    def touch(self, matrix, labeled_mx, start_index):
        self.ground = True
        self.start_index = start_index
        self.area = get_paw_area(matrix, labeled_mx, labeled_mx[start_index])


FL = Paw((-1, -1), True, False, [[]], (0, 0))
FR = Paw((-1, -1), True, True, [[]], (0, 1))
BL = Paw((-1, -1), False, False, [[]], (1, 0))
BR = Paw((-1, -1), False, True, [[]], (1, 1))
TheDog = Dog(FL, FR, BL, BR)


def paw_recognition(matrix):
    """
    detects number of paws on the mat. Areas with less than __ cells between each other are considered one paw.
    :param matrix:
    :return: number of paws and their left uppermost index
    """
    paws, start_ind, labeled_mx = find_nzero_clusters(matrix, 5)
    if len(paws) != len(start_ind): warnings.warn('no. of paws does not match no. of paw starting points')

    paw_count = len(paws)
    print('%i paw(s)' % paw_count)
    print(start_ind)
    if paw_count == 0:  # Trab == diagonal paws
        pass
    elif paw_count == 1:
        pass
    elif paw_count == 2:
        front = min(start_ind)  # first distinction b/w front <=> back paw (smaller row means front)
        back = max(start_ind)
        if front[1] < back[1]:  # second dist. b/w left <=> right depending on what start_index is more left
            FL.touch(matrix, labeled_mx, front)
            FR.lift()

            BR.touch(matrix, labeled_mx, back)
            BL.lift()
        else:
            FR.touch(matrix, labeled_mx, front)
            FL.lift()

            BL.touch(matrix, labeled_mx, back)
            BR.lift()
    elif paw_count == 3:
        front = min(start_ind)
        mid = start_ind[1]
        back = max(start_ind)
        if front[1] < mid[1]:
            FL.touch(matrix, labeled_mx, front)
            BL.touch(matrix, labeled_mx, back)

            # TODO FR.lift()? or BR?
        else:
            FR.touch(matrix, labeled_mx, front)
            BR.touch(matrix, labeled_mx, back)

            # which ones are lifted?
    elif paw_count == 4:
        front = min(start_ind)
        back = max(start_ind)
        if front[1] < back[1]:
            # TODO: new function for updating matrix and labeled_mx for each touch()-call
            FL.touch(matrix, labeled_mx, front)
            BR.touch(matrix, labeled_mx, back)
            FR.touch(matrix, labeled_mx, start_ind[1])
            BL.touch(matrix, labeled_mx, start_ind[2])
        else:
            FR.touch(matrix, labeled_mx, front)
            BL.touch(matrix, labeled_mx, back)
            FL.touch(matrix, labeled_mx, start_ind[1])
            BR.touch(matrix, labeled_mx, start_ind[2])
    else:
        warnings.warn('Dog has too few or many paws!')

    return len(paws)


def get_paw_area(matrix, labeled_mx, paw_cluster):
    assert matrix.shape == labeled_mx.shape
    mask = labeled_mx == paw_cluster
    ret_mx = matrix.copy()
    ret_mx[~mask] = 0
    non_zero_rows = np.any(ret_mx != 0, axis=1)
    start_index = np.argmax(non_zero_rows)
    end_index = len(matrix) - np.argmax(non_zero_rows[::-1])  # Reverse-Index
    non_zero_cols = np.any(ret_mx != 0, axis=0)
    non_zero_cols_indices = np.where(non_zero_cols)[0]
    return ret_mx[start_index:end_index, non_zero_cols_indices]


def find_nzero_clusters(matrix, neighbor_distance):  # values in matrix are used to find neighbouring (also diag.) elems
    labeled_matrix, num_clusters = label(matrix, structure=adj)
    # print('in:', labeled_matrix)

    nzero = np.transpose(np.nonzero(labeled_matrix))

    nbh = neighbor_distance  # set size of neighborhood
    for elem in nzero:
        r = elem[0]
        c = elem[1]
        cluster = labeled_matrix[r, c]
        try:
            # only limit lower bounds since slicing handles upper bounds of matrix
            lower_r = r - nbh if r - nbh >= 0 else 0
            lower_c = c - nbh if c - nbh >= 0 else 0
            subset = labeled_matrix[lower_r:r + nbh + 1, lower_c:c + nbh + 1]
            for index, nb in np.ndenumerate(subset):
                if nb > cluster: subset[index] = cluster
        except IndexError as e:
            warnings.warn('out of bounds at neighborhood check\n')
            print(traceback.format_exc())

    uniques, flat_indices = np.unique(labeled_matrix, return_index=True)
    indices = []
    for i in flat_indices:
        tmp_ind = np.unravel_index(i, labeled_matrix.shape)  # left uppermost index of area
        if labeled_matrix[tmp_ind] != 0:  # not include 0-cluster since it is not a paw area
            indices.append(tmp_ind)
    # print('\ndetected paw area(s):\n', labeled_matrix)
    return np.delete(uniques, 0), indices, labeled_matrix


adj = np.ones((3, 3))  # structure element defining connection rules between features

# for testing
tmparr = np.array([[1, 0, 0, 5, 0, 1.1, 0, 0, 0],
                   [0, 0, 0, 0, 4., 7.4, 3.4, 0., 0.],
                   [0, 0, 0, 0, 6.2, 9.7, 4., 0., 0.],
                   [0, 1.1, 5.7, 6.8, 3.4, 4., 2.3, 1.7, 0.6],
                   [0, 4.5, 5.7, 1.7, 0., 0., 2.3, 7.4, 2.8],
                   [0.6, 7.9, 9.1, 1.1, 1.7, 1.1, 2.3, 5.1, 1.7],
                   [0, 3.4, 4., 2.3, 6.2, 4.5, 0.6, 0., 0.],
                   [0, 0, 0, 0, 0, 6.2, 0.6, 0., 0.],
                   [0, 0, 3, 3.4, 0, 1.7, 0., 0., 0.]])

matrix = np.array([[0.0, 0.0, 1.7, 1.7, 1.7, 6.2, 1.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.6, 12.5, 14.8, 9.1, 22.1, 5.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 1.7, 18.2, 17.0, 10.2, 14.8, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 4.0, 7.9, 5.1, 2.8, 3.4, 2.8, 4.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [2.8, 17.6, 8.5, 0.6, 0.0, 0.0, 6.8, 12.5, 2.3, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [3.4, 19.3, 8.5, 3.4, 5.1, 1.7, 5.1, 7.9, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 3.4, 2.8, 7.4, 11.4, 4.0, 0.0, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 3.4, 8.5, 9.7, 4.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.],
                   [0.0, 0.0, 1.7, 2.3, 1.7, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.],
                   [5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.],
                   [5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1, 6.2, 2.8, 0.6, 0.],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.5, 21.6, 9.7, 9.7, 4.],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1, 4.0, 11.9, 8.5, 15.3, 6.],
                   [0.0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.6, 5.1, 4.5, 2.3, 2.8, 6.2, 2.],
                   [0.0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.6, 5.7, 4.0, 0.0, 0.0, 3.4, 5.],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.6, 5.1, 7.],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 4.0, 1.1, 2.3, 2.],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 2.8, 0.6, 0.0, 0.0]])

if __name__ == '__main__':
    paw_recognition(matrix)
    pprint(vars(FL))
    pass
