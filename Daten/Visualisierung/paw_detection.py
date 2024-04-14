import logging
import traceback
import warnings
import numpy as np
import skimage.measure

from scipy.ndimage import label

import mylib


class Dog(object):
    def __init__(self, front_left, front_right, back_left, back_right):
        self.paws = []
        self.paws.append(front_left)
        self.paws.append(front_right)
        self.paws.append(back_left)
        self.paws.append(back_right)

        self.local_mx = np.array([[]])
        self.offset = []
        self.labeled_mx = np.array([[]])

        logging.getLogger(__name__)


class Paw(object):
    def __init__(self, ax_ind, name=None, ground=False):
        self.start_index = (-1, -1)
        self.global_pos = (-1, -1)
        self.area = [[]]
        self.labeled_area = [[]]
        self.ground = ground
        self.ax_ind = ax_ind
        self.name = name
        self.lastContact = 0  # time steps since last time paw touched ground
        self.set_since = 0
        self.time = -1
        self.props = None
        self.valid = True

    def lift(self):
        self.ground = False
        self.area = [[]]
        self.labeled_area = [[]]
        self.start_index = (-1, -1)
        self.global_pos = (-1, -1)
        self.lastContact += 1
        self.set_since = 0
        self.props = None

    def touch(self, start_index, time):
        self.ground = True
        self.start_index = start_index
        self.area, self.labeled_area = get_paw_area(TheDog.labeled_mx[start_index])
        self.global_pos = calc_global_pos(start_index)
        self.lastContact = 0
        self.set_since += 1
        self.props = skimage.measure.regionprops(self.labeled_area, self.area)
        self.time = time


A = Paw((0, 0), 'A')
B = Paw((0, 1), 'B')
C = Paw((1, 0), 'C')
D = Paw((1, 1), 'D')
TheDog = Dog(A, B, C, D)


def paw_recognition(matrix, local_mx_offset, ctr):
    """
    detects number of paws on the mat. Areas with less than [_] cells between each other are considered one paw.
    :param matrix: local matrix from data set
    :param local_mx_offset: offset of local matrix
    :param ctr: number of matrix that is passed
    """

    if matrix.any():
        paws, start_ind, labeled_mx = find_nzero_clusters(matrix, mylib.NEIGHBOR_DIST)
        paw_count = len(paws)

        TheDog.local_mx = matrix
        TheDog.offset = local_mx_offset
        TheDog.labeled_mx = labeled_mx

        if paw_count > 4:
            warnings.warn('Dog has too many paws!')

        for paw in TheDog.paws:
            start = compare_glob_pos(paw, start_ind)
            if type(paw) is Paw:
                try:
                    if start in start_ind:  # found prev. paw start
                        glob_pos = calc_global_pos(start)  # debugging
                        paw.touch(start, ctr)
                        start_ind.remove(start)
                    else:
                        paw.lift()
                except TypeError as e:
                    raise Exception('at %i paw(s)' % paw_count) from e

        for unused_start in start_ind:
            get_max_airborne_paw().touch(unused_start, ctr)

    else:
        for paw in TheDog.paws:
            paw.lift()

    return TheDog.paws


def compare_glob_pos(paw_obj, start_ind):  # param: one paw, all new paw areas
    # 'traceback' of prev. used areas on the mat to the corresponding paw that has been in that area
    nbh = mylib.NEIGHBOR_DIST
    if start_ind:
        for start in start_ind:
            dist = np.subtract(paw_obj.global_pos, calc_global_pos(start))
            if all(abs(val) <= nbh for val in dist):
                return start
    return -1, -1


def calc_global_pos(start_ind):
    y_off, x_off = TheDog.offset
    local_r, local_c = TheDog.local_mx.shape
    x = 481 - 1 - x_off - local_r + start_ind[0]
    if x < 0: x = 0
    y = y_off - 1 + start_ind[1]
    if y < 0: y = 0
    return x, y


def get_paw_area(paw_cluster):  # return paw area without empty rows or col
    assert TheDog.local_mx.shape == TheDog.labeled_mx.shape
    assert paw_cluster
    mask = TheDog.labeled_mx == paw_cluster
    ret_mx = TheDog.local_mx.copy()
    ret_mx[~mask] = 0

    non_zero_rows = [i for i, row in enumerate(ret_mx) if any(row)]
    non_zero_cols = [i for i, col in enumerate(ret_mx.transpose()) if any(col)]

    ret_mx = ret_mx[min(non_zero_rows):max(non_zero_rows) + 1, min(non_zero_cols):max(non_zero_cols) + 1]
    labeled_area = ret_mx > 0

    return ret_mx, labeled_area.astype(int)


def find_nzero_clusters(matrix, neighbor_distance):  # values in matrix are used to find neighbouring (also diag.) elems
    labeled_matrix, num_clusters = label(matrix, structure=adj)

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
            for index, nb in np.ndenumerate(subset):  # group diff. clusters in one when they are in nbh-dist.
                if nb > cluster: subset[index] = cluster
        except IndexError:
            warnings.warn('out of bounds at neighborhood check\n')
            print(traceback.format_exc())

    uniques, flat_indices = np.unique(labeled_matrix, return_index=True)
    indices = []
    for i in flat_indices:
        tmp_ind = np.unravel_index(i, labeled_matrix.shape)  # left uppermost index of area
        if labeled_matrix[tmp_ind] != 0:  # not include 0-cluster since it is not a paw area
            indices.append(tmp_ind)
    # print('\ndetected paw area(s):\n', labeled_matrix)
    paws = uniques
    if len(uniques) > 1:
        paws = np.delete(uniques, 0)  # 'paw'/cluster with 0 is not needed
    return paws, indices, labeled_matrix


def get_max_airborne_paw(only_front=False):  # returns paw obj. with highest time since last ground contact
    longest_air_paw = None
    max_t = -np.inf
    for paw_obj in TheDog.paws:
        if only_front:
            if paw_obj.lastContact > max_t and paw_obj.is_front:
                longest_air_paw = paw_obj
                max_t = paw_obj.lastContact
        else:
            if paw_obj.lastContact > max_t:
                longest_air_paw = paw_obj
                max_t = paw_obj.lastContact

    return longest_air_paw


def valid_data(global_mx, th_sides=1,
               th_top=1):  # discard data if values are too close to the edge/ends of mat
    # th == threshold
    if not global_mx.any():  # empty global mx is valid
        return True
    # top = global_mx[0:th_top].flat
    # bot = global_mx[-1:-th_top - 1:-1].flat
    left = global_mx[:, 0:th_sides].flat
    right = global_mx[:, -1:-th_sides - 1:-1].flat
    combined = list(np.concatenate([left, right]))  # top, bot,
    if np.any(combined):
        return False
    else:
        return True


adj = np.ones((3, 3))  # structure element defining neighborhood dist. in matrix
