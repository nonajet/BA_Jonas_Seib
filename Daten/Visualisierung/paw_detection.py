import logging
import traceback
import warnings
import numpy as np

from scipy.ndimage import label

import feature_creation
import mylib


class Dog(object):
    def __init__(self, front_left, front_right, back_left, back_right):
        self.paws = []
        self.paws.append(front_left)
        self.paws.append(front_right)
        self.paws.append(back_left)
        self.paws.append(back_right)

        self.local_mx = np.array([[]])
        self.prev_offset = []
        self.offset = []
        self.labeled_mx = np.array([[]])
        self.prev_global_mx = np.array([[]])
        self.global_mx = np.array([[]])
        self.prev_front_most_ind = (-1, -1)
        self.front_most_ind = (-1, -1)

        self.newly_planted_paws = set()

        logging.getLogger(__name__)


class Paw(object):
    def __init__(self, start_index, area, ax_ind, name, ground=False):
        self.start_index = start_index
        self.global_pos = (-1, -1)
        self.area = area
        self.ground = ground
        self.ax_ind = ax_ind
        self.name = name
        self.lastContact = 0  # time steps since last time paw touched ground
        self.set_since = 0

    def lift(self):
        self.ground = False
        self.area = [[]]
        self.start_index = (-1, -1)
        self.global_pos = (-1, -1)
        self.lastContact += 1
        self.set_since = 0

    def touch(self, start_index):
        self.ground = True
        self.start_index = start_index
        self.area = get_paw_area(TheDog.labeled_mx[start_index])
        self.global_pos = calc_global_pos(start_index)
        self.lastContact = 0
        self.set_since += 1
        TheDog.newly_planted_paws.add(self)


FL = Paw((-1, -1), [[]], (0, 0), 'fl')
FR = Paw((-1, -1), [[]], (0, 1), 'fr')
BL = Paw((-1, -1), [[]], (1, 0), 'bl')
BR = Paw((-1, -1), [[]], (1, 1), 'br')
TheDog = Dog(FL, FR, BL, BR)


def paw_recognition(matrix, local_mx_offset, global_mx):
    """
    detects number of paws on the mat. Areas with less than [_] cells between each other are considered one paw.
    :param matrix: local matrix from data set
    :param local_mx_offset: offset of local matrix
    :param global_mx: whole mat as matrix
    """
    paws, start_ind, labeled_mx = find_nzero_clusters(matrix, mylib.NEIGHBOR_DIST)
    paw_count = len(paws)
    print('start_ind:', start_ind)
    print(paw_count, 'paw(s)')

    TheDog.local_mx = matrix
    TheDog.prev_offset = TheDog.offset
    TheDog.offset = local_mx_offset
    TheDog.labeled_mx = labeled_mx
    TheDog.prev_global_mx = TheDog.global_mx
    TheDog.global_mx = global_mx
    TheDog.prev_front_most_ind = TheDog.front_most_ind
    TheDog.front_most_ind = min(start_ind)
    TheDog.newly_planted_paws = set()

    if len(paws) != len(start_ind):
        warnings.warn('no. of paws %i does not match no. of paw starting points:' % paw_count)
        print('paws:', paws, '\nstart_ind:', start_ind)

    if paw_count == 1 and mylib.GAIT_TYPE == 0 and not mylib.three_paws:
        return
    elif paw_count == 1 and mylib.GAIT_TYPE == 1 and not mylib.two_paws:  # for Trab wait for at least 2 paws to start
        return

    paw_on_ground = None  # only relevant for 1 paw case
    paws_planted = 0
    # default for 1 - 3 paws ('backtracing')
    for paw_obj in get_active_paws():
        try:
            start = compare_glob_pos(paw_obj, start_ind)
            if start in start_ind and type(paw_obj) is Paw:
                paw_obj.touch(start)  # TODO: dynamic call?
                paws_planted += 1  # keeps track of no. of backtraced paws found in this iteration
                paw_on_ground = paw_obj
        except TypeError as e:
            raise Exception('at %i paw(s)' % paw_count) from e

    if paw_count == 1 and not paw_on_ground:
        print('no paw at', start_ind[0])
        paw_on_ground = get_max_airborne_paw(True)  # favors front paw when last_contact is tied
        if paw_on_ground:
            paw_on_ground.touch(start_ind[0])  # TODO: declaration not found
            print('code not broken yet...guessed new single paw')

        lift_other_paws([paw_on_ground])

    elif paw_count == 2 and paws_planted < paw_count:
        front = min(start_ind)  # first distinction b/w front <=> back paw (smaller row means front)
        back = max(start_ind)
        if mylib.GAIT_TYPE:  # Trab
            if front[1] < back[1]:  # second dist. b/w left <=> right depending on what start_index is more left
                FL.touch(front)  # TODO: test for already planted paws and missing paws
                BR.touch(back)
            else:
                FR.touch(front)
                BL.touch(back)

        else:  # Schritt
            # TODO: should not work correctly yet
            if FR.ground or BR.ground:  # accurate dist. possible (since paw still on ground from timestep before)
                FR.touch(front)
                BR.touch(back)
            elif FL.ground or BL.ground:  # accurate dist. possible -> just 'update' already set paws
                FL.touch(front)
                BL.touch(back)
            else:  # guess needed at first
                guess_paw = get_max_airborne_paw()
                corr_paw = get_corresponding_paw(guess_paw, 0)
                assert guess_paw, corr_paw

                guess_paw.touch(front)
                corr_paw.touch(back)
                print('side guess (Schritt)')

                if guess_paw.name == 'fl' or guess_paw.name == 'bl':
                    FR.lift()
                    BR.lift()
                else:
                    FL.lift()
                    BL.lift()

        if not mylib.two_paws:
            mylib.two_paws = True  # only relevant for first time with 2 paws on ground (Trab)
    elif paw_count == 3 and paws_planted < paw_count:
        front = min(start_ind)
        mid = start_ind[1]
        back = max(start_ind)

        if mylib.GAIT_TYPE:  # Trab
            active_paw_start_indices = [act_paw.start_index for act_paw in get_active_paws()]
            for start in start_ind:
                if start not in active_paw_start_indices:
                    if start == front:  # set 'new' front paw
                        if front[1] < mid[1]:
                            FL.touch(start)
                        else:
                            FR.touch(start)
                        break
                    else:
                        try:
                            next_paw = get_max_airborne_paw()
                            if next_paw:
                                next_paw.touch(mid)
                        except TypeError as e:
                            raise Exception('at 3 paws') from e

        else:  # Schritt
            if front[1] < mid[1]:  # left side
                FL.touch(front)
                BL.touch(back)
                # TODO FR.lift()? or BR? is front paw really first while Schritt gait?

                if not mylib.three_paws:  # first time 3 paws on mat the 2nd front paw is guessed
                    FR.touch(mid)
                else:
                    next_paw = get_max_airborne_paw()
                    next_paw.touch(mid)
                    # get_corresponding_paw(next_paw, 0).lift()

            else:  # right side
                FR.touch(front)
                BR.touch(back)

                if not mylib.three_paws:
                    FL.touch(mid)
                else:
                    next_paw = get_max_airborne_paw()
                    next_paw.touch(mid)
                    # get_corresponding_paw(next_paw, 0).lift()

        if not mylib.three_paws:
            mylib.three_paws = True  # only relevant for first time with 3 paws on ground
    elif paw_count == 4:
        front = min(start_ind)
        back = max(start_ind)
        if front[1] < back[1]:
            FL.touch(front)
            BR.touch(back)
            FR.touch(start_ind[1])
            BL.touch(start_ind[2])
        else:
            FR.touch(front)
            BL.touch(back)
            FL.touch(start_ind[1])
            BR.touch(start_ind[2])
    elif paw_count > 4:
        warnings.warn('Dog has too many paws!')
    else:
        print('only backtracing')

    lift_other_paws(TheDog.newly_planted_paws)
    # TODO: call paw_processing
    feature_creation.save_paws(TheDog.paws)


def compare_glob_pos(paw_obj, start_ind):  # param: one paw, all new paw areas
    # 'traceback' of prev. used areas on the mat to the corresponding paw that has been in that area
    nbh = mylib.NEIGHBOR_DIST
    assert start_ind
    for start in start_ind:
        dist = np.subtract(paw_obj.global_pos, calc_global_pos(start))
        if all(abs(val) <= nbh for val in dist):
            # print('glob_pos:', calc_global_pos(start), 'for', start)
            # print('with dist:', dist)
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


def lift_other_paws(newly_planted_paws):
    for paw_obj in TheDog.paws:
        try:
            if paw_obj not in newly_planted_paws:
                paw_obj.lift()
        except TypeError as e:
            raise Exception('paw could not be lifted') from e


def get_active_paws():
    paw_list = []
    for paw_obj in TheDog.paws:
        try:
            if paw_obj.ground:
                paw_list.append(paw_obj)
        except TypeError as e:
            raise Exception('paw_obj not found') from e
    return paw_list


def get_paw_area(paw_cluster):  # return paw area without empty rows or col
    assert TheDog.local_mx.shape == TheDog.labeled_mx.shape
    # print('cluster:\n', paw_cluster)
    assert paw_cluster
    mask = TheDog.labeled_mx == paw_cluster
    ret_mx = TheDog.local_mx.copy()
    ret_mx[~mask] = 0

    non_zero_rows = np.any(ret_mx != 0, axis=1)
    start_index = np.argmax(non_zero_rows)
    end_index = len(TheDog.local_mx) - np.argmax(non_zero_rows[::-1])

    non_zero_cols = np.any(ret_mx != 0, axis=0)
    non_zero_cols_indices = np.where(non_zero_cols)[0]
    return ret_mx[start_index:end_index, non_zero_cols_indices]


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


def is_nonzero_neighborhood(startindex, neighborhood):
    # TODO: duplicate code in def 'find_nzero_clusters'
    nbh = neighborhood  # determines radius in which paw area is searched
    r, c = startindex
    y_off, x_off = TheDog.offset
    try:
        # only limit lower bounds since slicing handles upper bounds of matrix
        lower_r = r - nbh + x_off if r - nbh >= 0 else 0
        lower_c = c - nbh + y_off if c - nbh >= 0 else 0
        subset = TheDog.prev_global_mx[lower_r:r + nbh + 1, lower_c:c + nbh + 1]
        for index, nb in np.ndenumerate(subset):
            if nb != 0:
                print('at index', index, 'value:', nb)
                print('labeled:\n', TheDog.labeled_mx)
                return True
        return False
    except IndexError:
        warnings.warn('out of bounds at zero-neighborhood check\n')
        print(traceback.format_exc())


def get_max_airborne_paw(only_front=False):  # returns paw obj. with highest time since last ground contact
    longest_air_paw = None
    max_t = -np.inf
    for paw_obj in TheDog.paws:
        if only_front:
            if paw_obj.lastContact > max_t and 'f' in paw_obj.name:
                longest_air_paw = paw_obj
                max_t = paw_obj.lastContact
        else:
            if paw_obj.lastContact > max_t:
                longest_air_paw = paw_obj
                max_t = paw_obj.lastContact

    return longest_air_paw


def get_corresponding_paw(paw,
                          gait_type):  # returns diagonal or same-side paw that is next to touch ground for gait_type
    assert paw
    if gait_type:  # Trab
        if paw.name == 'fl':
            return BR
        elif paw.name == 'fr':
            return BL
        elif paw.name == 'bl':
            return FR
        elif paw.name == 'br':
            return FL
        else:
            print('wrong parameter for paw')
            return None
    else:  # Schritt
        if paw.name == 'fl':
            return BL
        elif paw.name == 'fr':
            return BR
        elif paw.name == 'bl':
            return FL
        elif paw.name == 'br':
            return FR
        else:
            print('wrong parameter for paw')
            return None


adj = np.ones((3, 3))  # structure element defining connection rules between features

if __name__ == '__main__':
    pass
