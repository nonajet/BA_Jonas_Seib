import traceback
import warnings

import numpy as np
import settings
from scipy.ndimage import label


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


class Paw(object):
    def __init__(self, start_index, area, ax_ind, name, ground=False):
        self.start_index = start_index
        self.global_pos = (-1, -1)
        self.area = area
        self.ground = ground
        self.ax_ind = ax_ind
        self.name = name
        self.sure = False  # confidence in correct paw recognition
        # TODO maybe lastContact = 1 better
        self.lastContact = 0  # time steps since last time paw touched ground

    def lift(self):
        self.ground = False
        self.area = [[]]
        self.start_index = (-1, -1)
        self.global_pos = (-1, -1)
        self.lastContact += 1

    def touch(self, start_index):
        self.ground = True
        self.start_index = start_index
        self.area = get_paw_area(TheDog.labeled_mx[start_index])
        self.global_pos = calc_global_pos(start_index)
        self.lastContact = 0


# TODO: increase performance by dog init in exctract_xml.py?
FL = Paw((-1, -1), [[]], (0, 0), 'fl')
FR = Paw((-1, -1), [[]], (0, 1), 'fr')
BL = Paw((-1, -1), [[]], (1, 0), 'bl')
BR = Paw((-1, -1), [[]], (1, 1), 'br')
TheDog = Dog(FL, FR, BL, BR)


def compare_glob_pos(paw_obj, start_ind):  # param: one paw, all new paw areas
    nbh = settings.NEIGHBOR_DIST
    assert start_ind
    for start in start_ind:
        dist = np.subtract(paw_obj.global_pos, calc_global_pos(start))
        print('glob_pos:', calc_global_pos(start), 'for', start)
        if all(val <= nbh for val in dist):
            print('dist:', dist)
            return start
    return -1, -1


def calc_global_pos(start_ind):  # TODO: testing
    y_off, x_off = TheDog.offset
    local_r, local_c = TheDog.local_mx.shape
    x = 481 - 1 - x_off - local_r + start_ind[0]
    if x < 0: x = 0
    y = y_off - 1 + start_ind[1]
    if y < 0: y = 0
    return x, y


def paw_recognition(matrix, local_mx_offset, global_mx):
    """
    detects number of paws on the mat. Areas with less than [_] cells between each other are considered one paw.
    :param matrix: local matrix from data set
    :param local_mx_offset: offset of local matrix
    :param global_mx: whole mat as matrix
    :return: number of paws and their left uppermost index
    """
    paws, start_ind, labeled_mx = find_nzero_clusters(matrix, settings.NEIGHBOR_DIST)
    print('start_ind:', start_ind)
    print(len(paws), 'paws')
    TheDog.local_mx = matrix
    TheDog.prev_offset = TheDog.offset
    TheDog.offset = local_mx_offset
    TheDog.labeled_mx = labeled_mx
    TheDog.prev_global_mx = TheDog.global_mx
    TheDog.global_mx = global_mx
    if len(paws) != len(start_ind):
        warnings.warn('no. of paws %i does not match no. of paw starting points:' % len(paws))
        print('paws:', paws, '\nstart_ind:', start_ind)

    paw_count = len(paws)
    if paw_count == 0:
        # settings.three_paws = False  # should only occur at beginning of data set
        print('lift all paws since 0 on mat')
        FL.lift()
        FR.lift()
        BL.lift()
        BR.lift()
    elif paw_count != 3 and settings.GAIT_TYPE == 0 and not settings.three_paws:
        return paw_count
    elif paw_count == 1:
        paw_on_ground = None
        for paw_obj in get_active_paws():
            try:
                start = compare_glob_pos(paw_obj, start_ind)
                if start in start_ind:
                    paw_obj.touch(start)
                    paw_on_ground = paw_obj
            except TypeError as e:
                raise Exception('1 paw') from e

        if not paw_on_ground:
            lift_other_paws([])
            print('no paw before at', start_ind[0])
            return paw_count

        lift_other_paws(paw_on_ground)

    elif paw_count == 2:
        front = min(start_ind)  # first distinction b/w front <=> back paw (smaller row means front)
        back = max(start_ind)
        paws_planted = 0
        if settings.GAIT_TYPE:  # Trab
            # TODO def 'traceback_paws', untraceable areas are newly planted then
            for paw_obj in get_active_paws():
                try:
                    start = compare_glob_pos(paw_obj, start_ind)
                    if start in start_ind:
                        paw_obj.touch(start)
                        paws_planted += 1
                except TypeError as e:
                    raise Exception('2 paws') from e

            if paws_planted < paw_count:
                if front[1] < back[1]:  # second dist. b/w left <=> right depending on what start_index is more left
                    FL.touch(front)
                    FL.sure = True
                    BR.touch(back)
                    BR.sure = True
                    FR.lift()
                    BL.lift()
                else:
                    FR.touch(front)
                    FR.sure = True
                    BL.touch(back)
                    BL.sure = True
                    FL.lift()
                    BR.lift()
        else:  # Schritt
            # TODO: should not work correctly yet
            if FR.ground or BR.ground:  # accurate dist. possible (since paw still on ground from timestep before)
                FR.touch(front)
                FR.sure = True
                BR.touch(back)
                BR.sure = True
                FL.lift()
                BL.lift()
            elif FL.ground or BL.ground:  # accurate dist. possible -> just 'update' already set paws
                FL.touch(front)
                FL.sure = True
                BL.touch(back)
                BL.sure = True
                FR.lift()
                BR.lift()
            else:  # guess needed at first
                guess_paw = get_max_airborne_paw()
                corr_paw = get_corresponding_paw(guess_paw, 0)
                assert guess_paw, corr_paw

                guess_paw.touch(front)
                guess_paw.sure = False
                corr_paw.touch(back)
                corr_paw.sure = False
                print('side guess (Schritt)')

                if guess_paw.name == 'fl' or guess_paw.name == 'bl':
                    FR.lift()
                    BR.lift()
                else:
                    FL.lift()
                    BL.lift()
    elif paw_count == 3:
        front = min(start_ind)
        mid = start_ind[1]
        back = max(start_ind)

        if settings.GAIT_TYPE:  # Trab
            if front[1] < mid[1]:
                FL.touch(front)
                FL.sure = True
                BR.touch(back)
                BR.sure = True

                if not settings.three_paws:  # first time 3 paws on mat the 2nd front paw is guessed
                    FR.touch(mid)
                    FR.sure = False
                    BL.lift()
                    BL.sure = False
                else:
                    try:
                        next_paw = get_max_airborne_paw()
                        if next_paw:
                            next_paw.touch(mid)
                        get_corresponding_paw(next_paw, 1).lift()
                    except TypeError:
                        print(traceback.format_exc())
            else:
                FR.touch(front)
                FR.sure = True
                BL.touch(back)
                BL.sure = True

                if not settings.three_paws:
                    FL.touch(mid)
                    FL.sure = False
                    BR.lift()
                    BR.sure = False
                else:
                    try:
                        next_paw = get_max_airborne_paw()
                        if next_paw:
                            next_paw.touch(mid)
                        get_corresponding_paw(next_paw, 0).lift()
                    except TypeError as e:
                        raise Exception('3 paws') from e
        else:  # Schritt
            if front[1] < mid[1]:  # left side
                FL.touch(front)
                FL.sure = True
                BL.touch(back)
                BL.sure = True
                # TODO FR.lift()? or BR? is front paw really first while Schritt gait?

                if not settings.three_paws:  # first time 3 paws on mat the 2nd front paw is guessed
                    FR.touch(mid)
                    FR.sure = False
                    BR.lift()
                    BR.sure = False
                else:
                    next_paw = get_max_airborne_paw()
                    next_paw.touch(mid)
                    get_corresponding_paw(next_paw, 0).lift()

            else:  # right side
                FR.touch(front)
                FR.sure = True
                BR.touch(back)
                BR.sure = True

                if not settings.three_paws:
                    FL.touch(mid)
                    FL.sure = False
                    BL.lift()
                    BL.sure = False
                else:
                    next_paw = get_max_airborne_paw()
                    next_paw.touch(mid)
                    get_corresponding_paw(next_paw, 0).lift()

        if not settings.three_paws:
            settings.three_paws = True  # only relevant for first time with 3 paws on ground
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
    else:
        warnings.warn('Dog has too few or many paws!')

    # if not paw_count == 0:
    #     calc_glob_poss()

    return paw_count


def lift_other_paws(newly_planted_paws):
    for paw_obj in TheDog.paws:
        try:
            if paw_obj is not newly_planted_paws:
                paw_obj.lift()
        except TypeError as e:
            raise Exception('paw not lifted') from e


# def calc_glob_poss():
#     global_r, global_c = TheDog.global_mx.shape
#     local_r, local_c = TheDog.local_mx.shape
#     for paw in TheDog.paws:
#         for i in range(global_r - local_r + 1):
#             for j in range(global_c - local_c + 1):
#                 try:
#                     if TheDog.global_mx[i:i + local_r][j:j + local_c] == TheDog.local_mx:
#                         paw.global_pos = tuple([i, j])
#                 except ValueError as e:
#                     raise Exception('error in global pos. computing') from e
#         paw.global_pos = (-1, -1)


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


def get_max_airborne_paw():  # returns paw obj. with highest time since last ground contact
    longest_air_paw = FL
    for paw_obj in TheDog.paws:
        try:
            if paw_obj.lastContact < longest_air_paw.lastContact:
                longest_air_paw = paw_obj
        except TypeError:
            print(traceback.format_exc())
            print('err in:', paw_obj.area)

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
    lift_other_paws([])
