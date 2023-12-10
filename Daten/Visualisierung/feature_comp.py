import traceback
import warnings
from pprint import pprint

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

        self.local_mx = [[]]
        self.offset = []
        self.labeled_mx = [[]]


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

    def touch(self, matrix, labeled_mx, start_index):
        self.ground = True
        self.start_index = start_index
        self.area = get_paw_area(matrix, labeled_mx, labeled_mx[start_index])
        self.global_pos = self._get_global_pos()
        self.lastContact = 0

    def _get_global_pos(self):
        print(self.start_index + TheDog.offset)
        return self.start_index + TheDog.offset


FL = Paw((-1, -1), [[]], (0, 0), 'fl')
FR = Paw((-1, -1), [[]], (0, 1), 'fr')
BL = Paw((-1, -1), [[]], (1, 0), 'bl')
BR = Paw((-1, -1), [[]], (1, 1), 'br')
TheDog = Dog(FL, FR, BL, BR)


def paw_recognition(matrix, local_mx_offset):
    print('\n##########new#########\n')
    """
    detects number of paws on the mat. Areas with less than [_] cells between each other are considered one paw.
    :param matrix:
    :return: number of paws and their left uppermost index
    """
    paws, start_ind, labeled_mx = find_nzero_clusters(matrix, 5)
    if len(paws) != len(start_ind):
        warnings.warn('no. of paws does not match no. of paw starting points:\n %i paw(s) <->' % len(paws))
        print(paws)
        print(start_ind)

    paw_count = len(paws)
    # print('starts:', start_ind)
    # print('mx:', matrix)
    if paw_count == 0:
        # settings.three_paws = False  # should only occur at beginning of data set
        FL.lift()
        FL.sure = False
        FR.lift()
        FR.sure = False
        BL.lift()
        BL.sure = False
        BR.lift()
        BR.sure = False
    elif paw_count != 3 and settings.GAIT_TYPE == 0 and not settings.three_paws:
        return paw_count
    elif paw_count == 1:
        # TODO: def 'find_paw_by_start_ind' and remove for-loop
        paw_on_ground = find_closest_paw(matrix, labeled_mx)
        if not paw_on_ground:
            return paw_count
        assert paw_on_ground
        paw_on_ground.touch(matrix, labeled_mx, start_ind)
        # TODO: create def for 'lift_others(param=newly_planted_paws)'
        for paw_name, paw_obj in vars(TheDog).items():  # lift other paws that are not on mat (anymore)
            try:
                if paw_obj is not paw_on_ground:
                    paw_obj.lift()
            except TypeError:
                print(traceback.format_exc())

    elif paw_count == 2:
        front = min(start_ind)  # first distinction b/w front <=> back paw (smaller row means front)
        back = max(start_ind)
        if settings.GAIT_TYPE:  # Trab
            for paw_name, paw_obj in vars(TheDog).items():  # lift other paws that are not on mat (anymore)
                try:
                    if paw_obj is is_nonzero_neighborhood(labeled_mx, paw_obj.start_index, 4):
                        paw_obj.touch(matrix, labeled_mx,
                                      front)  # front is placeholder; return from is_nonzero... needed
                except TypeError:
                    print(traceback.format_exc())

            if front[1] < back[1]:  # second dist. b/w left <=> right depending on what start_index is more left
                FL.touch(matrix, labeled_mx, front)
                FL.sure = True
                BR.touch(matrix, labeled_mx, back)
                BR.sure = True
                FR.lift()
                BL.lift()
            else:
                FR.touch(matrix, labeled_mx, front)
                FR.sure = True
                BL.touch(matrix, labeled_mx, back)
                BL.sure = True
                FL.lift()
                BR.lift()
        else:  # Schritt
            # TODO: should not work correctly yet
            if FR.ground or BR.ground:  # accurate dist. possible (since paw still on ground from timestep before)
                FR.touch(matrix, labeled_mx, front)
                FR.sure = True
                BR.touch(matrix, labeled_mx, back)
                BR.sure = True
                FL.lift()
                BL.lift()
            elif FL.ground or BL.ground:  # accurate dist. possible -> just 'update' already set paws
                FL.touch(matrix, labeled_mx, front)
                FL.sure = True
                BL.touch(matrix, labeled_mx, back)
                BL.sure = True
                FR.lift()
                BR.lift()
            else:  # guess needed at first
                guess_paw = get_max_airborne_paw()
                corr_paw = get_corresponding_paw(guess_paw, 0)
                assert guess_paw, corr_paw

                guess_paw.touch(matrix, labeled_mx, front)
                guess_paw.sure = False
                corr_paw.touch(matrix, labeled_mx, back)
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
                FL.touch(matrix, labeled_mx, front)
                FL.sure = True
                BR.touch(matrix, labeled_mx, back)
                BR.sure = True

                if not settings.three_paws:  # first time 3 paws on mat the 2nd front paw is guessed
                    FR.touch(matrix, labeled_mx, mid)
                    FR.sure = False
                    BL.lift()
                    BL.sure = False
                else:
                    try:
                        next_paw = get_max_airborne_paw()
                        if next_paw:
                            next_paw.touch(matrix, labeled_mx, mid)
                        get_corresponding_paw(next_paw, 1).lift()
                    except TypeError:
                        print(traceback.format_exc())
            else:
                FR.touch(matrix, labeled_mx, front)
                FR.sure = True
                BL.touch(matrix, labeled_mx, back)
                BL.sure = True

                if not settings.three_paws:
                    FL.touch(matrix, labeled_mx, mid)
                    FL.sure = False
                    BR.lift()
                    BR.sure = False
                else:
                    try:
                        next_paw = get_max_airborne_paw()
                        if next_paw:
                            next_paw.touch(matrix, labeled_mx, mid)
                        get_corresponding_paw(next_paw, 0).lift()
                    except TypeError:
                        print(traceback.format_exc())
        else:  # Schritt
            if front[1] < mid[1]:  # left side
                FL.touch(matrix, labeled_mx, front)
                FL.sure = True
                BL.touch(matrix, labeled_mx, back)
                BL.sure = True
                # TODO FR.lift()? or BR? is front paw really first while Schritt gait?

                if not settings.three_paws:  # first time 3 paws on mat the 2nd front paw is guessed
                    FR.touch(matrix, labeled_mx, mid)
                    FR.sure = False
                    BR.lift()
                    BR.sure = False
                else:
                    next_paw = get_max_airborne_paw()
                    next_paw.touch(matrix, labeled_mx, mid)
                    get_corresponding_paw(next_paw, 0).lift()

            else:  # right side
                FR.touch(matrix, labeled_mx, front)
                FR.sure = True
                BR.touch(matrix, labeled_mx, back)
                BR.sure = True

                if not settings.three_paws:
                    FL.touch(matrix, labeled_mx, mid)
                    FL.sure = False
                    BL.lift()
                    BL.sure = False
                else:
                    next_paw = get_max_airborne_paw()
                    next_paw.touch(matrix, labeled_mx, mid)
                    get_corresponding_paw(next_paw, 0).lift()

        if not settings.three_paws:
            settings.three_paws = True  # only relevant for first time with 3 paws on ground
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

    return paw_count


def find_closest_paw(matrix, labeled_mx):
    active_paws = get_active_paws()  # only consider paws on ground as active
    if not active_paws:
        return None
    assert active_paws
    start_indices = []
    for paw in active_paws:
        start_indices.append(paw.start_index)

    non_zero_index = np.transpose(np.nonzero(matrix))[0]  # Index des 1. Nicht-Null-Elements
    print('nz_ind:', non_zero_index)
    # min_distance = np.inf
    # closest_index = None
    #
    # for start_index in start_indices:
    #     distance = np.linalg.norm(np.array(start_index) - np.array(non_zero_index))
    #     print('dist: ', distance)
    #     if distance < min_distance:
    #         min_distance = distance
    #         closest_index = start_index
    # for paw in active_paws:
    #     print('closest:', type(closest_index), '\npaw:', type(paw.start_index))
    #     print('closest:', closest_index, '\npaw:', paw.start_index)
    #     if closest_index[0] == paw.start_index[0] and closest_index[1] == paw.start_index[1]:
    #         return paw

    for paw_name, paw_obj in vars(TheDog).items():  # TODO: testing
        try:
            if paw_obj.ground and is_nonzero_neighborhood(labeled_mx, paw_obj.start_index, 4):
                # paw_obj.touch(matrix, labeled_mx, non_zero_index)
                return paw_obj
        except TypeError:
            print(traceback.format_exc())
            print('no closest paw for touching paw found')
            return None

    print('no closest paw for touching paw found')
    return None


def get_active_paws():
    paw_list = []
    for paw_name, paw_obj in vars(TheDog).items():
        try:
            if paw_obj.ground:
                paw_list.append(paw_obj)
        except TypeError:
            print(traceback.format_exc())
    return paw_list


def get_paw_area(matrix, labeled_mx, paw_cluster):  # return paw area without empty rows or col
    assert matrix.shape == labeled_mx.shape
    print('cluster:', paw_cluster)
    assert paw_cluster.all()
    mask = labeled_mx == paw_cluster
    ret_mx = matrix.copy()
    ret_mx[~mask] = 0

    non_zero_rows = np.any(ret_mx != 0, axis=1)
    start_index = np.argmax(non_zero_rows)
    end_index = len(matrix) - np.argmax(non_zero_rows[::-1])

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
            for index, nb in np.ndenumerate(subset):
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
    return np.delete(uniques, 0), indices, labeled_matrix


def is_nonzero_neighborhood(labeled_matrix, startindex, neighborhood):
    nbh = neighborhood  # determines radius in which paw area is searched
    r, c = startindex
    try:
        # only limit lower bounds since slicing handles upper bounds of matrix
        lower_r = r - nbh if r - nbh >= 0 else 0
        lower_c = c - nbh if c - nbh >= 0 else 0
        subset = labeled_matrix[lower_r:r + nbh + 1, lower_c:c + nbh + 1]
        for index, nb in np.ndenumerate(subset):
            if nb != 0:
                print('index:', index)
                print('labeled:', labeled_matrix)
                return True
        return False
    except IndexError:
        warnings.warn('out of bounds at zero-neighborhood check\n')
        print(traceback.format_exc())


def get_max_airborne_paw():  # returns paw obj. with highest time since last ground contact
    longest_air_paw = FL
    for paw_name, paw_obj in vars(TheDog).items():
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
    paw_recognition(matrix)
    pprint(vars(TheDog).items())
    pass
