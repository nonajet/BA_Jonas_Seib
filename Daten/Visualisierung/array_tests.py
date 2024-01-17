import numpy as np


class Paw(object):
    def __init__(self, name, ground=False):
        self.global_pos = (-1, -1)
        self.ground = ground
        self.name = name


paw_obj1 = Paw('fl')
paw_obj2 = Paw('fr')
paw_obj3 = Paw('bl')
paws = [paw_obj3, paw_obj2, paw_obj1]
paw_dict = {'fl': -1, 'fr': 2, 'bl': 3, 'br': 10}


def get_paw_area(_matrix):  # return paw area without empty rows or col
    ret_mx = _matrix

    non_zero_rows = np.any(ret_mx != 0, axis=1)
    start_index = np.argmax(non_zero_rows)
    end_index = len(ret_mx) - np.argmax(non_zero_rows[::-1])

    non_zero_cols = np.any(ret_mx != 0, axis=0)
    col_start_ind = np.argmax(non_zero_cols)
    col_end_ind = len(ret_mx[1]) - np.argmax(non_zero_cols[::-1])
    return ret_mx[start_index:end_index, col_start_ind:col_end_ind]
