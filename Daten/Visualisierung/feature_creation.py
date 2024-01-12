import copy
import logging

import numpy as np

from Daten.Visualisierung.mylib import get_dog_log

UNINIT_VAL = ''  # default value for not yet initialised paws (order)
raw_paw_order = []
cleaned_paw_order = []


class Paw(object):

    def __init__(self):
        self.peak_pressure = {}


def save_paws(dog_paws):
    assert dog_paws
    # log_namestring = get_dog_log()
    # logger = logging.getLogger(log_namestring)
    # for paw_obj in dog_paws:
    #     if paw_obj:
    #         logger.info(f"{'name'}: {paw_obj.name}")
    #         logger.info(f"{'on_ground'}: {paw_obj.ground}")
    #         logger.info(f"{'last_contact'}: {paw_obj.lastContact}")
    #         logger.info(f"{'set_since'}: {paw_obj.set_since}")
    #         logger.info(f"{'glob_pos'}: {paw_obj.global_pos}")
    # logger.info(paw_obj.area)

    active_paws = [copy.copy(paw) for paw in dog_paws if paw.ground]
    raw_paw_order.append(active_paws)


def restart_order_with(new_start_paw):
    assert new_start_paw
    paw_order = [UNINIT_VAL] * 4
    paw_order[0] = new_start_paw
    return paw_order


def get_changed_paws(t_step):
    try:
        prev_active_paws = {paw.name for paw in raw_paw_order[t_step - 1]}
        active_paws = {paw.name for paw in raw_paw_order[t_step]}
        if t_step <= 1:
            return list(active_paws), []  # very first paws to touch mat are newly planted of course
        else:
            lifted_paws = list(prev_active_paws - active_paws)
            planted_paws = list(active_paws - prev_active_paws)
            return planted_paws, lifted_paws
    except IndexError:
        return [], []


def get_max_valid_seq(doubted_paws):
    max_seq_size = -1
    start_ind = 0
    end_ind = 0
    for key in doubted_paws.keys():
        start_ind = end_ind
        end_ind = key
        if end_ind - start_ind > max_seq_size:
            max_seq_size = end_ind - start_ind
    return start_ind, end_ind


def paw_validation():  # validates and creates dataset of validated paw steps
    t_step = 1  # time step
    tot_steps = 0  # total no. of steps
    paws_in_order = [UNINIT_VAL] * 4
    doubtful_paws = {}
    orders = {}
    # planted & lifted both in comparison to time step before
    for paws_on_ground in raw_paw_order:
        paws_planted, paws_lifted = get_changed_paws(t_step)
        for new_paw in paws_planted:
            paw_turn = tot_steps % 4
            if UNINIT_VAL in paws_in_order:  # some yet uninit. values for paw order -> set order first
                if paws_in_order[paw_turn] == UNINIT_VAL and new_paw not in paws_in_order:
                    paws_in_order[paw_turn] = new_paw
                else:  # paw already had order no. assigned
                    doubtful_paws[t_step] = new_paw
                    orders[t_step] = paws_in_order
                    paws_in_order = restart_order_with(new_paw)
                    tot_steps = 0
            else:  # order already init.
                ex_next_paw = paws_in_order[tot_steps % 4]
                if ex_next_paw != new_paw:
                    print('other paw than expected at t=', t_step)
                    doubtful_paws[t_step] = new_paw
                    orders[t_step] = paws_in_order
                    paws_in_order = restart_order_with(new_paw)
                    tot_steps = 0
            tot_steps += 1
        t_step += 1
    print(doubtful_paws)
    print(orders)

    if doubtful_paws:  # irregular paw order detected
        global cleaned_paw_order
        t_min, t_max = get_max_valid_seq(doubtful_paws)  # TODO: 'backpropagation' of order onto faulty sequences
        cleaned_paw_order = raw_paw_order[0:t_max]
    else:
        cleaned_paw_order = raw_paw_order


def calc_features():
    logger = logging.getLogger(get_dog_log())
    logger.info(peak_pressure())


def peak_pressure():
    pressures = {}
    maximums = {'fl': -1, 'fr': -1, 'bl': -1, 'br': -1}
    for paws_on_ground in cleaned_paw_order:
        paw_names = [paw.name for paw in paws_on_ground]
        for key in maximums.keys():
            if key not in paw_names and maximums[key] != -1:  # reset max of non-touch. paws to def. after saving
                pressures[key] = maximums
                maximums[key] = -1
        for paw in paws_on_ground:
            tmp_max = np.amax(paw.area)
            if tmp_max > maximums[paw.name]:
                maximums[paw.name] = tmp_max

    return pressures
