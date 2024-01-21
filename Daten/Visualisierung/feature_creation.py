import copy
import math
import warnings
from collections import defaultdict

import numpy as np


class FeatureContainer(object):

    def __init__(self):
        self.cleaned_paw_order_names = []

        self.no_of_steps = {}
        self.peak_pressure = {}
        self.step_lengths = {}  # in m

        self.step_start_glob_paw_positions = {}
        self.peak_glob_positions = {}
        self.step_lift_glob_paw_positions = {}

        self.step_start_times = {}
        self.peak_times = {}
        self.step_lift_times = {}
        self.step_contact_durations = {}


UNINIT_VAL = ''  # default value for not yet initialised paws (order)
CELL_SIZE_MM = 8.4688  # in mm
FREQ = 200
FC = FeatureContainer()
raw_paw_order = []


def calc_features():
    FC.cleaned_paw_order_names = paw_validation()
    FC.no_of_steps, FC.step_start_times, FC.step_lift_times = count_steps()
    FC.step_start_glob_paw_positions = calc_glob_pos_of_steps()
    _, FC.step_contact_durations = step_contact_time()
    FC.peak_pressure, FC.peak_times, FC.peak_glob_positions = peak_pressure()
    FC.step_lengths = step_length()

    # logger = logging.getLogger(get_dog_log())
    # logger.info(peak_pressure())


def save_paws(dog_paws):
    assert dog_paws
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

    if doubtful_paws:  # irregular paw order detected
        t_min, t_max = get_max_valid_seq(doubtful_paws)  # TODO: 'backpropagation' of order onto faulty sequences
        cleaned_paw_order = raw_paw_order[t_min:t_max]
    else:
        cleaned_paw_order = raw_paw_order

    return cleaned_paw_order


def calc_glob_pos_of_steps():
    glob_pos = defaultdict(list)
    for key, val_list in FC.step_start_times.items():
        for time in val_list:
            for paw in FC.cleaned_paw_order_names[time]:
                if paw.name is key:
                    glob_pos[key].append(paw.global_pos)

    print('glob pos:')
    print(dict(glob_pos))
    return dict(glob_pos)


def count_steps():
    steps = defaultdict(int)
    step_times = defaultdict(list)
    lift_times = defaultdict(list)
    prev_paw_names = []
    last_ind = 0
    for index, paws_on_ground in enumerate(FC.cleaned_paw_order_names):
        rec_paw_names = [paw.name for paw in paws_on_ground]
        for paw_name in rec_paw_names:
            if paw_name not in prev_paw_names:  # paw stepped down
                steps[paw_name] += 1
                step_times[paw_name].append(index)
        for prev_paw in prev_paw_names:
            if prev_paw not in rec_paw_names:  # paw lifted
                lift_times[prev_paw].append(index)
        prev_paw_names = rec_paw_names
        last_ind = index

    for last_paw in prev_paw_names:
        lift_times[last_paw].append(last_ind)

    print('\nsteps:')
    print(dict(steps))
    print('step times:')
    print(dict(step_times))
    print('lift times:')
    print(dict(lift_times))
    return dict(steps), dict(step_times), dict(lift_times)


def step_contact_time():
    step_l = defaultdict(list)
    for key, times_list in FC.step_start_times.items():
        for ind in range(len(times_list)):
            try:
                dur = FC.step_lift_times[key][ind] - FC.step_start_times[key][ind]
                step_l[key].append(dur)
            except ValueError:
                warnings.warn('no. of steps and lifts prob. not same')

    step_durations = {}
    for paw in step_l:
        step_durations[paw] = [abs_contacts / FREQ for abs_contacts in
                               step_l[paw]]  # div. by FREQ since 200 matrices equal 1s

    # print('step contacts:')
    # print(dict(step_l))
    print('step durations: (in s)')
    print(step_durations)
    return dict(step_l), step_durations


def peak_pressure():
    ctr = 0
    pressures = defaultdict(list)
    peak_times = defaultdict(list)
    peak_glob_pos = defaultdict(list)
    maximums = {'fl': -1, 'fr': -1, 'bl': -1, 'br': -1}
    tmp_times = {'fl': -1, 'fr': -1, 'bl': -1, 'br': -1}
    tmp_glob_pos = {'fl': (-1, -1), 'fr': (-1, -1), 'bl': (-1, -1), 'br': (-1, -1)}
    for paws_on_ground in FC.cleaned_paw_order_names:
        ctr += 1
        paw_names = [paw.name for paw in paws_on_ground]
        for key in maximums.keys():
            if key not in paw_names and maximums[key] != -1:  # reset max of non-touch. paws to def. after saving
                pressures[key].append(maximums[key])
                maximums[key] = -1
                peak_times[key].append(tmp_times[key])
                tmp_times[key] = -1
                peak_glob_pos[key].append(tmp_glob_pos[key])
                tmp_glob_pos[key] = (-1, -1)
        for paw in paws_on_ground:
            tmp_max = np.amax(paw.area)  # TODO: maybe change to mean/median of upper 5/10%
            if tmp_max > maximums[paw.name]:
                maximums[paw.name] = tmp_max
                tmp_times[paw.name] = ctr
                tmp_glob_pos[paw.name] = paw.global_pos

    for key in maximums.keys():
        if maximums[key] != -1:  # save not yet lifted paws at last time step since data ends before lifting
            pressures[key].append(maximums[key])
            maximums[key] = -1

    print('peak pressures:')
    print(dict(pressures))
    print('peak times:')
    print(dict(peak_times))
    print('peak glob pos:')
    print(dict(peak_glob_pos))
    return dict(pressures), dict(peak_times), dict(peak_glob_pos)


def step_length():
    step_l = {}
    for paw_name, glob_positions in FC.peak_glob_positions.items():
        step_dist = []
        for pos_ind in range(len(glob_positions) - 1):
            x1, y1 = FC.peak_glob_positions[paw_name][pos_ind]
            x2, y2 = FC.peak_glob_positions[paw_name][pos_ind + 1]
            x_square = abs(x2 - x1) * CELL_SIZE_MM / 1000
            y_square = abs((y2 - y1)) * CELL_SIZE_MM / 1000
            dist = round(math.sqrt(x_square ** 2 + y_square ** 2), 3)
            step_dist.append(dist)

        step_l[paw_name] = step_dist

    print('step length:')
    print(step_l)
    return step_l
