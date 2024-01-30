import copy
import math
import warnings
from collections import defaultdict

import numpy as np


class FeatureContainer(object):

    def __init__(self):
        self.cleaned_paw_order_names = []
        self.paw_props = []

        self.no_of_steps = {}
        self.peak_pressure = {}
        self.step_lengths = {}  # in m
        self.step_length_no_of_matrices = {}  # in no. of matrices

        self.step_start_glob_paw_positions = {}
        self.peak_glob_positions = {}
        self.step_lift_glob_paw_positions = {}

        self.step_start_times = {}
        self.peak_times = {}
        self.step_lift_times = {}
        self.step_contact_durations = {}
        self.pace = -1


UNINIT_VAL = ''  # default value for not yet initialised paws (order)
CELL_SIZE_MM = 8.4688  # in mm
FREQ = 200
FC = FeatureContainer()
raw_paw_order = []
raw_props = []


def calc_features():
    FC.cleaned_paw_order_names = raw_paw_order

    FC.no_of_steps, FC.step_start_times, FC.step_lift_times = count_steps()
    FC.step_start_glob_paw_positions = calc_glob_pos_of_steps()
    FC.step_length_no_of_matrices, FC.step_contact_durations = step_contact_time()
    FC.peak_pressure, FC.peak_times, FC.peak_glob_positions = peak_pressure()
    FC.step_lengths_time = step_length()
    FC.pace = pace()

    # logger = logging.getLogger(get_dog_log())
    # logger.info(peak_pressure())


def save_paws(dog_paws):
    assert dog_paws
    active_paws = [copy.copy(paw) for paw in dog_paws if paw.ground]
    raw_paw_order.append(active_paws)


def calc_glob_pos_of_steps():
    glob_pos = defaultdict(list)
    for key, val_list in FC.step_start_times.items():
        for time in val_list:
            for paw in FC.cleaned_paw_order_names[time]:
                if paw.name is key:
                    glob_pos[key].append(paw.global_pos)

    print('step glob pos:')
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


def region_properties():
    pass


def pace():
    time = len(FC.cleaned_paw_order_names) / FREQ
    start_pos = FC.cleaned_paw_order_names[0][0].global_pos
    end_pos = FC.cleaned_paw_order_names[-1][0].global_pos

    x_square = abs(end_pos[0] - start_pos[0]) * CELL_SIZE_MM / 1000
    y_square = abs((end_pos[1] - start_pos[1])) * CELL_SIZE_MM / 1000
    dist = round(math.sqrt(x_square ** 2 + y_square ** 2), 3)
    avg_pace = round((dist / time) * 3.6, 2)
    print('pace: (in km/h)')
    print(avg_pace)
    # print('pace: (in m/s)')
    # print(round(avg_pace / 3.6, 2))
    return avg_pace


def air_time():
    air_t = defaultdict(list)
    for key, times_list in FC.step_lift_times.items():
        for ind in range(len(times_list)):
            dur = FC.step_lift_times


def step_contact_time():  # time from setting to lifting a paw
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

    print('step contacts: (in #matrices)')
    print(dict(step_l))
    print('step durations: (in s)')
    print(step_durations)
    return dict(step_l), step_durations


def peak_pressure():
    ctr = 0
    pressures = defaultdict(list)
    peak_times = defaultdict(list)
    peak_glob_pos = defaultdict(list)
    maximums = {'A': -1, 'B': -1, 'C': -1, 'D': -1}
    tmp_times = {'A': -1, 'B': -1, 'C': -1, 'D': -1}
    tmp_glob_pos = {'A': (-1, -1), 'B': (-1, -1), 'C': (-1, -1), 'D': (-1, -1)}
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
            # upper_prc = np.percentile(paw.area, 97)
            # tmp_max = np.median(paw.area[paw.area > upper_prc])
            if tmp_max > maximums[paw.name]:
                maximums[paw.name] = tmp_max
                tmp_times[paw.name] = ctr
                tmp_glob_pos[paw.name] = paw.global_pos

    # save not yet lifted paws at last time step since data ends with lifting
    for key in maximums.keys():
        if maximums[key] != -1:
            pressures[key].append(maximums[key])
            maximums[key] = -1
    for key in tmp_times.keys():
        if tmp_times[key] != -1:
            peak_times[key].append(tmp_times[key])
            tmp_times[key] = -1
    for key in tmp_glob_pos.keys():
        if tmp_glob_pos[key] != (-1, -1):
            peak_glob_pos[key].append(tmp_glob_pos[key])
            tmp_glob_pos[key] = (-1, -1)

    print('peak pressures:')
    print(dict(pressures))
    print('peak times:')
    print(dict(peak_times))
    print('peak glob pos:')
    print(dict(peak_glob_pos))
    return dict(pressures), dict(peak_times), dict(peak_glob_pos)


def step_length():  # spatial distance/length
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

    print('step length: (in m)')
    print(step_l)
    return step_l
