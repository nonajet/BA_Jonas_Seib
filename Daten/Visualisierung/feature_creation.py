import copy
import math
import warnings
from collections import defaultdict

import numpy as np

CELL_SIZE_MM = 8.4688  # in mm
UNINIT_VAL = ''  # default value for not yet initialised paws (order)
FREQ = 200


class FeatureContainer(object):

    def __init__(self, dog_id=None):
        self.dog_id = dog_id

        self.raw_paw_order = []
        self.cleaned_paw_order_names = []
        self.paw_order_by_steps = {}  # each step per paw is separately saved

        # misc
        self.no_of_steps = {}
        self.peak_pressure = {}
        self.step_lengths = {}  # in m
        self.step_length_no_of_matrices = {}  # in no. of matrices
        self.avg_pres = {}
        self.ratio_pres2area = {}
        self.asc_peak_slopes = {}
        self.des_peak_slopes = {}

        # positions
        self.step_start_glob_pos = {}
        self.peak_glob_positions = {}
        self.step_lift_glob_pos = {}
        self.paw_area_cm2 = {}

        # times
        self.step_start_times = {}  # 'times' refers to index of the event (e.g. start of step)
        self.peak_times = {}
        self.step_lift_times = {}
        self.contact_durations = {}
        self.air_durations = {}
        self.pace = -1

    def calc_features(self):
        self.cleaned_paw_order_names = self.raw_paw_order
        self.get_ordered_paw_data()

        if not self.count_steps():
            return False
        self.calc_glob_pos_of_steps()
        if not self.step_contact_time():
            return False
        self.air_time()
        self.calc_peak_pressure()
        self.step_length()
        self.calc_pace()
        self.region_properties()
        self.peak_slopes()
        return True

    def save_paws(self, dog_paws):
        active_paws = [copy.copy(paw) for paw in dog_paws if paw.ground]
        self.raw_paw_order.append(active_paws)

    def get_ordered_paw_data(self):
        """
        orders each step of one paw as a list of adjacent paw-ground contacts.
        As soon as paw is lifted (=step process is finished) that step for that paw is added to
        the ordered paw data (=return value).
        Is foundation for easier propregions calculation later on.
        """
        paw_data_per_paw_step = defaultdict(list)
        rec_paws = []
        step = defaultdict(list)
        for paws_on_ground in self.cleaned_paw_order_names:
            for paw in paws_on_ground:  # each paw that sets down is added to step of their key
                step[paw.name].append(paw)
            paws_on_ground = dict({(paw_obj.name, paw_obj) for paw_obj in paws_on_ground})  # convert from list to dict
            prev_paws = rec_paws
            rec_paws = [keys for keys in paws_on_ground.keys()]
            paws_lifted = list(set(prev_paws) - set(rec_paws))
            if paws_lifted:
                for paw_key in paws_lifted:  # asa paw lifts the whole step (as a list) is added to data_per_step
                    paw_data_per_paw_step[paw_key].append(step[paw_key])
                    step[paw_key] = []

        for key in step:  # don't forget last paw(s) that are on last time step/matrix
            if step[key]:
                paw_data_per_paw_step[key].append(step[key])
                step[key] = []

        self.paw_order_by_steps = dict(paw_data_per_paw_step)

    def calc_glob_pos_of_steps(self):
        glob_pos = defaultdict(list)
        for key, val_list in self.step_start_times.items():
            for time in val_list:
                for paw in self.cleaned_paw_order_names[time]:
                    if paw.name is key:
                        glob_pos[key].append(paw.global_pos)

        self.step_start_glob_pos = dict(glob_pos)

    def count_steps(self):
        steps = defaultdict(int)
        step_times = defaultdict(list)
        lift_times = defaultdict(list)
        prev_paw_names = []
        last_ind = 0
        for index, paws_on_ground in enumerate(self.cleaned_paw_order_names):
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

        self.no_of_steps, self.step_start_times, self.step_lift_times = dict(steps), dict(step_times), dict(lift_times)

        for val in steps:  # at least 2 steps per paw must be detected
            if steps[val] < 2:
                return False
        return True

    def region_properties(self):
        CONV_FAC = 8.5 * 8.5 / 100  # convert from cell area to cm²
        step_area_cm2 = defaultdict(list)
        avg_pres = defaultdict(list)
        ratio_pres_area = defaultdict(list)
        for key, steps in self.paw_order_by_steps.items():
            areas_in_step = []
            avg_pres_in_step = []
            for step in steps:
                for paw_data in step:
                    areas_in_step.append(paw_data.props[0].area)
                    avg_pres_in_step.append(np.mean(paw_data.area[paw_data.area != 0]))  # ignore 0 values in mx for avg

                avg_area_in_cm2 = np.mean(areas_in_step) * CONV_FAC
                step_area_cm2[key].append(round(avg_area_in_cm2, 3))  # mean area for single step
                areas_in_step = []

                avg_pres[key].append(np.round(np.mean(avg_pres_in_step), 3))
                avg_pres_in_step = []

            ratio_pres_area[key].append(np.round(avg_pres[key][-1] / step_area_cm2[key][-1], 3))
        self.paw_area_cm2, self.avg_pres, self.ratio_pres2area = dict(step_area_cm2), dict(avg_pres), dict(
            ratio_pres_area)

    def calc_pace(self):
        time = len(self.cleaned_paw_order_names) / FREQ
        start_pos = self.cleaned_paw_order_names[0][0].global_pos
        end_pos = self.cleaned_paw_order_names[-1][0].global_pos

        x_square = abs(end_pos[0] - start_pos[0]) * CELL_SIZE_MM / 1000
        y_square = abs((end_pos[1] - start_pos[1])) * CELL_SIZE_MM / 1000
        dist = round(math.sqrt(x_square ** 2 + y_square ** 2), 3)
        avg_pace = round((dist / time) * 3.6, 2)

        self.pace = avg_pace

    def air_time(self):  # expects step and lift time lists to be same length
        air_t_matrices = defaultdict(list)
        air_t_s = defaultdict(list)
        for key, lift_times in self.step_lift_times.items():
            for ind in range(len(lift_times) - 1):
                dur = self.step_start_times[key][ind + 1] - self.step_lift_times[key][ind]
                air_t_matrices[key].append(dur)
                air_t_s[key].append(dur / FREQ)

        self.air_durations = dict(air_t_s)

    def step_contact_time(self):  # time from setting to lifting a paw
        step_l = defaultdict(list)
        for key, times_list in self.step_start_times.items():
            for ind in range(len(times_list)):
                try:
                    dur = self.step_lift_times[key][ind] - self.step_start_times[key][ind]
                    step_l[key].append(dur)
                except ValueError:
                    warnings.warn('no. of steps and lifts prob. not same')

        step_durations = {}
        for paw in step_l:
            step_durations[paw] = [abs_contacts / FREQ for abs_contacts in
                                   step_l[paw]]  # div. by FREQ since 200 matrices equal 1s

        for val in step_l:
            if any(dur <= 5 for dur in step_l[val]):  # discard measures for too short contact durations
                return False

        self.step_length_no_of_matrices, self.contact_durations = dict(step_l), step_durations
        return True

    def peak_slopes(self):
        asc = defaultdict(list)
        des = defaultdict(list)
        starts = self.step_start_times
        ends = self.step_lift_times
        for key, peaks in self.peak_times.items():
            paw_asc = []
            paw_des = []
            for ind, peak_t in enumerate(peaks):
                peak_val = self.peak_pressure[key][ind]

                asc_t = abs(peak_t - starts[key][ind])
                if asc_t == 0:
                    asc_t = 1  # avoid div by 0
                asc_slope = peak_val / asc_t
                paw_asc.append(asc_slope)

                des_t = abs(peak_t - ends[key][ind])
                if des_t == 0:
                    des_t = 1
                des_slope = peak_val / des_t
                paw_des.append(des_slope)

            asc[key].append(np.round(np.mean(paw_asc), 3))
            des[key].append(np.round(np.mean(paw_des), 3))

        self.asc_peak_slopes, self.des_peak_slopes = dict(asc), dict(des)

    def calc_peak_pressure(self):
        ctr = 0
        pressures = defaultdict(list)
        peak_times = defaultdict(list)
        peak_glob_pos = defaultdict(list)
        maximums = {'A': -1, 'B': -1, 'C': -1, 'D': -1}
        tmp_times = {'A': -1, 'B': -1, 'C': -1, 'D': -1}
        tmp_glob_pos = {'A': (-1, -1), 'B': (-1, -1), 'C': (-1, -1), 'D': (-1, -1)}
        for paws_on_ground in self.cleaned_paw_order_names:
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

        self.peak_pressure, self.peak_times, self.peak_glob_positions = dict(pressures), dict(peak_times), dict(
            peak_glob_pos)

    def step_length(self):  # spatial distance/length
        step_l = {}
        for paw_name, glob_positions in self.peak_glob_positions.items():
            step_dist = []
            for pos_ind in range(len(glob_positions) - 1):
                x1, y1 = self.peak_glob_positions[paw_name][pos_ind]
                x2, y2 = self.peak_glob_positions[paw_name][pos_ind + 1]
                x_square = abs(x2 - x1) * CELL_SIZE_MM / 1000
                y_square = abs((y2 - y1)) * CELL_SIZE_MM / 1000
                dist = round(math.sqrt(x_square ** 2 + y_square ** 2), 3)
                step_dist.append(dist)

            step_l[paw_name] = step_dist

        self.step_lengths = step_l
