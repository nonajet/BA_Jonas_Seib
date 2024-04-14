import copy
import math
import sys
import traceback
import warnings
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

from Daten.Visualisierung import mylib

CELL_SIZE_MM = 8.4688  # in mm
UNINIT_VAL = ''  # default value for not yet initialised paws (order)
FREQ = 200
letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}


class StepSeq(object):
    def __init__(self, paw_seq):
        self.paw_seq = paw_seq
        self.start = -1
        self.end = -1
        self.peak = -1
        self.pos = (-1, -1)
        self.was_overtkn = []
        self.overtook = []
        self.parallel = []
        self.front = []
        self.back = []


def create_seq(steps_nAssigned):
    seqs = []
    for step in steps_nAssigned:
        seq = StepSeq(step)
        # seq.paw_seq = step
        seq.start = step[0].time
        seq.pos = step[0].global_pos
        seq.end = step[-1].time
        seqs.append(seq)
    return seqs


class FeatureContainer(object):

    def __init__(self, dog_id=None):
        self.dog_id = dog_id
        self.direction = mylib.DIRECTION  # 0 == backward; 1 == forward

        self.raw_paw_order = []
        self.paws_time_order = []
        self.paw_order_by_steps = {}  # each step per paw is separately saved
        self.paw_name_per_step = []  # names of all paws which touch the ground at each time slot
        self.paw_order_by_time = []
        self.paw_order_by_dist = []

        # misc
        self.no_of_steps = {}
        # self.peak_pressure = {}
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
        self.sorted_peak_times = []

    def calc_features(self):
        self.paws_time_order = self.raw_paw_order
        # cleanup time order (delete noise, TODO delete paw names)
        self.paw_order_by_steps = self.get_ordered_paw_data_by_name()  # used to group adj. paw steps
        steps_nAssigned = []  # reassign paw names while filtering noisy data/steps
        for paw_key in self.paw_order_by_steps:
            for step in self.paw_order_by_steps[paw_key]:
                if len(step) > 5:
                    steps_nAssigned.append(step)
                else:
                    for frame in step:  # remove recorded steps which are too short
                        frame.valid = False
                        self.paws_time_order[frame.time].remove(frame)

        step_seqs = create_seq(steps_nAssigned)
        self.paw_order_by_time = sorted(step_seqs, key=lambda x: x.start)
        self.paw_order_by_dist = sorted(step_seqs, key=lambda x: x.pos[0], reverse=self.direction)
        self.calc_peaks()
        self.sorted_peak_times = sorted(step_seqs, key=lambda x: x.peak)  # sort all steps by their peak time value
        self.calc_parallel_paws()
        self.count_front_occ()

        # RENAMING
        # for i, step in enumerate(self.sorted_peak_times):
        #     paw_name_key = letter[i % 4]
        #     for tframe in steps_nAssigned[step[0]]:
        #         tframe.name = paw_name_key  # rewrite each paws internal name according to new order
        #     self.paw_order_by_steps[paw_name_key].append(steps_nAssigned[step[0]])
        # fill w new assigned/named pawsteps

        # feature calculation
        if not self.count_steps_glob_pos():
            return False
        # self.calc_glob_pos_of_steps()
        self.step_contact_time()
        self.calc_peak_times(steps_nAssigned)
        self.air_time()
        self.step_length()
        self.calc_pace()
        self.region_properties()
        self.peak_slopes()

        visualize_data(self, visuals=True, total_view=False, mx_skip=1, mx_start=0, vis_from=0)
        return True

    def calc_peaks(self):
        for seq in self.paw_order_by_time:
            step_max = -1.0
            peak_frame = None
            for tframe in seq.paw_seq:
                tmp_max = np.amax(tframe.area)
                if tmp_max > step_max:
                    step_max = tmp_max
                    peak_frame = tframe

            seq.peak = peak_frame.time

    def calc_peak_times(self, unassigned_steps):
        pressures = defaultdict(list)
        peak_times = defaultdict(list)
        peak_glob_pos = defaultdict(list)
        for nr, step in enumerate(unassigned_steps):
            step_max = -1.0
            peak_frame = None
            for i, tframe in enumerate(step):
                tmp_max = np.amax(tframe.area)
                if tmp_max > step_max:
                    step_max = tmp_max
                    peak_frame = tframe

            key = peak_frame.name
            pressures[key].append(step_max)
            peak_times[key].append(peak_frame.time)
            peak_glob_pos[key].append(peak_frame.global_pos)

        for k in pressures.keys():
            pressures[k].sort()
            peak_times[k].sort()
            peak_glob_pos[k].sort()

        self.peak_times = dict(peak_times)
        self.peak_glob_positions = dict(peak_glob_pos)
        # self.peak_pressure = dict(pressures)

    def save_paws(self, dog_paws):
        active_paws = [copy.copy(paw) for paw in dog_paws if paw.ground]
        self.raw_paw_order.append(active_paws)

    def get_ordered_paw_data_by_name(self):
        """
        orders each step of one paw as a list of adjacent paw-ground contacts.
        As soon as paw is lifted (=step process is finished) that step for that paw is added to
        the ordered paw data (=return value).
        Is foundation for easier propregions calculation later on.
        """
        paw_data_per_paw_step = defaultdict(list)
        paw_name_per_step = []
        rec_paws = []
        step = defaultdict(list)
        for paws_on_ground in self.paws_time_order:
            for paw in paws_on_ground:  # each paw that sets down is added to step of their key
                step[paw.name].append(paw)
            paws_on_ground = dict({(paw_obj.name, paw_obj) for paw_obj in paws_on_ground})  # convert from list to dict
            prev_paws = rec_paws
            rec_paws = [keys for keys in paws_on_ground.keys()]
            paw_name_per_step.append(sorted(rec_paws))
            paws_lifted = list(set(prev_paws) - set(rec_paws))
            if paws_lifted:
                for paw_key in paws_lifted:  # asa paw lifts the whole step (as a list) is added to data_per_step
                    paw_data_per_paw_step[paw_key].append(step[paw_key])
                    step[paw_key] = []

        for key in step:  # don't forget last paw(s) that are on last time step/matrix
            if step[key]:
                paw_data_per_paw_step[key].append(step[key])
                step[key] = []

        for i in range(len(paw_name_per_step)):
            print(i, paw_name_per_step[i])
        return dict(paw_data_per_paw_step)  # , paw_name_per_step

    def calc_glob_pos_of_steps(self):
        glob_pos = defaultdict(list)
        for key, val_list in self.step_start_times.items():
            for time in val_list:
                for paw in self.paws_time_order[time]:
                    if paw.name is key:
                        glob_pos[key].append(paw.global_pos)

        self.step_start_glob_pos = dict(glob_pos)

    def count_steps_glob_pos(self):
        steps = defaultdict(int)
        step_times = defaultdict(list)
        lift_times = defaultdict(list)

        start_glob_pos = defaultdict(list)
        lift_glob_pos = defaultdict(list)
        for paw_key, strides in self.paw_order_by_steps.items():
            steps[paw_key] = len(strides)
            for step_ind in range(len(strides)):
                step_times[paw_key].append(strides[step_ind][0].time)
                start_glob_pos[paw_key].append(strides[step_ind][0].global_pos)

                lift_times[paw_key].append(strides[step_ind][-1].time)
                lift_glob_pos[paw_key].append(strides[step_ind][-1].global_pos)

        self.no_of_steps = dict(steps)
        self.step_start_times = dict(step_times)
        self.step_lift_times = dict(lift_times)

        self.step_start_glob_pos = dict(start_glob_pos)
        self.step_lift_glob_pos = dict(lift_glob_pos)

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
        time = len(self.paws_time_order) / FREQ
        start_pos = self.paws_time_order[0][0].global_pos
        end_pos = self.paws_time_order[-1][0].global_pos

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

        self.step_length_no_of_matrices, self.contact_durations = dict(step_l), step_durations

    def peak_slopes(self):
        asc = defaultdict(list)
        des = defaultdict(list)
        starts = self.step_start_times
        ends = self.step_lift_times
        for key, peaks in self.peak_times.items():
            paw_asc = []
            paw_des = []
            for ind, peak_t in enumerate(peaks):
                peak_val = -1
                for paw in self.paws_time_order[peak_t]:
                    if paw.name == key:
                        peak_val = np.amax(paw.area)

                asc_t = abs(peak_t - starts[key][ind])
                asc_slope = peak_val / asc_t
                paw_asc.append(asc_slope)

                des_t = abs(peak_t - ends[key][ind])
                des_slope = peak_val / des_t
                paw_des.append(des_slope)

            asc[key].append(np.round(np.mean(paw_asc), 3))
            des[key].append(np.round(np.mean(paw_des), 3))

        self.asc_peak_slopes, self.des_peak_slopes = dict(asc), dict(des)

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

    def calc_parallel_paws(self):
        for seq in self.paw_order_by_time:
            t_start = seq.start
            t_end = seq.end
            for other in self.paw_order_by_time:
                if seq is not other:
                    if t_start < other.start <= t_end:
                        seq.was_overtkn.append(other)
                        seq.parallel.append(other)
                    elif t_start <= other.end <= t_end:
                        seq.overtook.append(other)
                        seq.parallel.append(other)
                    # TODO both cases simultaneously?

    def count_front_occ(self):
        for seq in self.paw_order_by_time:
            t_start = seq.start
            t_end = seq.end
            my_pos = seq.pos
            for other in seq.parallel:
                if mylib.DIRECTION == 0:
                    if my_pos[0] > other.pos[0]:
                        seq.front.append(other)
                    else:
                        other.back.append(seq)
                else:
                    if my_pos[0] < other.pos[0]:
                        seq.front.append(other)
                    else:
                        other.back.append(seq)


def visualize_data(feature_container, visuals=False, total_view=False, mx_start=0, mx_skip=1, vis_from=0):
    if not visuals:
        return
    np.set_printoptions(threshold=sys.maxsize)

    fig, ax = plt.subplots(1, 3)  # (ax_local, ax_global, ax_total)
    plt.ion()

    ax_local = ax[0]
    ax_local.set_title('local')
    ax_global = ax[1]
    ax_global.set_title('global')
    ax_total = ax[2]
    total_mx = np.zeros((481, 64))
    plt.axis('off')

    fig_paws, axes_paws = plt.subplots(2, 2)
    plt.figure(fig_paws)

    data = feature_container.paws_time_order
    mx_ctr = mx_start
    for ground_paws in data[mx_ctr:]:
        global_mx = np.zeros((481, 64))
        for paw in ground_paws:
            rows, cols = paw.area.shape
            row_start, col_start = paw.global_pos
            global_mx[row_start:row_start + rows, col_start:col_start + cols] += paw.area

        total_mx += global_mx

        if visuals and mx_ctr % mx_skip == 0 and mx_ctr >= vis_from:
            vis_paws(ground_paws, fig_paws, axes_paws)

            # local
            # if mx_np.any():
            #     ax_local.imshow(mx_np)
            #     ax_local.set_axis_off()
            # ax_local.imshow(np.flipud(np.fliplr(mx_np)))  # rot. 180° to fit glob. view direction

            # global
            ax_global.imshow(global_mx)
            ax_global.set_axis_off()

            # total (drains performance heavily)
            if total_view:
                ax_total.imshow(total_mx)
                ax_total.set_title('total')

        mx_ctr += 1


def vis_paws(data_obj, figure, axes):
    plt.ion()
    plt.figure(figure)
    ax_2_name = {'A': (0, 0), 'B': (0, 1), 'C': (1, 0), 'D': (1, 1)}
    paws_shown = []

    for paw in data_obj:
        try:
            name = paw.name
            paws_shown.append(name)
            if paw.valid:
                axes[ax_2_name[name]].matshow(paw.area)
        except TypeError:
            print(traceback.format_exc())
            print('err in:', paw.area)

    for key in ax_2_name.keys():
        axes[ax_2_name[key]].set_title(key)
        axes[ax_2_name[key]].set_axis_off()
        if key not in paws_shown:
            axes[ax_2_name[key]].cla()

    plt.pause(0.000001)
