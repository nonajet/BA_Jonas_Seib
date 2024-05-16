import copy
import math
import sys
import traceback
import numpy as np

from collections import defaultdict
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
        self.pos = (-1, -1)

        self.peak_t = -1
        self.peak_val = -1

        self.paws_after = []
        self.paws_before = []
        self.parallel = []
        self.was_front_of = []
        self.was_back_of = []


def create_seq(steps_nAssigned):
    seqs = []
    for step in steps_nAssigned:
        seq = StepSeq(step)
        seq.start = step[0].time
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
        self.peak_pressure = {}  # todo: include and rerun
        self.step_lengths = {}  # in m
        self.step_length_no_of_matrices = {}  # in no. of matrices
        self.avg_pres = {}
        self.vert_forces = {}
        self.ratio_pres2area = {}
        self.asc_peak_slopes = {}
        self.des_peak_slopes = {}
        self.impulses = {}
        # positions
        self.step_start_glob_pos = {}
        self.peak_glob_positions = {}
        self.step_lift_glob_pos = {}
        self.paw_area = {}

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
        # cleanup time order (delete noise)
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
        self.set_peak_data()
        self.paw_order_by_dist = sorted(step_seqs, key=lambda x: x.pos[0], reverse=self.direction)
        self.sorted_peak_times = sorted(step_seqs, key=lambda x: x.peak_t)  # sort all steps by their peak_t time value
        self.calc_parallel_paws()
        self.count_front_occ()

        self.rename_seq()
        self.paw_order_by_steps = {}  # unusable after renaming
        self.group_by_paws()  # usable again

        # feature calculation
        err = self.count_steps_glob_pos()
        if not err:
            return err
        self.step_contact_time()
        self.calc_peak_times()
        # self.air_time()
        self.step_length()
        self.calc_pace()
        self.region_properties()
        self.peak_slopes()

        visualize_data(self.paws_time_order, visuals=False, total_view=True, mx_skip=2, mx_start=0, vis_from=0,
                       paws=True)
        return True

    def set_peak_data(self):
        for seq in self.paw_order_by_time:
            step_max = -1.0
            peak_frame = None
            for tframe in seq.paw_seq:
                tmp_max = np.amax(tframe.area)
                if tmp_max > step_max:
                    step_max = tmp_max
                    peak_frame = tframe

            seq.peak_t = peak_frame.time
            seq.peak_val = np.amax(peak_frame.area)
            seq.pos = peak_frame.global_pos

    def calc_peak_times(self):
        peak_times = defaultdict(list)
        peak_glob_pos = defaultdict(list)
        for seq in self.paw_order_by_time:
            key = seq.name
            peak_times[key].append(seq.peak_t)
            peak_glob_pos[key].append(seq.pos)

        self.peak_times = dict(peak_times)
        self.peak_glob_positions = dict(peak_glob_pos)

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
        for seq in self.paw_order_by_time:
            paw_key = seq.name
            steps[paw_key] += 1
            step_times[paw_key].append(seq.start)
            start_glob_pos[paw_key].append(seq.paw_seq[0].global_pos)

            lift_times[paw_key].append(seq.end)
            lift_glob_pos[paw_key].append(seq.paw_seq[-1].global_pos)

        step_ctr = []
        for val in steps:  # at least 2 steps per paw must be detected
            step_ctr.append(steps[val])
            if steps[val] < 2:
                return -1

        self.no_of_steps = dict(steps)
        self.step_start_times = dict(step_times)
        self.step_lift_times = dict(lift_times)

        self.step_start_glob_pos = dict(start_glob_pos)
        self.step_lift_glob_pos = dict(lift_glob_pos)

        if abs(max(step_ctr) - min(step_ctr)) >= 2:
            print('difference in step count too high -- {} to {}'.format(max(step_ctr), min(step_ctr)))
            return -2

        return True

    def region_properties(self):
        CONV_FAC = 8.5 * 8.5 / 100  # convert from cell area to cm²

        step_area = defaultdict(list)
        avg_pres = defaultdict(list)
        ratio_pres_area = defaultdict(list)
        vert_forces = defaultdict(list)
        impulses = defaultdict(list)

        for seq in self.paw_order_by_time:
            key = seq.name
            areas_in_step = []
            avg_pres_in_step = []
            forces_in_step = []

            for step in seq.paw_seq:
                area = step.props[0].area  # given in number of cells
                areas_in_step.append(area)
                avg_pres_in_step.append(np.round(np.mean(step.area[step.area != 0]), 3))  # ignore 0 val for avg

                force = np.sum(step.area) * CONV_FAC  # in N; [pressure * area]
                forces_in_step.append(np.round(force, 3))

            m_force = np.mean(forces_in_step)
            paw_impulse = m_force * (len(seq.paw_seq) / FREQ)  # convert to N*s
            impulses[key].append(np.round(paw_impulse, 3))

            avg_area_in_cm2 = np.mean(areas_in_step) * CONV_FAC
            step_area[key].append(round(avg_area_in_cm2, 3))  # mean area for single step sequence
            avg_pres[key].append(np.round(np.mean(avg_pres_in_step), 3))

            vert_forces[key].append(np.round(m_force, 3))

            # append value of newest step sequence of that paw
            ratio_pres_area[key].append(np.round(avg_pres[key][-1] / step_area[key][-1], 3))

        self.paw_area = dict(step_area)
        self.avg_pres = dict(avg_pres)
        self.ratio_pres2area = dict(ratio_pres_area)
        self.vert_forces = dict(vert_forces)
        self.impulses = dict(impulses)

    def calc_pace(self):  # in km/h
        time = len(self.paws_time_order) / FREQ
        start_pos = self.paw_order_by_time[0].pos
        end_pos = self.paw_order_by_time[-1].pos

        x_square = abs(end_pos[0] - start_pos[0]) * CELL_SIZE_MM / 1000
        y_square = abs((end_pos[1] - start_pos[1])) * CELL_SIZE_MM / 1000
        dist = round(math.sqrt(x_square ** 2 + y_square ** 2), 3)
        avg_pace = round((dist / time) * 3.6, 2)

        self.pace = avg_pace

    def air_time(self):  # not usable due to mixed up paw pairs
        air_t_matrices = defaultdict(list)  # not used as output feature
        air_t_s = defaultdict(list)
        for seq in self.paw_order_by_time:
            key = seq.name
            dur = seq.end - seq.start  # TODO: subtract start of x+1 from end of x
            air_t_matrices[key].append(dur)
            air_t_s[key].append(dur / FREQ)

        # self.air_durations = dict(air_t_s)

    def step_contact_time(self):  # time from setting to lifting a paw
        step_l = defaultdict(list)
        for seq in self.paw_order_by_time:
            dur = seq.paw_seq[-1].time - seq.paw_seq[0].time
            step_l[seq.paw_seq[0].name].append(dur)

        step_durations = {}
        for paw in step_l:
            step_durations[paw] = [abs_contacts / FREQ for abs_contacts in
                                   step_l[paw]]  # div. by FREQ since 200 matrices equal 1s

        self.step_length_no_of_matrices, self.contact_durations = dict(step_l), step_durations

    def peak_slopes(self):
        asc = defaultdict(list)
        des = defaultdict(list)
        for seq in self.paw_order_by_time:
            paw_asc = []
            paw_des = []
            key = seq.name

            start_t = seq.start
            peak_t = seq.peak_t
            peak_val = seq.peak_val
            end_t = seq.end

            # ascend
            asc_t = abs(peak_t - start_t)
            if not asc_t: asc_t = 1
            asc_slope = peak_val / asc_t
            paw_asc.append(asc_slope)

            # descend
            des_t = abs(peak_t - end_t)
            if not des_t: des_t = 1
            des_slope = peak_val / des_t
            paw_des.append(des_slope)

            asc[key].append(np.round(np.mean(paw_asc), 3))
            des[key].append(np.round(np.mean(paw_des), 3))

        self.asc_peak_slopes, self.des_peak_slopes = dict(asc), dict(des)

    def step_length(self):  # spatial distance/length between both paws alternating of one pair
        step_l = {}
        for paw_key, seqs in self.paw_order_by_steps.items():
            step_dist = []
            for ind in range(len(seqs) - 1):
                x1, y1 = seqs[ind].pos
                x2, y2 = seqs[ind + 1].pos
                x_square = abs(x2 - x1) * CELL_SIZE_MM / 1000
                y_square = abs((y2 - y1)) * CELL_SIZE_MM / 1000
                dist = round(math.sqrt(x_square ** 2 + y_square ** 2), 3)
                step_dist.append(dist)

            step_l[paw_key] = step_dist

        self.step_lengths = step_l

    def calc_parallel_paws(self):  # calc all paws which touch ground at same time
        for seq in self.paw_order_by_time:
            t_start = seq.start
            t_end = seq.end
            for other in self.paw_order_by_time:
                other_s = other.start
                other_e = other.end
                if seq is not other:
                    if other.end < t_start or t_end < other.start:  # not simultan. on ground
                        pass
                    elif t_start < other.start:  # <= t_end:
                        seq.paws_after.append(other)
                        seq.parallel.append(other)
                    elif other.start < t_start:  # t_start <= other.end <= t_end:
                        seq.paws_before.append(other)
                        seq.parallel.append(other)
                    elif other.start == t_start:  # same time of contact --> count as before & after simultan.
                        seq.parallel.append(other)
                        seq.paws_after.append(other)
                        seq.paws_before.append(other)

    def count_front_occ(self):
        for seq in self.paw_order_by_time:
            t_start = seq.start
            t_end = seq.end
            my_pos = seq.pos
            for other in seq.parallel:
                # if mylib.DIRECTION == 0:  # independent of direction?
                if my_pos[0] < other.pos[0]:  # 0 == backwards
                    seq.was_front_of.append(other)
                else:
                    seq.was_back_of.append(other)
                # else:
                #     if my_pos[0] > other.pos[0]:  # 1 == forward
                #         seq.was_front_of.append(other)
                #     else:
                #         seq.was_back_of.append(other)

    def rename_seq(self):
        # front = 0
        # back = 0
        most_upfront = self.paw_order_by_time[0].pos[0]  # order/start goes from 480 to 0 on matrix/mat
        for seq in self.paw_order_by_time:
            if seq is self.paw_order_by_time[0] and all(seq.peak_t < par.start for par in seq.parallel):  # first paw
                seq.name = 'A'
                paw_name_key = 'A'
            elif seq is self.paw_order_by_time[-1]:  # last paw special case as this must be back paw
                seq.name = 'C'
                paw_name_key = 'C'
            else:
                if len(seq.was_front_of) > len(seq.was_back_of):  # front
                    seq.name = 'A'
                    paw_name_key = 'A'
                    # if front % 2:
                    #     paw_name_key = 'A'
                    # else:
                    #     paw_name_key = 'B'
                    # front += 1
                elif len(seq.was_front_of) < len(seq.was_back_of):  # back
                    seq.name = 'C'
                    paw_name_key = 'C'
                    # if back % 2:
                    #     paw_name_key = 'C'
                    # else:
                    #     paw_name_key = 'D'
                    # back += 1
                else:  # same length, e.g. same amount 'was overtaken' vs. 'overtook'
                    if seq.pos[0] >= most_upfront:  # lower x coordinate means closer to end of mat--> further up front
                        seq.name = 'C'
                        paw_name_key = 'C'
                    else:
                        seq.name = 'A'
                        paw_name_key = 'A'

            if most_upfront > seq.pos[0]:
                most_upfront = seq.pos[0]  # new frontmost paw position

            for tframe in seq.paw_seq:  # rename whole step sequence according to new name
                tframe.name = paw_name_key

    def group_by_paws(self):
        dic = defaultdict(list)
        for seq in self.paw_order_by_time:
            key = seq.name
            dic[key].append(seq)

        self.paw_order_by_steps = dict(dic)


def visualize_data(data, visuals=False, total_view=False, mx_start=0, mx_skip=1, vis_from=0, paws=True):
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
    ax_total.set_title('total')
    total_mx = np.zeros((481, 64))
    # plt.axis('off')

    plt.figure(fig)
    if paws:
        fig_paws, axes_paws = plt.subplots(2, 2)
        plt.figure(fig_paws)
    else:
        fig_paws, axes_paws = None, None

    mx_ctr = mx_start
    for ground_paws in data[mx_ctr:]:
        global_mx = np.zeros((481, 64))
        for paw in ground_paws:
            rows, cols = paw.area.shape
            row_start, col_start = paw.global_pos
            global_mx[row_start:row_start + rows, col_start:col_start + cols] += paw.area

        total_mx += global_mx

        if visuals and mx_ctr % mx_skip == 0 and mx_ctr >= vis_from:
            if paws:
                vis_paws(ground_paws, fig_paws, axes_paws, mx_ctr)

            # global
            ax_global.imshow(global_mx)
            ax_global.set_axis_off()

            # total (drains performance heavily)
            if total_view and mx_ctr % 4 == 0:
                ax_total.imshow(total_mx)
                # ax_total.set_title('total')

        mx_ctr += 1


def vis_paws(data_obj, figure, axes, mx_ctr):
    # plt.ion()
    plt.figure(figure)
    mod_ctr = 0
    no_2_name = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    ax_2_name = {'A': (0, 0), 'B': (0, 1), 'C': (1, 0), 'D': (1, 1)}
    axes_used = []

    for paw in data_obj:
        ax = ax_2_name[no_2_name[mod_ctr % 4]]
        mod_ctr += 1
        name = paw.name
        axes_used.append(ax)
        try:
            if paw.valid:
                axes[ax].matshow(paw.area)
                axes[ax].set_title(name)
                axes[ax].set_axis_off()
                figure.suptitle(mx_ctr)
        except TypeError:
            print(traceback.format_exc())
            print('err in:', paw.area)

    for axis in ax_2_name.values():
        if axis not in axes_used:
            axes[axis].cla()

    plt.pause(0.000001)
