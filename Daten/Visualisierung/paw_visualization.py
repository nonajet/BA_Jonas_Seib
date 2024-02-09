import sys
import time
import traceback
import warnings

import numpy as np
from matplotlib import pyplot as plt

from Daten.Visualisierung import paw_detection, feature_creation
from Daten.Visualisierung.extract_xml import extract_matrices, create_global_mx
from paw_detection import paw_recognition, TheDog


def visualize(filepath, _id, mx_start=0, visuals=False, total_view=False, mx_skip=1, vis_from=0):
    t1 = time.time()
    feature_cont = feature_creation.FeatureContainer()
    np.set_printoptions(threshold=sys.maxsize)
    matrix, offset = extract_matrices(filepath, _id)

    fig, ax = plt.subplots(1, 3)  # (ax_local, ax_global, ax_total)
    plt.ion()
    measure_name = filepath.split('\\')[-1].split()[0] + ' - ' + _id
    plt.title(measure_name)

    ax_local = ax[0]
    ax_local.set_title('local')
    ax_global = ax[1]
    ax_global.set_title('global')
    ax_total = ax[2]
    total_mx = np.zeros((481, 64))
    plt.axis('off')

    fig_paws, axes_paws = plt.subplots(2, 2)
    if visuals:
        plt.figure(fig_paws)
    else:
        plt.close(fig)
        plt.close(fig_paws)

    threshold = 2
    # default filter for "noise" (mx < th)
    matrix = [np.array(mx) for mx in matrix]
    matrix = [np.where(mx_np < threshold, 0, mx_np) for mx_np in matrix]
    while not matrix[0].any():  # empty data on start&end allowed since measurement starts/is finished there
        del matrix[0]  # empty data in between means dog levitates or left mat (= unreliable data)
        del offset[0]
    while not matrix[-1].any():
        del matrix[-1]
        del offset[-1]

    mx_ctr = mx_start
    for mx in matrix[mx_ctr:]:
        mx_np = mx
        if mx_np.any():
            global_mx = create_global_mx(mx_np, offset[mx_ctr])
            if not paw_detection.valid_data(global_mx):  # TODO: check if dog walks across whole mat
                raise UserWarning('paws too close to edge')
            total_mx += global_mx
            paws = paw_recognition(mx_np, offset[mx_ctr], global_mx, mx_ctr)
            feature_cont.save_paws(paws)

            if visuals and mx_ctr % mx_skip == 0 and mx_ctr >= vis_from:
                vis_paws(fig_paws, axes_paws)

                # local
                ax_local.imshow(mx_np)
                ax_local.set_axis_off()
                # ax_local.imshow(np.flipud(np.fliplr(mx_np)))  # rot. 180° to fit glob. view direction

                # global
                ax_global.imshow(global_mx)
                ax_global.set_axis_off()

                if total_view:
                    # total (drains performance heavily)
                    ax_total.imshow(total_mx)
                    ax_total.set_title('total')

        else:
            if mx_ctr >= 0.50 * len(matrix):  # cut measurement at empty mx if dog has crossed majority of the mat
                print('cut at empty matrix, still using data')
                return feature_cont
            else:
                raise UserWarning('matrix empty at {}/{}'.format(mx_ctr, len(matrix)))
        mx_ctr += 1

    t2 = time.time()
    return feature_cont
    # print('\n\nduration:', t2 - t1)


def vis_paws(figure, axes):
    plt.ion()
    plt.figure(figure)

    for paw_obj in TheDog.paws:
        try:
            if paw_obj.ground:
                axes[paw_obj.ax_ind].matshow(paw_obj.area)
            else:
                axes[paw_obj.ax_ind].cla()
            axes[paw_obj.ax_ind].set_title(paw_obj.name)
            axes[paw_obj.ax_ind].set_axis_off()
        except TypeError:
            print(traceback.format_exc())
            print('err in:', paw_obj.area)

    plt.pause(0.000001)
