import sys
import traceback

import numpy as np
from matplotlib import pyplot as plt

from Daten.Visualisierung import paw_detection, feature_creation
from Daten.Visualisierung.extract_xml import extract_matrices, create_global_mx
from paw_detection import paw_recognition, TheDog


def visualize(filepath, _id, mx_start=0, visuals=False, total_view=False, mx_skip=1, vis_from=0):
    np.set_printoptions(threshold=sys.maxsize)
    matrix, offset = extract_matrices(filepath, _id)

    fig, ax = plt.subplots(1, 3)  # (ax_local, ax_global, ax_total)
    plt.ion()
    measure_name = filepath.split('\\')[-1].split()[0] + ' - ' + _id
    dog_id = measure_name.split()[0]
    feature_cont = feature_creation.FeatureContainer(dog_id)
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

    matrix = [np.array(mx) for mx in matrix]

    while not matrix[0].any():  # empty data on start&end offers no value since measurement starts/is finished there
        del matrix[0]  # trim zero matrices on start and end
        del offset[0]
    while not matrix[-1].any():
        del matrix[-1]
        del offset[-1]

    mx_ctr = mx_start
    for mx in matrix[mx_ctr:]:
        mx_np = mx
        if mx_np.any():
            global_mx = create_global_mx(mx_np, offset[mx_ctr])
        else:
            global_mx = np.zeros((481, 64))  # global is zeros only if local was empty
        if not paw_detection.valid_data(global_mx):
            raise UserWarning('paws too close to edge')
        total_mx += global_mx
        paws = paw_recognition(mx_np, offset[mx_ctr], mx_ctr)
        feature_cont.save_paws(paws)

        if visuals and mx_ctr % mx_skip == 0 and mx_ctr >= vis_from:
            vis_paws(fig_paws, axes_paws)

            # local
            if mx_np.any():
                ax_local.imshow(mx_np)
                ax_local.set_axis_off()
            # ax_local.imshow(np.flipud(np.fliplr(mx_np)))  # rot. 180° to fit glob. view direction

            # global
            ax_global.imshow(global_mx)
            ax_global.set_axis_off()

            # total (drains performance heavily)
            if total_view:
                ax_total.imshow(total_mx)
                ax_total.set_title('total')

        mx_ctr += 1

    # only total at the end
    # fig, ax = plt.subplots(1, 1)
    # ax.imshow(total_mx)
    # ax.set_title(measure_name)
    # plt.close(fig)
    return feature_cont


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
