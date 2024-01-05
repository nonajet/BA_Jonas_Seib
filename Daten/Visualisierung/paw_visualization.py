import gc
import logging
import time
import traceback

import numpy as np
from matplotlib import pyplot as plt

from Daten.Visualisierung.extract_xml import extract_matrices, create_global_mx
from paw_detection import paw_recognition, TheDog


def visualize(filepath, _id, mx_start=0, total_view=False, mx_skip=1):
    t1 = time.time()
    # np.set_printoptions(threshold=sys.maxsize)
    logger = logging.getLogger()

    matrix, offset = extract_matrices(filepath, _id)
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

    mx_ctr = mx_start
    for mx in matrix[mx_start:]:
        if mx:
            print('\n######################')
            print('mx_ctr:', mx_ctr)
            mx_np = np.array(mx)
            global_mx = create_global_mx(mx_np, offset[mx_ctr])
            paw_recognition(mx_np, offset[mx_ctr], global_mx)
            # TODO: correct logger?
            logger.info('\n###################### id: %i ######################' % mx_ctr)
            vis_paws(fig_paws, axes_paws)

            if mx_ctr % mx_skip == 0:
                # local
                ax_local.imshow(mx_np)
                ax_local.set_axis_off()
                # ax_local.imshow(np.flipud(np.fliplr(mx_np)))  # rot. 180° to fit glob. view direction

                # global
                ax_global.imshow(global_mx)
                ax_global.set_axis_off()

                if total_view:
                    # total (drains performance heavily)
                    total_mx += global_mx
                    ax_total.imshow(total_mx)

                plt.pause(0.000001)
                gc.collect()
        mx_ctr += 1

    t2 = time.time()
    print('\n\nduration:', t2 - t1)


def vis_paws(figure, axes):
    plt.ion()
    plt.figure(figure)

    for paw_obj in TheDog.paws:
        try:
            axes[paw_obj.ax_ind].matshow(paw_obj.area)
            axes[paw_obj.ax_ind].set_title(paw_obj.name)
            axes[paw_obj.ax_ind].set_axis_off()
        except TypeError:
            print(traceback.format_exc())
            print('err in:', paw_obj.area)

    plt.pause(0.000001)
