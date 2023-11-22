import sys
import traceback
import xml.etree.ElementTree as et

import numpy as np
import matplotlib.pyplot as plt

NAMESPACE = {'zb': 'http://www.zebris.de/measurements'}


def get_cell_count(filepath,
                   id):  # not robustified yet [IMPORTANT: returns uppermost field 'cell_count' from one movement]
    root = get_movement(filepath, id)
    x = -1
    y = -1
    if root.findall('zb:id', NAMESPACE)[0].text == id:
        cc = root.findall('.//zb:clip/zb:cell_count', NAMESPACE)
        x = int(cc[0][0].text)
        y = int(cc[0][1].text)
    return y, x  # return inverted tuple


def get_count(filepath, id):  # get No. of matrices in one data set
    root = get_movement(filepath, id)
    if root.findall('id')[0].text == id:
        return int(root.findall('.//clip/count')[0].text)


def get_movement(filepath, id):
    et.register_namespace("zb", "http://www.zebris.de/measurements")  # hardcoded URI
    root = et.parse(filepath).getroot()

    try:
        for movement in root.findall('.//zb:movements/zb:movement', NAMESPACE):
            # print('mvmt.tag: ', movement.find('.//zb:id', NAMESPACE).text)
            if movement.find('.//zb:id', NAMESPACE).text == id:
                print('id found: ', movement.tag)
                return movement
    except AttributeError:
        print(traceback.format_exc())
        # sys.exit('error in XML structure (get_movement)')


def extract_matrices(filepath, id):
    cell_begin = []
    data = []  # final lists with matrices or cell_begin

    movement = get_movement(filepath, id)
    # extract matrix data from xml
    try:
        quant_elements = movement.findall('.//zb:data/zb:quant', NAMESPACE)
    except AttributeError:
        print(traceback.format_exc())
        sys.exit('error in XML structure (extract_matrices)')

    matrices = []  # temp. array for matrices from xml; may contain empty lists too

    for quant_element in quant_elements:
        quant_data = {
            "cells": [
                list(map(float, cell_data.split())) for cell_data in
                quant_element.find('zb:cells', NAMESPACE).text.splitlines()
            ],
        }
        matrices.append(quant_data)

        cell_begin_level = quant_element.find('zb:cell_begin', NAMESPACE)
        x = int(cell_begin_level[0].text)
        y = int(cell_begin_level[1].text)
        cell_begin.append((x, y))

    for matrix in matrices:  # remove empty lists from 'cells'-field/array
        tmp_matrix = []
        for rows in matrix['cells']:
            if len(rows) >= 1:
                tmp_matrix.append(rows)
        data.append(tmp_matrix)

    return data, cell_begin


def global_max(matrix):  # not tested or used yet; used for finding global max in one or more matrices
    gmax = -1
    for mx in matrix:
        for row in mx:
            if max(row) > global_max:
                gmax = max(row)
    return gmax


def visualize(filepath, id):
    # np.set_printoptions(threshold=sys.maxsize)  # turn off matrix truncate when printing
    print(id)
    matrix, cell_begin = extract_matrices(filepath, id)
    if len(matrix) != len(
            cell_begin):  # sizes must be same since every (local) matrix has own offset (in global)
        print('matrix size (%i) not cell_begin size: (%i)' % (len(matrix), len(cell_begin)))

    fig, ax = plt.subplots(1, 3)
    ax_local = ax[0]

    ax_global = ax[1]
    cell_count = get_cell_count(filepath, id)

    total_mx = np.zeros(cell_count)
    ax_total = ax[2]
    ax_total.set_title('total')

    plt.ion()
    mx_count = 0  # keeps track of No. of matrix that is processed -> equivalent in cell_begin (e.g. offset matrix)
    for mx in matrix:
        if mx and mx_count % 4 == 0:  # assert mx is non '[]' // mod-operator can 'set' simulation pace (e.g. skip
            # data points)
            # transform matrix into numpy matrix for better handling
            mx_np = np.array(
                mx)  # np.flipud(np.fliplr(np.array(mx)))  # transform to np.matrix and rotate 180° for better vis.

            ax_local.matshow(mx_np)
            ax_local.set_title('local')

            mx_np_global = create_global_mx(mx_np, cell_count, cell_begin[mx_count])
            ax_global.matshow(mx_np_global)
            ax_global.set_title('global')

            # total_mx += mx_np_global
            # ax_total.matshow(total_mx)
            # ax_total.set_title('total')

            plt.pause(0.0001)

            ax_local.clear()
            ax_global.clear()

        mx_count += 1

    print(
        '################################################################################\nfinished vis. of movement:',
        id, '\n################################################################################')


def create_global_mx(matrix, zero_mx_size,
                     offset):  # calc global mx (whole mat) depending on offset of given (local) matrix from XML data
    if offset[1] > 1:  # when x/y_off == 0 might throw ValueError due to failed matrix slicing
        x_off = offset[1] - 1
    else:
        x_off = 1
    if offset[0] > 1:
        y_off = offset[0] - 1
    else:
        y_off = 1

    zeros = np.zeros(zero_mx_size)
    try:
        zeros[-x_off - matrix.shape[0]:-x_off, y_off:y_off + matrix.shape[1]] = matrix
    except IndexError:
        print('xoff:', x_off, 'yoff:', y_off)
    except ValueError:
        print('input: ', matrix.shape, 'xoff:', x_off, 'yoff:', y_off)

    return zeros


if __name__ == '__main__':
    filep = r'C:\Users\jonas\OneDrive\Desktop\Studium_OvGU\WiSe23_24\BA\Daten\Rohdaten\T0307068 Trab.xml'
    visualize(filep, 'gait_2')
