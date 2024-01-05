import os.path
import traceback
import warnings
import xml.etree.ElementTree as et
import numpy as np

from Daten.Visualisierung import mylib

NS = {'zb': 'http://www.zebris.de/measurements'}


def get_cell_count(filepath, _id):  # not robustified yet
    movement = get_movement(filepath, _id)
    cc_tuple = movement.find('.//zb:cell_count', NS).text
    x = int(cc_tuple[0])
    y = int(cc_tuple[1])
    return y, x


def get_count(filepath, _id):  # get No. of matrices in one data set
    movement = get_movement(filepath, _id)
    return int(movement.find('.//zb:count', NS).text)


def get_movement(filepath, _id):
    root = et.parse(filepath).getroot()
    movements = root.findall('.//zb:movement', NS)
    for movement in movements:
        if movement.find('.//zb:id', NS).text == _id:
            return movement


def set_movement_type(filepath):
    print('file:', os.path.basename(filepath))
    if 'Schritt' in filepath:
        mylib.GAIT_TYPE = 0
    elif 'Trab' in filepath:
        mylib.GAIT_TYPE = 1
    else:
        warnings.warn('XML file misses gait type declaration')


def set_direction(movement):
    if movement.find('.//zb:type', NS).text == 'forward':
        mylib.DIRECTION = 1
        print('direction: forward')
    else:
        mylib.DIRECTION = 0
        print('direction: backward')


def extract_matrices(filepath, _id):
    set_movement_type(filepath)
    root = get_movement(filepath, _id)
    set_direction(root)

    # extract matrix data from xml
    quant_elements = root.findall(".//zb:data/zb:quant", NS)
    matrices = []  # temp. array for matrices from xml; may contain empty lists too
    cell_begin = []
    for quant_element in quant_elements:
        quant_data = {
            "cells": [
                list(map(float, cell_data.split())) for cell_data in
                quant_element.find("zb:cells", NS).text.splitlines()
            ],
        }
        matrices.append(quant_data)

        tmp_x = int(quant_element.find('.//zb:cell_begin/zb:x', NS).text)
        tmp_y = int(quant_element.find('.//zb:cell_begin/zb:y', NS).text)
        cell_begin.append([tmp_x, tmp_y])

    data = []  # final list with matrices

    for matrix in matrices:  # remove empty lists from 'cells'-field/array
        tmp_matrix = []
        for rows in matrix['cells']:
            if len(rows) >= 1:
                tmp_matrix.append(rows)
        data.append(tmp_matrix)

    return data, cell_begin


def create_global_mx(local_mx, offset):
    global_mx = np.zeros((481, 64))
    x_off = offset[1] - 1
    if x_off < 0: x_off = 0
    y_off = offset[0] - 1
    if y_off < 0: y_off = 0
    rows = local_mx.shape[0]
    cols = local_mx.shape[1]

    try:
        global_mx[-1 - x_off - rows:-1 - x_off, y_off:y_off + cols] = local_mx
    except ValueError:
        print(traceback.format_exc())
    return global_mx
