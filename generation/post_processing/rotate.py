#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Project: data_generation
@Author: sol@JinGroup
@File: rotate.py
@Time: 9/5/19 10:34 AM
@E-mail: hesuozhang@gmail.com
'''

import math
from PIL import Image
import cv2
import numpy as np
import random


def rotate_np(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]

    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)

    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h), borderValue=(255, 255, 255))

    # 返回旋转后的图像
    return rotated, M


# def rotate_coordinate(angle, rect, w, h, x_offs, y_offs):
#     rect = [rect[0] + x_offs, rect[1] + y_offs,
#             rect[0] + x_offs + rect[2], rect[1] + y_offs,
#             rect[0] + x_offs + rect[2], rect[1] + y_offs + rect[3],
#             rect[0] + x_offs, rect[1] + y_offs + rect[3]]
#     angle = angle * math.pi / 180
#     n = w  # w
#     m = h  # h
#
#     def onepoint(x, y):
#         X = x * math.cos(angle) - y * math.sin(angle) - 0.5 * n * math.cos(angle) + 0.5 * m * math.sin(angle) + 0.5 * n
#         Y = y * math.cos(angle) + x * math.sin(angle) - 0.5 * n * math.sin(angle) - 0.5 * m * math.cos(angle) + 0.5 * m
#         return [int(X), int(Y)]
#
#     newrect = []
#     for i in range(4):
#         point = onepoint(rect[i * 2], rect[i * 2 + 1])
#         newrect.extend(point)
#     return newrect


def rotate(gt, image):
    H, W, C = image.shape
    angle = int(5/(H/W))
    if not angle:
        angle = 1
    angle = random.randrange(-angle, angle)
    IM_r, mat_rotation = rotate_np(image, -angle)

    new_para_coor = []
    new_char_coor = []
    for para_coor, char_coor_list in zip(gt["location"], gt["char_location"]):
        x, y, w, h = para_coor
        para_coor = [x, y, x+w, y, x+w, y+h, x, y+h]
        new_para_coor_one = []
        for i in range(4):
            Q = np.dot(mat_rotation, np.array([[para_coor[i*2]], [para_coor[i*2 + 1]], [1]]))
            new_para_coor_one.extend(Q.reshape([2]).tolist())
        new_para_coor.append(new_para_coor_one)

        new_char_coor_list = []
        for char_coor in char_coor_list:
            char, char_coor = char_coor
            x0, y0, w0, h0 = char_coor
            char_coor = [x0, y0, x0 + w0, y0, x0 + w0, y0 + h0, x0, y0 + h0]
            new_char_coor_list_one = []
            for i in range(4):
                Q = np.dot(mat_rotation, np.array([[char_coor[i * 2]], [char_coor[i * 2 + 1]], [1]]))
                new_char_coor_list_one.extend(Q.reshape([2]).tolist())
            new_char_coor_list.append([char, new_char_coor_list_one])
        new_char_coor.append(new_char_coor_list)

    gt["location"] = new_para_coor
    gt["char_location"] = new_char_coor
    return gt, IM_r
