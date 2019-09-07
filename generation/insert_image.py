#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Project: data_generation
@Author: sol@JinGroup
@File: insert_image.py
@Time: 9/4/19 3:21 PM
@E-mail: hesuozhang@gmail.com
'''


import cv2
import os
import random
import numpy as np


def image_patse(image, para_coor):
    image_dir = "/home/sol/data/Project/general_text_det/data/coco_val2017"
    image_path = [os.path.join(image_dir, item) for item in os.listdir(image_dir)]
    random.shuffle(image_path)
    im = image_path[0]
    im = cv2.imread(im)
    margin_left = random.randint(5, 15)
    margin_top = random.randint(5, 15)
    im = np.hstack([255 * np.ones([im.shape[0], margin_left, im.shape[2]]), im])
    im = np.vstack([255 * np.ones([margin_top, im.shape[1], im.shape[2]]), im])
    x = para_coor[0] + margin_left
    y = para_coor[1] + margin_top
    w = para_coor[2] - margin_left
    h = para_coor[3] - margin_top

    # in y direction
    if im.shape[0] < para_coor[3]:
        blank = 255 * np.ones([para_coor[3] - im.shape[0], im.shape[1], im.shape[2]])
        h = im.shape[0] - margin_top
        im = np.vstack([im, blank])

    # in x direction
    if im.shape[1] < para_coor[2]:
        blank = 255 * np.ones([im.shape[0], para_coor[2] - im.shape[1], im.shape[2]])
        w = im.shape[1] - margin_left
        im = np.hstack([im, blank])

    new_para_coor = [x, y, w, h]
    image[para_coor[1]:para_coor[1]+para_coor[3], para_coor[0]:para_coor[0]+para_coor[2], :] = im[:para_coor[3], :para_coor[2], :]
    return image, new_para_coor


def insert_images(gt, image):
    new_coor_list = []
    new_char_coor_list = []
    new_text_list = []
    for para_coor, char_coor, text in zip(gt["location"], gt["char_location"], gt["text"]):
        if "$" in text:
            image, new_para_coor = image_patse(image, para_coor)
            new_coor_list.append(new_para_coor)
            new_char_coor_list.append([])
            new_text_list.append("[Image]")
        else:
            new_coor_list.append(para_coor)
            new_char_coor_list.append(char_coor)
            new_text_list.append(text)
    gt["location"] = new_coor_list
    gt["char_location"] = new_char_coor_list
    gt["text"] = new_text_list
    return gt, image
