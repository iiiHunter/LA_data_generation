#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Project: data_generation
@Author: sol@JinGroup
@File: add_noise.py
@Time: 9/5/19 3:04 PM
@E-mail: hesuozhang@gmail.com
'''

import skimage
import numpy as np
import random
import cv2


def add_pepper_noise(im):
    out_im = skimage.util.random_noise(im, mode="pepper", amount=0.001 * random.randint(1, 10)) * 255
    return out_im


def corrde_image(im):
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(im, kernel)
    return erosion


def sharpen(im):
    im = np.array(im)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    dst = cv2.filter2D(im, -1, kernel=kernel)
    return dst


def add_noise(im):
    # im = add_pepper_noise(im)
    # im = corrde_image(im)
    im = sharpen(im)
    return im