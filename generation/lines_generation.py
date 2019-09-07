#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2019/8/22 上午11:24
# @Author : hesuo_zhang
# @Project : data_generation
# @File : lines_generation.py
# @E-mail: hesuo_zhang@intsig.net


import json
import random
import cv2
import numpy as np

IMAGESIZE = 48
TITLE_IMAGESIZE = 80


class LinesGeneration(object):
    def __init__(self, parapraphs, config):
        self.parapraphs = parapraphs
        self.config = config
        self.char_num_per_lines = config["page_parameter"]["char_num_per_lines"]
        self.char_interval = config["page_parameter"]["char_interval"]
        with open(config["font"]["font_images_search_dict"], "r") as f_search:
            self.search_dict = json.loads(f_search.read())
        with open(config["font"]["title_font_images_search_dict"], "r") as f_title_search:
            self.title_search_dict = json.loads(f_title_search.read())
        self.indent_fisrt_line = []
        for i in range(len(self.parapraphs)):
            self.indent_fisrt_line.append(random.randint(0, 1))
        self.random_layout_list = config["page_parameter"]["random_layout_list"]

    @staticmethod
    def end_with_symbol(s):
        with open("/home/sol/data/HEHE/code/data_generation/data/all_chars3.txt", "r") as f:
            symbols = f.read()
        if s[-1] in symbols:
            return True
        else:
            return False

    def generate_images(self, s, char_num_per_lines, indent=0, line_num=0):
        if line_num == 1 and not self.end_with_symbol(s):
            title = True
            current_imagesize = TITLE_IMAGESIZE
        else:
            title = False
            current_imagesize = IMAGESIZE
        if title:
            search_dict = self.title_search_dict
        else:
            search_dict = self.search_dict
        label = []
        char_label = []
        char_np_list = []
        if indent:
            char_np_list.append(255 * np.ones([current_imagesize, current_imagesize, 3]))
            char_np_list.append(255 * np.ones([current_imagesize, self.char_interval, 3]))
            char_np_list.append(255 * np.ones([current_imagesize, current_imagesize, 3]))
            char_np_list.append(255 * np.ones([current_imagesize, self.char_interval, 3]))
        jump_indx = 0
        for char_indx, char in enumerate(s):
            if not char in search_dict.keys():
                jump_indx += 1
                continue
            char_np_list.append(cv2.imread(search_dict[char]))
            char_np_list.append(255 * np.ones([current_imagesize, self.char_interval, 3]))
            label.append(char)
            if indent:
                char_label.append([char, [(char_indx + 2 - jump_indx) * (current_imagesize + self.char_interval), 0,
                                          current_imagesize, current_imagesize]])
            else:
                char_label.append([char, [(char_indx - jump_indx) * (current_imagesize + self.char_interval), 0,
                                          current_imagesize, current_imagesize]])
        current_w = (current_imagesize + self.char_interval) * len(char_np_list)/2
        max_w = char_num_per_lines * (IMAGESIZE + self.char_interval)
        while current_w < max_w:
            char_np_list.append(255 * np.ones([current_imagesize, current_imagesize, 3]))
            char_np_list.append(255 * np.ones([current_imagesize, self.char_interval, 3]))
            current_w = current_w + current_imagesize + self.char_interval

        line_np = np.hstack(char_np_list)
        if current_w > max_w:
            scale = float(max_w)/current_w
            line_np = cv2.resize(line_np, (int(max_w), int(line_np.shape[0] * scale)))
            char_label_resized = []
            for item in char_label:
                char_label_resized.append([item[0], [it * scale for it in item[1]]])
        else:
            char_label_resized = char_label
        line_np = line_np.astype(np.uint8)
        label = "".join(label)
        return [line_np, label, char_label_resized]

    def generate(self):
        parapraphs_list = []
        for indent, parapraph, layout_num in zip(self.indent_fisrt_line, self.parapraphs, self.random_layout_list):
            # each page cut
            # char_num_per_lines = int(self.char_num_per_lines/layout_num) - 1
            if layout_num > 1:
                # make sure the sum width of page cut with multiple layout is the same the width of the page cut
                # with one layout
                char_num_per_lines = int((self.char_num_per_lines - (layout_num - 1) *
                                          self.config["page_parameter"]["interval_between_layouts"]/IMAGESIZE)/layout_num)
            else:
                char_num_per_lines = self.char_num_per_lines
            sub_parapraphs_list = []
            for sub_paragraph in parapraph:
                # each paragraph in one page cut
                line_num = len(sub_paragraph)//char_num_per_lines + 1
                lines = []
                for line_indx in range(line_num):
                    if indent:
                        if line_indx == 0:
                            s = sub_paragraph[:char_num_per_lines - 2]
                            line_np = self.generate_images(s, char_num_per_lines, indent, line_num=line_num)
                        else:
                            s = sub_paragraph[line_indx * char_num_per_lines - 2:
                                              (line_indx + 1) * char_num_per_lines - 2]
                            line_np = self.generate_images(s, char_num_per_lines, line_num=line_num)
                    else:
                        s = sub_paragraph[line_indx * char_num_per_lines:(line_indx+1) * char_num_per_lines]
                        line_np = self.generate_images(s, char_num_per_lines, indent, line_num=line_num)
                    lines.append(line_np)
                sub_parapraphs_list.append(lines)
            parapraphs_list.append(sub_parapraphs_list)
        return parapraphs_list


if __name__ == "__main__":
    gray_path = "/home/sol/data/HEHE/code/data_generation/data/all_char_imgs3/gray.jpg"
    gray_img = np.ones([IMAGESIZE, IMAGESIZE, 3]) * 255
    cv2.imwrite(gray_path, gray_img)
    search_dict_path = "/home/sol/data/HEHE/code/data_generation/data/search_dict_all.json"
    with open(search_dict_path, "r") as f:
        d = json.loads(f.read())
    d["$"] = gray_path
    with open(search_dict_path, "w") as f:
        f.write(json.dumps(d))
