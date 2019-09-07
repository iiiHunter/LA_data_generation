#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2019/8/22 上午10:20
# @Author : hesuo_zhang
# @Project : data_generation
# @File : paragraphs_generation.py
# @E-mail: hesuo_zhang@intsig.net

from generation.pick_valid_corpus import pick
import random
from generation.lines_generation import LinesGeneration
import cv2
import yaml
import copy
import json
import os
import numpy as np
from generation.insert_image import insert_images
from generation.post_processing.rotate import rotate
from generation.post_processing.add_noise import add_noise


IMAGESIZE = 48
TITLE_IMAGESIZE = 80
# np.random.seed(123)
# random.seed = 123


class ParagraphsGeneration(object):
    def __init__(self, config):
        self.corpus = pick()  # list[page, page, ...], page: list[para, para, ...]
        self.corpus_length = len(self.corpus)
        self.config = config

    @staticmethod
    def paragraphs_divider(page):
        '''
        divide all the paragraphs into several page cut, that is, in one cut, their layout is same.
        :param page: one corpus, a list of several paragraphs
        :return: divided corpus, a list of several cut, a cut is a list of several paragraphs
        '''
        after_divide = []

        # determine how to cut page
        random_list = []
        for i in range(len(page)):
            random_list.append(random.randint(0, 6))
        print(random_list)

        # cut page, the paragraph with non-zero/zero index in random list belong to one cut
        sub_paras = []
        flag = 0
        for i, (judge, paragraph) in enumerate(zip(random_list, page)):
            if judge:
                sub_paras.append(paragraph)
            else:
                sub_paras2 = [paragraph]
                flag = 1
            if flag:
                if sub_paras:
                    after_divide.append(sub_paras)
                    sub_paras = []
                after_divide.append(sub_paras2)
                flag = 0
            if i == len(random_list) - 1 and sub_paras:
                after_divide.append(sub_paras)
        return after_divide

    @staticmethod
    def save(gt, label_save_path, img_np, img_save_path):
        with open(label_save_path, "w") as f:
            f.write(json.dumps(gt))
        cv2.imwrite(img_save_path, img_np)
        return gt

    @staticmethod
    def unpack(page, cut_coor_in_page_copy):
        location_list = []
        text_list = []
        char_label_list = []
        for cut, cut_coor in zip(page, cut_coor_in_page_copy):
            for layout, layout_coor in zip(cut, cut_coor):
                for para, para_coor in zip(layout, layout_coor):
                    _, label, char_label = para
                    location_list.append(para_coor)
                    text_list.append(label)
                    # new_char_label = []
                    # for char_l in char_label:
                    #     new_char_label.append([char_l[0] + para_coor[0], char_l[1] + para_coor[1], char_l[2], char_l[3]])
                    char_label_list.append(char_label)
        gt = dict()
        gt["location"] = location_list
        gt["text"] = text_list
        gt["char_location"] = char_label_list
        return gt

    @staticmethod
    def vis(gt, image, path):
        for para_coor, char_coor_list in zip(gt["location"], gt["char_location"]):
            cv2.rectangle(image, (para_coor[0], para_coor[1]),
                          (para_coor[0] + para_coor[2], para_coor[1] + para_coor[3]), (255, 0, 0), 3)
            for char_coor in char_coor_list:
                char, char_coor = char_coor
                char_coor = [int(item) for item in char_coor]
                cv2.rectangle(image, (char_coor[0], char_coor[1]),
                              (char_coor[0] + char_coor[2], char_coor[1] + char_coor[3]), (0, 255, 0), 1)
        cv2.imwrite(path, image)

    @staticmethod
    def vis_polygon(gt, image, path):
        for para_coor, char_coor_list in zip(gt["location"], gt["char_location"]):
            para_coor = [int(item) for item in para_coor]
            para_coor = np.reshape(np.array(para_coor, dtype=np.int32), [4, 2])
            cv2.polylines(image, np.int32([para_coor]), 1, (255, 0, 0), 2)
            for char_coor in char_coor_list:
                char, char_coor = char_coor
                char_coor = [int(item) for item in char_coor]
                char_coor = np.reshape(np.array(char_coor, dtype=np.int32), [4, 2])
                cv2.polylines(image, np.int32([char_coor]), 1, (0, 255, 0), 1)
        cv2.imwrite(path, image)

    @staticmethod
    def get_char_absolute_location(gt):
        absolute_char_location_list = []
        for para_coor, char_coor in zip(gt["location"], gt["char_location"]):
            absolute_char_coor = []
            for one_char_coor in char_coor:
                char, one_char_coor = one_char_coor
                absolute_char_coor.append([char, [one_char_coor[0] + para_coor[0], one_char_coor[1] + para_coor[1],
                                                  one_char_coor[2], one_char_coor[3]]])
            absolute_char_location_list.append(absolute_char_coor)
        gt["char_location"] = absolute_char_location_list
        return gt

    @staticmethod
    def get_compact_para_box(gt):
        new_para_coor_list = []
        for para_coor, char_coor in zip(gt["location"], gt["char_location"]):
            x_min, y_min, x_max, y_max = 9999, 9999, 0, 0
            for one_char_coor in char_coor:
                _, one_char_coor = one_char_coor
                if one_char_coor[0] < x_min:
                    x_min = one_char_coor[0]
                if one_char_coor[1] < y_min:
                    y_min = one_char_coor[1]
                if one_char_coor[0] + one_char_coor[2] > x_max:
                    x_max = one_char_coor[0] + one_char_coor[2]
                if one_char_coor[1] + one_char_coor[3] > y_max:
                    y_max = one_char_coor[1] + one_char_coor[3]
            new_para_coor_list.append([x_min, y_min, x_max - x_min, y_max - y_min])
        gt["location"] = new_para_coor_list
        return gt

    def resize_img(self, gt, img_np):
        max_width = config["page_parameter"]["max_width"]
        h, w, c = img_np.shape
        if w < max_width:
            return gt, img_np
        SCALE = w/max_width
        img_np_des = cv2.resize(img_np, (int(max_width), int(h/SCALE)), interpolation=cv2.INTER_CUBIC)
        new_gt_coor_list = []
        new_gt_char_coor_list = []
        for para_coor, char_coor_list in zip(gt["location"], gt["char_location"]):
            x, y, w, h = para_coor
            char_coor_list = char_coor_list
            new_gt_coor_list.append([int(x/SCALE), int(y/SCALE), int(w/SCALE), int(h/SCALE)])
            new_char_coor_list = []
            for char_coor in char_coor_list:
                char, char_coor = char_coor
                x, y, w, h = char_coor
                new_char_coor_list.append([char, [int(x/SCALE), int(y/SCALE), int(w/SCALE), int(h/SCALE)]])
            new_gt_char_coor_list.append(new_char_coor_list)

        gt["location"] = new_gt_coor_list
        gt["char_location"] = new_gt_char_coor_list
        return gt, img_np_des

    def add_margin(self, gt, img_np):
        margin_left = self.config["page_parameter"]["margin_left"]
        margin_right = self.config["page_parameter"]["margin_right"]
        margin_top = self.config["page_parameter"]["margin_top"]
        margin_bottom = self.config["page_parameter"]["margin_bottom"]
        h, w, c = img_np.shape
        left_np = 255 * np.ones([h, margin_left, c])
        right_np = 255 * np.ones([h, margin_right, c])
        top_np = 255 * np.ones([margin_top, margin_left + w + margin_right, c])
        bottom_np = 255 * np.ones([margin_bottom, margin_left + w + margin_right, c])
        merge_h = np.hstack([left_np, img_np, right_np])
        merge_v = np.vstack([top_np, merge_h, bottom_np])
        new_gt_coor_list = []
        new_gt_char_coor_list = []
        for coor, char_coor_list in zip(gt["location"], gt["char_location"]):
            x, y, w, h = coor
            new_gt_coor_list.append([x + margin_left, y + margin_top, w, h])
            new_char_coor_list = []
            for char_coor in char_coor_list:
                char, char_coor = char_coor
                new_char_coor_list.append([char, [char_coor[0] + margin_left, char_coor[1] + margin_top, char_coor[2], char_coor[3]]])
            new_gt_char_coor_list.append(new_char_coor_list)
        gt["location"] = new_gt_coor_list
        gt["char_location"] = new_gt_char_coor_list
        return gt, merge_v

    def limit_max_length(self, chosen_page):
        total_char_length = 0
        for item in chosen_page:
            total_char_length += len(item)
        if total_char_length < self.config["page_parameter"]["max_char_num_in_one_page"]:
            return chosen_page
        cut_chosen_page = []
        current_length = 0
        for item in chosen_page:
            current_length += len(item)
            if current_length < config["page_parameter"]["max_char_num_in_one_page"]:
                cut_chosen_page.append(item)
            else:
                break
        return cut_chosen_page

    def insert_image_symbols(self, chosen_page):
        image_num = config["page_parameter"]["image_num"]
        if not image_num:
            return chosen_page
        image_spaceholders = ["$" * self.config["page_parameter"][k] for k in ["image1_size", "image2_size", "image3_size", "image4_size"]]
        for spaceholder in image_spaceholders[:image_num]:
            chosen_page.insert(random.randint(0, len(chosen_page)), spaceholder)
        return chosen_page

    def generate_para_np(self):
        '''
        general all the text lines in one page and then combined the lines belong to the same text block
        :return:
        '''
        chosen_indx = random.randint(0, self.corpus_length-1)
        chosen_page = self.corpus[chosen_indx]
        chosen_page = self.insert_image_symbols(chosen_page)
        print(chosen_page)
        chosen_page = self.limit_max_length(chosen_page)
        after_divide = self.paragraphs_divider(chosen_page)
        # determine how many layouts there are for a specific page cut

        random_layout_list = []
        for _ in range(len(after_divide)):
            random_one = random.randint(1, self.config["page_parameter"]["max_layout_num"] - 1)
            if random_layout_list:
                while random_one != 1 and (random_one == random_layout_list[-1]):
                    random_one = random.randint(1, self.config["page_parameter"]["max_layout_num"] - 1)
            random_layout_list.append(random_one)

        print(random_layout_list)
        self.config["page_parameter"]["random_layout_list"] = random_layout_list

        # generate lines
        lg = LinesGeneration(after_divide, self.config)
        all_lines = lg.generate()
        assert len(all_lines) == len(random_layout_list)

        page = []
        for random_layout, lines_per_cut in zip(random_layout_list, all_lines):
            # each page cut

            # determine the line num in a specific layout
            lines_in_one_cut = []
            for lines_per_layout in lines_per_cut:
                for line in lines_per_layout:
                    lines_in_one_cut.append(line)
            full_layout_line_num = (len(lines_in_one_cut) + random.randint(0, 10)) // random_layout + 1
            if not full_layout_line_num:
                full_layout_line_num = 1

            # determine the text lines each text block contains
            count_indx = 0
            mark_para_indx_list = [0]
            mark_layout_indx_list = [0]
            for lines_per_layout in lines_per_cut:
                for line in lines_per_layout:
                    lines_in_one_cut.append(line)
                    if (count_indx > 0) and ((count_indx + 1) % full_layout_line_num == 0):
                        mark_layout_indx_list.append(count_indx)
                    count_indx += 1
                mark_para_indx_list.append(count_indx)
            mark_indx_list = mark_para_indx_list + mark_layout_indx_list
            mark_indx_list = list(set(mark_indx_list))  # remove repeated
            mark_indx_list.sort()

            layout_in_cut = []
            para_in_layout = []
            for i in range(len(mark_indx_list)-1):
                para_in_layout.append(lines_in_one_cut[mark_indx_list[i]: mark_indx_list[i + 1]])
                if mark_indx_list[i + 1] in mark_layout_indx_list:
                    layout_in_cut.append(para_in_layout)
                    para_in_layout = []
            layout_in_cut.append(para_in_layout)

            # combine the text lines in one text block
            layouts = []
            for layout in layout_in_cut:
                blocks = []
                for para in layout:
                    para_only_img = []
                    para_only_label = ""
                    para_char_label = []
                    for line_indx, line in enumerate(para):
                        para_only_img.append(line[0])
                        para_only_img.append(255 * np.ones([self.config["page_parameter"]["interval_between_lines"],
                                                            line[0].shape[1], line[0].shape[2]]))
                        para_only_label += line[1]
                        for coor in line[2]:
                            char, coor = coor
                            new_coor = [coor[0], coor[1] + (self.config["page_parameter"]["interval_between_lines"] +
                                                            IMAGESIZE) * line_indx, coor[2], coor[3]]
                            para_char_label.append([char, new_coor])
                    para_only_img = np.vstack(para_only_img)
                    blocks.append([para_only_img, para_only_label, para_char_label])
                layouts.append(blocks)
            page.append(layouts)
        return page

    def generate(self, id):
        page_np_list = self.generate_para_np()
        cut_coor_in_page = []
        page_combined = []
        for i, cut_np_list in enumerate(page_np_list):
            layout_coor_in_cut = []
            cut_list = []
            for j, layout_np_list in enumerate(cut_np_list):
                layout_with_interval = []
                block_coor_in_layout = []
                to_append_coor = [0, 0, 0, 0]
                for k, block in enumerate(layout_np_list):
                    block_np, _, _ = block
                    h, w, c = block_np.shape
                    blank_between_block = 255 * np.ones([self.config["page_parameter"]["interval_between_paragraphs"], w, c])
                    layout_with_interval.append(block_np)
                    layout_with_interval.append(blank_between_block)
                    if k == 0:
                        to_append_coor = [0, 0, w, h]
                    else:
                        to_append_coor = [0, to_append_coor[1] + to_append_coor[3] +
                                          self.config["page_parameter"]["interval_between_paragraphs"], w, h]
                    block_coor_in_layout.append(to_append_coor)
                layout_combined = np.vstack(layout_with_interval)
                cut_list.append(layout_combined)
                layout_coor_in_cut.append(block_coor_in_layout)

            max_h = max([item.shape[0] for item in cut_list])
            padding_cut_combined = []
            layout_coor_in_cut_copy = copy.deepcopy(layout_coor_in_cut)
            for index, (layout_coor, item) in enumerate(zip(layout_coor_in_cut, cut_list)):
                h_tmp, w_tmp, c_tmp = item.shape
                if h_tmp < max_h:
                    new_item = np.vstack([item, 255 * np.ones([max_h - h_tmp, w_tmp, c_tmp])])
                else:
                    new_item = item
                padding_cut_combined.append(new_item)
                padding_cut_combined.append(255 * np.ones([max_h, config["page_parameter"]["interval_between_layouts"], c_tmp]))

                if index == 0:
                    w_bias = 0
                else:
                    w_bias = sum([layout_coor_in_cut[jj][0][2] for jj in range(index)])
                for k, para in enumerate(layout_coor):
                    x_min, y_min, w, h = para
                    new_x_min = self.config["page_parameter"]["interval_between_layouts"] * index + x_min + w_bias
                    new_y_min = y_min
                    layout_coor_in_cut_copy[index][k] = [new_x_min, new_y_min, w, h]

            cut_combined = np.hstack(padding_cut_combined)
            page_combined.append(cut_combined)
            cut_coor_in_page.append(layout_coor_in_cut_copy)

        max_w = max(item.shape[1] for item in page_combined)
        padding_page_combined = []
        cut_coor_in_page_copy = copy.deepcopy(cut_coor_in_page)
        for index, (cut_coor, item) in enumerate(zip(cut_coor_in_page, page_combined)):
            h_tmp, w_tmp, c_tmp = item.shape
            if w_tmp < max_w:
                new_item = np.hstack([item, 255 * np.ones([h_tmp, max_w - w_tmp, c_tmp])])
            else:
                new_item = item
            padding_page_combined.append(new_item)
            padding_page_combined.append(255 * np.ones([config["page_parameter"]["interval_between_paragraphs"], max_w, c_tmp]))

            if index == 0:
                h_bias = 0
            else:
                max_h_list = []
                for jj in range(index):
                    l_tmp = []
                    for layout_ in cut_coor_in_page[jj]:
                        l_tmp.append(layout_[-1][1] + layout_[-1][3])
                    max_h = max(l_tmp)
                    max_h_list.append(max_h)
                h_bias = sum(max_h_list) + self.config["page_parameter"]["interval_between_paragraphs"] * index

            for j, layout_coor in enumerate(cut_coor):
                for k, para in enumerate(layout_coor):
                    x_min, y_min, w, h = para
                    new_x_min = x_min
                    new_y_min = y_min + self.config["page_parameter"]["interval_between_paragraphs"] * index + h_bias
                    cut_coor_in_page_copy[index][j][k] = [new_x_min, new_y_min, w, h]

        page_combined = np.vstack(padding_page_combined)
        gt = self.unpack(page_np_list, cut_coor_in_page_copy)
        gt = self.get_char_absolute_location(gt)
        gt = self.get_compact_para_box(gt)
        gt, img_np = self.add_margin(gt, page_combined)
        gt, img_np = self.resize_img(gt, img_np)
        gt, img_np = insert_images(gt, img_np)
        gt, img_np = rotate(gt, img_np)
        img_np = add_noise(img_np)
        self.save(gt, os.path.join(self.config["save_path"]["label_save_path"], str(id) + ".json"),
                  img_np, os.path.join(self.config["save_path"]["image_save_path"], str(id) + self.config["image_format"]))
        self.vis_polygon(gt, img_np, os.path.join(self.config["save_path"]["vis_save_path"], str(id) + self.config["image_format"]))


def random_generator(config):
    random_keys = []
    for k in config["page_parameter"].keys():
        if isinstance(config["page_parameter"][k], list):
            random_keys.append(k)
    for k in random_keys:
        config["page_parameter"][k] = random.randint(config["page_parameter"][k][0], config["page_parameter"][k][1])
    return config


if __name__ == "__main__":
    random_type_num = 1000
    num_per_type = 10
    for i in range(random_type_num):
        try:
            with open('configs/conf.yaml', encoding='utf-8') as f:
                config = yaml.load(f)
                print(config)
            config = random_generator(config)
            P = ParagraphsGeneration(config)
            for j in range(num_per_type):
                print("Processing random type %d: %d/%d" % (i, j, num_per_type))
                P.generate(id="%d_%d" % (i, j))
        except:
            print("Found Error when generating document images.")
            continue

    # with open('configs/conf.yaml', encoding='utf-8') as f:
    #     config = yaml.load(f)
    #     print(config)
    # config = random_generator(config)
    # P = ParagraphsGeneration(config)
    # P.generate(0)
