from PIL import Image, ImageDraw, ImageFont
import os
import json


# IMAGE_SIZE = 48
# y_bias = -15
# font1_size = 50
# font2_size = 40

IMAGE_SIZE = 80
y_bias1 = -25
y_bias2 = -28
font1_size = 83
font2_size = 60


def char_draw(save_dir, font_ttc, debug=False):
    '''
    通过指定字体生成单字字符图片
    :param save_dir: 单字图片保存目录
    :param font_ttc: 字体文件
    :param debug:
    :return:
    '''
    img_save_dir = os.path.join(save_dir, "all_char_imgs")
    if not os.path.exists(img_save_dir):
        os.mkdir(img_save_dir)
    search_dict_save_path = os.path.join(save_dir, "search_dict.json")
    font1 = ImageFont.truetype(font_ttc, font1_size)
    font2 = ImageFont.truetype(font_ttc, font1_size)
    chars1_txt = "/home/sol/data/HEHE/code/data_generation/data/all_chars.txt"  # Chinese characters
    chars2_txt = "/home/sol/data/HEHE/code/data_generation/data/all_chars3.txt"  # letters, digits and symbols
    search_dict = {}
    count = 0

    with open(chars1_txt, "r") as f:
        s = f.read()
    char_list = s.split(" ")
    for char in char_list:
        save_path = os.path.join(img_save_dir, str(count) + ".jpg")
        im = Image.new("L", [IMAGE_SIZE, IMAGE_SIZE], 255)
        draw = ImageDraw.Draw(im)
        draw.text([0, y_bias1], char, font=font1)
        im.save(save_path)
        search_dict[char] = save_path
        count += 1
        if debug:
            if count >= 9:
                exit(3)

    with open(chars2_txt, "r") as f:
        s2 = f.read()
    char_list2 = s2.split(" ")
    for char in char_list2:
        save_path = os.path.join(img_save_dir, str(count) + ".jpg")
        im = Image.new("L", [IMAGE_SIZE, IMAGE_SIZE], 255)
        draw = ImageDraw.Draw(im)
        draw.text([0, y_bias2], char, font=font2)
        im.save(save_path)
        search_dict[char] = save_path
        count += 1

    with open(search_dict_save_path, "w") as f:
        f.write(json.dumps(search_dict))


if __name__ == "__main__":
    save_dir = "/home/sol/data/HEHE/code/data_generation/data/font2"
    # font_ttc = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
    font_ttc = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    char_draw(save_dir, font_ttc, debug=False)
