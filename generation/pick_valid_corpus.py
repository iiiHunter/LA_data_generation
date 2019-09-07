import json
import os


def pick():
    '''
    筛选符合条件的预料，段落数量不宜太长
    :return:
    '''
    with open("/home/sol/data/HEHE/code/data_generation/data/corpus.json", "r") as f:
        corpus = json.loads(f.read())
    valid_page = []
    for k in corpus.keys():
        for one_page in corpus[k]:
            flag = 0
            for one_para in one_page:
                if len(one_para) > 200:
                    flag = 1
                    break
            if not flag:
                valid_page.append(one_page)
    valid_page.sort(key=lambda x: len(x))
    return valid_page


if __name__ == "__main__":
    pick()
