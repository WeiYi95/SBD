# -*- coding:utf-8 -*-
# @Author: Wei Yi

import random
import os

MAX_LEN = 62  # 由于模型需要[CLS]和[SEP]，所以数据的最大长度=模型最大长度-2

PUNC_TYPE = {'，': 0, '。': 1, '？': 2, '！': 3, '、': 4, '：': 5, '；': 6}
PUNC_SYM = [("[B_,] ", "[I_,] "), ("[B_.] ", "[I_.] "), ("[B_?] ", "[I_?] "), ("[B_!] ", "[I_!] "),
            ("[B_`] ", "[I_`] "), ("[B_:] ", "[I_:] "), ("[B_;] ", "[I_;] ")]


def get_label(punc):
    return PUNC_SYM[PUNC_TYPE[punc]]


def to_label(line, punc):
    length = len(line)
    label, suffix = get_label(punc)
    if length == 1:
        return label[:-1]
    suffix = suffix * (length - 1)
    return label + suffix[:-1]


def clean_line(line):
    if len(line) <= 1:
        return "", ""
    line = line.strip()
    line = line.replace('|', '')
    return line[:-1], line[-1]


# files = os.listdir("shi")
files = ["seg_test_simple.txt"]
# files = os.listdir("pred_data")
# prefix = "./data/pred"
cnt = 0
seg_lines = list()
save_path = "punc_test.txt"
for f in files:
    cnt += 1
    file_path = "./" + f
    print(file_path)
    with open(file_path, 'r', encoding="utf-8") as file:
        lines = file.read().split('\n')
        tot_lines = len(lines)
        used_line = 0

        while used_line < tot_lines:
            cur_line, punc = clean_line(lines[used_line])
            used_line += 1
            if len(cur_line) > 25:
                continue
            if cur_line == '':
                used_line += 1
                continue
            cur_lable = to_label(cur_line, punc)
            cur_len = len(cur_line)
            add_lines = max(random.randint(5, 20), random.randint(5, 20))
            # print(add_lines)
            # print(cur_line)
            # print_label(cur_lable)
            for i in range(add_lines):
                if used_line >= tot_lines:
                    break
                temp_line, punc = clean_line(lines[used_line])
                temp_len = len(temp_line)
                if temp_len > 25:
                    break
                if cur_len + temp_len > MAX_LEN:
                    break
                if temp_line == '':
                    used_line += 1
                    break
                cur_len += temp_len
                temp_label = to_label(temp_line, punc)
                cur_line += temp_line
                cur_lable = cur_lable + ' ' + temp_label
                used_line += 1
            # print(cur_line)
            # print_label(cur_lable)
            seg_lines.append(cur_line + '|' + cur_lable)
with open(save_path, 'w', encoding="utf-8") as file:
    for seg in seg_lines:
        file.write(seg + '\n')
