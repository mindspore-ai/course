# -*- coding: utf-8 -*-
# @Time    : 2021/8/1 21:57
# @Author  : Falcon
# @FileName: generate_dataset.py
import struct
import os
import numpy as np
import cv2 as cv
import os
from glob import glob
import re
from tqdm import tqdm

def read_from_dgrl(dgrl):
    if not os.path.exists(dgrl):
        print('DGRL not exis!')
        return

    dir_name, base_name = os.path.split(dgrl)
    label_dir = dir_name + '_labels'
    image_dir = dir_name + '_images'
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    with open(dgrl, 'rb') as f:
        # 读取表头尺寸
        header_size = np.fromfile(f, dtype='uint8', count=4)
        header_size = sum([j << (i * 8) for i, j in enumerate(header_size)])
        # print(header_size)

        # 读取表头剩下内容，提取 code_length
        header = np.fromfile(f, dtype='uint8', count=header_size - 4)
        code_length = sum([j << (i * 8) for i, j in enumerate(header[-4:-2])])
        # print(code_length)

        # 读取图像尺寸信息，提取图像中行数量
        image_record = np.fromfile(f, dtype='uint8', count=12)
        height = sum([j << (i * 8) for i, j in enumerate(image_record[:4])])
        width = sum([j << (i * 8) for i, j in enumerate(image_record[4:8])])
        line_num = sum([j << (i * 8) for i, j in enumerate(image_record[8:])])
        print('图像尺寸:')
        print(height, width, line_num)

        # 读取每一行的信息
        for k in range(line_num):
            print(k + 1)

            # 读取该行的字符数量
            char_num = np.fromfile(f, dtype='uint8', count=4)
            char_num = sum([j << (i * 8) for i, j in enumerate(char_num)])
            print('字符数量:', char_num)

            # 读取该行的标注信息
            label = np.fromfile(f, dtype='uint8', count=code_length * char_num)
            label = [label[i] << (8 * (i % code_length)) for i in range(code_length * char_num)]
            label = [sum(label[i * code_length:(i + 1) * code_length]) for i in range(char_num)]
            label = [struct.pack('I', i).decode('gbk', 'ignore')[0] for i in label]
            print('合并前：', label)
            label = ''.join(label)
            label = ''.join(label.split(b'\x00'.decode()))  # 去掉不可见字符 \x00，这一步不加的话后面保存的内容会出现看不见的问题
            print('合并后：', label)

            # 读取该行的位置和尺寸
            pos_size = np.fromfile(f, dtype='uint8', count=16)
            y = sum([j << (i * 8) for i, j in enumerate(pos_size[:4])])
            x = sum([j << (i * 8) for i, j in enumerate(pos_size[4:8])])
            h = sum([j << (i * 8) for i, j in enumerate(pos_size[8:12])])
            w = sum([j << (i * 8) for i, j in enumerate(pos_size[12:])])
            print(x, y, w, h)

            # 读取该行的图片
            bitmap = np.fromfile(f, dtype='uint8', count=h * w)
            bitmap = np.array(bitmap).reshape(h, w)

            # 保存信息
            label_file = os.path.join(label_dir, base_name.replace('.dgrl', '_' + str(k) + '.txt'))
            with open(label_file, 'w') as f1:
                f1.write(label)
            bitmap_file = os.path.join(image_dir, base_name.replace('.dgrl', '_' + str(k) + '.jpg'))
            cv.imwrite(bitmap_file, bitmap)

def get_char_nums(segments):
    nums = []
    chars = []
    for seg in segments:
        label_head = seg.split('.')[0]
        label_name = label_head + '.txt'
        with open(os.path.join(label_root, label_name), 'r', encoding='gbk') as f:
        # with open(os.path.join(label_root, label_name), 'r') as f:
            lines = f.readlines()
            if len(lines) == 0:
                print(seg)
                continue
            nums.append(len(lines[0]))
            chars.append(lines[0])
    return nums, chars


def addZeros(s_):
    head, tail = s_.split('_')
    num = ''.join(re.findall(r'\d', tail))
    head_num = '0' * (4 - len(num)) + num
    return head + '_' + head_num + '.jpg'


def strsort(alist):
    alist.sort(key=lambda i: addZeros(i))
    return alist


def pad(img, headpad, padding):
    assert padding >= 0
    if padding > 0:
        logi_matrix = np.where(img > 255 * 0.95, np.ones_like(img), np.zeros_like(img))
        ids = np.where(np.sum(logi_matrix, 0) == img.shape[0])
        if ids[0].tolist() != []:
            pad_array = np.tile(img[:, ids[0].tolist()[-1], :], (1, padding)).reshape((img.shape[0], -1, 3))
        else:
            pad_array = np.tile(np.ones_like(img[:, 0, :]) * 255, (1, padding)).reshape((img.shape[0], -1, 3))
        if headpad:
            return np.hstack((pad_array, img))
        else:
            return np.hstack((img, pad_array))
    else:
        return img


def pad_peripheral(img, pad_size):
    assert isinstance(pad_size, tuple)
    w, h = pad_size
    result = cv.copyMakeBorder(img, h, h, w, w, cv.BORDER_CONSTANT, value=[255, 255, 255])
    return result


if __name__ == '__main__':

    # convert from dgrl file to jpg file(textline images)
    paths = ['./dataset/pretrain'] #dgrl path
    for path in paths:
        read_from_dgrl(path)
    os.system('mv ./dataset/pretrain_images ./dataset/recognition/pretrain_images')
    os.system('mv ./dataset/pretrain_labels ./dataset/recognition/pretrain_labels')

    # concate images
    label_roots = ['./dataset/recognition/pretrain_labels']
    label_dets = ['./dataset/detection/pretrain_labels']
    pages_roots = ['./dataset/recognition/pretrain_images']
    pages_dets = ['./dataset/detection/pretrain_images']
    
    for label_root, label_det, pages_root, pages_det in zip(label_roots, label_dets, pages_roots, pages_dets):
        os.makedirs(label_det, exist_ok=True)
        os.makedirs(pages_det, exist_ok=True)
        pages_for_set = os.listdir(pages_root)
        pages_set = set([pfs.split('_')[0] for pfs in pages_for_set])
        for ds in tqdm(pages_set):
            boxes = []
            pages = []
            seg_sorted = strsort([d for d in pages_for_set if ds in d])
            widths = [cv.imread(os.path.join(pages_root, d)).shape[1] for d in seg_sorted]
            heights = [cv.imread(os.path.join(pages_root, d)).shape[0] for d in seg_sorted]
            max_width = max(widths)
            seg_nums, chars = get_char_nums(seg_sorted)
            pad_size = (500, 1000)
            w, h = pad_size
            label_name = ds + '.txt'
            with open(os.path.join(label_det, label_name), 'w') as f:
                for i, pg in enumerate(seg_sorted):
                    headpad = True if i == 0 else True if seg_nums[i] - seg_nums[i - 1] > 5 else False
                    pg_read = cv.imread(os.path.join(pages_root, pg))
                    padding = max_width - pg_read.shape[1]
                    page_new = pad(pg_read, headpad, padding)
                    pages.append(page_new)
                    if headpad:
                        x1 = str(w + padding)
                        x2 = str(w + max_width)
                        y1 = str(h + sum(heights[:i + 1]) - heights[i])
                        y2 = str(h + sum(heights[:i + 1]))
                        box = np.array([int(x1), int(y1), int(x2), int(y1), int(x2), int(y2), int(x1), int(y2)])
                    else:
                        x1 = str(w)
                        x2 = str(w + max_width - padding)
                        y1 = str(h + sum(heights[:i + 1]) - heights[i])
                        y2 = str(h + sum(heights[:i + 1]))
                        box = np.array([int(x1), int(y1), int(x2), int(y1), int(x2), int(y2), int(x1), int(y2)])
                    boxes.append(box.reshape((4, 2)))
                    char = chars[i]
                    f.writelines(
                        x1 + ',' + y1 + ',' + x2 + ',' + y1 + ',' + x2 + ',' + y2 + ',' + x1 + ',' + y2 + ',' + char + '\n')
            pages_array = np.vstack(pages)
            pages_array = pad_peripheral(pages_array, pad_size)
            pages_name = ds + '.jpg'
            # cv.polylines(pages_array, [box.astype('int32') for box in boxes], True, (0, 0, 255))
            cv.imwrite(os.path.join(pages_det, pages_name), pages_array)

