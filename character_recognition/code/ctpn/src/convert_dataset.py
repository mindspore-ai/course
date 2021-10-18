'''
Date: 2021-08-15 16:11:14
LastEditors: xgy
LastEditTime: 2021-10-15 11:49:05
FilePath: \code\ctpn\src\convert_dataset.py
'''

"""convert dataset label"""
import os
import argparse
from tqdm import tqdm

def init_args():
    parser = argparse.ArgumentParser('')
    parser.add_argument('-s', '--src_label_path', type=str, default='./',
                        help='Directory of train label')
    parser.add_argument('-t', '--target_label_path', type=str, default='test.xml',
                        help='Directory where save the  label after convert')
    return parser.parse_args()

def convert():
    args = init_args()
    anno_file = os.listdir(args.src_label_path)
    annos = {}
    # read
    for file in anno_file:
        gt = open(os.path.join(args.src_label_path, file), 'r', encoding='UTF-8-sig').read().splitlines()
        label_list = []
        label_name = os.path.basename(file)
        for each_label in gt:
            print(file)
            spt = each_label.split(',')
            print(spt)
            if "###" in spt[8]:
                continue
            else:
                x1 = min(int(spt[0]), int(spt[6]))
                y1 = min(int(spt[1]), int(spt[3]))
                x2 = max(int(spt[2]), int(spt[4]))
                y2 = max(int(spt[5]), int(spt[7]))
                label_list.append([x1, y1, x2, y2])
        annos[label_name] = label_list
    # write
    if not os.path.exists(args.target_label_path):
        os.makedirs(args.target_label_path)
    for label_file, pos in annos.items():
        tgt_anno_file = os.path.join(args.target_label_path, label_file)
        f = open(tgt_anno_file, 'w', encoding='gbk')
        for tgt_label in pos:
            str_pos = str(tgt_label[0]) + ',' + str(tgt_label[1]) + ',' + str(tgt_label[2]) + ',' + str(tgt_label[3])
            f.write(str_pos)
            f.write("\n")
        f.close()

if __name__ == "__main__":
    init_args()
    convert()
