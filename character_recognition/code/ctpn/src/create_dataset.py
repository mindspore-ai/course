'''
Date: 2021-08-02 22:38:28
LastEditors: xgy
LastEditTime: 2021-08-15 16:19:23
FilePath: \code\ctpn\src\create_dataset.py
'''
# create train dataset of mindspore
from __future__ import division
import os
import numpy as np
from PIL import Image
from mindspore.mindrecord import FileWriter
from src.config import config

def create_label(train_img_dir, train_txt_dir, prefix=''):
    image_files = []
    image_anno_dict = {}
    img_basenames = []
    for file_name in os.listdir(train_img_dir):
        if 'gif' not in file_name:
            img_basenames.append(os.path.basename(file_name))
    img_names = []
    for item in img_basenames:
        temp1, _ = os.path.splitext(item)
        img_names.append((temp1, item))
    for img, img_basename in img_names:
        image_path = train_img_dir + '/' + img_basename
        annos = []
        file_name = prefix + img + ".txt"
        file_path = os.path.join(train_txt_dir, file_name)
        gt = open(file_path, 'r', encoding='gbk').read().splitlines()
        if not gt:
            continue
        for img_each_label in gt:
            spt = img_each_label.replace(',', '').split(' ')
            if ' ' not in img_each_label:
                spt = img_each_label.split(',')
            annos.append([spt[0], spt[1], spt[2], spt[3]] + [1])
        if annos:
            image_anno_dict[image_path] = np.array(annos)
            image_files.append(image_path)
    return image_files, image_anno_dict

def create_train_dataset(dataset_type):
    image_files = []
    image_anno_dict = {}
    if dataset_type == "pretraining":
        pretrain_image_files, pretrain_anno_dict = create_label(config.hwdb_pretrain_path[0],config.hwdb_pretrain_path[1])
        data_to_mindrecord_byte_image(pretrain_image_files, pretrain_anno_dict,config.pretrain_dataset_path,prefix="ctpn_pretrain.mindrecord", file_num=8)
    elif dataset_type == "finetune":
        finetune_image_files, finetune_anno_dict = create_label(config.hwdb_finetune_path[0],config.hwdb_finetune_path[1])
        data_to_mindrecord_byte_image(finetune_image_files, finetune_anno_dict,config.finetune_dataset_path,prefix="ctpn_finetune.mindrecord", file_num=8)
    elif dataset_type == "test":
        test_image_files, test_anno_dict = create_label(config.hwdb_test_path[0],config.hwdb_test_path[1])
        data_to_mindrecord_byte_image(test_image_files, test_anno_dict,config.test_dataset_path,prefix="ctpn_test.mindrecord", file_num=1)
    else:
        print("dataset_type should be pretraining, finetune, test")

def data_to_mindrecord_byte_image(image_files, image_anno_dict, dst_dir, prefix="cptn_mlt.mindrecord", file_num=1):
    """Create MindRecord file."""
    mindrecord_path = os.path.join(dst_dir, prefix)
    writer = FileWriter(mindrecord_path, file_num)

    ctpn_json = {
        "image": {"type": "bytes"},
        "annotation": {"type": "int32", "shape": [-1, 5]},
    }
    writer.add_schema(ctpn_json, "ctpn_json")
    for image_name in image_files:
        with open(image_name, 'rb') as f:
            img = f.read()
        annos = np.array(image_anno_dict[image_name], dtype=np.int32)
        print("img name is {}, anno is {}".format(image_name, annos))
        row = {"image": img, "annotation": annos}
        writer.write_raw_data([row])
    writer.commit()

if __name__ == '__main__':
    create_train_dataset("pretraining")
    create_train_dataset("finetune")
    create_train_dataset("test")