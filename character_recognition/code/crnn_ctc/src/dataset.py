'''
Date: 2021-08-15 19:47:03
LastEditors: xgy
LastEditTime: 2021-10-09 21:40:22
FilePath: \code\crnn1\src\dataset.py
'''
# -*- coding: utf-8 -*-
# @Time    : 2021/9/16 10:43
# @Author  : Falcon
# @FileName: dataset.py.py

# create dataset of character recognition

import os
import numpy as np
from PIL import Image, ImageFile
from src.config import config1,label_dict
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as vc
ImageFile.LOAD_TRUNCATED_IMAGES = True

# class Dataset:
#     """
#     create train or evaluation dataset for crnn

#     Args:
#         img_root_dir(str): root path of images
#         max_text_length(int): max number of digits in images.
#         device_target(str): platform of training, support Ascend and GPU.
#     """

#     def __init__(self, img_dir, label_dir, config=config1):
#         if not os.path.exists(img_dir):
#             raise RuntimeError("the input image dir {} is invalid!".format(img_dir))
#         if not os.path.exists(label_dir):
#             raise RuntimeError("the label dir of input image {} is invalid!".format(label_dir))
#         self.img_dir = img_dir
#         self.label_dir = label_dir
#         img_files = os.listdir(img_dir)
#         label_files = os.listdir(label_dir)

#         self.img_names = {}
#         self.img_list = img_files
#         self.text_length = []

#         for img_file,label_file in zip(img_files,label_files):
#             with open(os.path.join(label_dir,label_file), 'r', encoding='gbk') as f:
#                 label = f.read()
#             self.img_names[img_file] = label
#             self.text_length.append(len(label))
#             # if len(label) > self.max_text_length:
#             #     self.max_text_length = len(label)
#         self.max_text_length = config.max_text_length
#         self.blank = config.blank
#         self.class_num = config.class_num
#         self.label_dict = label_dict
#         print(f'Finish loading {len(img_files)} images!')

#     def __len__(self):
#         return len(self.img_names)

#     def __getitem__(self, item):
#         img_name = self.img_list[item]
#         im = Image.open(os.path.join(self.img_dir, img_name))
#         im = im.convert("RGB")
#         r, g, b = im.split()
#         im = Image.merge("RGB", (b, g, r))
#         image = np.array(im)
#         label_str = self.img_names[img_name]
#         label = []
#         for c in label_str:
#             if c in label_dict:
#                 label.append(label_dict.index(c))
#         label.extend([int(self.blank)] * (self.max_text_length - len(label)))
#         label = np.array(label)
#         return image, label

# class Dataset:
#     """
#     create train or evaluation dataset for crnn

#     Args:
#         img_root_dir(str): root path of images
#         max_text_length(int): max number of digits in images.
#         device_target(str): platform of training, support Ascend and GPU.
#     """

#     def __init__(self, img_dir, label_dir, config=config1):
#         if not os.path.exists(img_dir):
#             raise RuntimeError("the input image dir {} is invalid!".format(img_dir))
#         if not os.path.exists(label_dir):
#             raise RuntimeError("the label dir of input image {} is invalid!".format(label_dir))
#         self.img_dir = img_dir
#         self.label_dir = label_dir
#         img_files = os.listdir(img_dir)
#         label_files = os.listdir(label_dir)

#         self.img_names = {}
#         self.img_list = img_files
#         self.text_length = []
#         # self.dict = {}

#         # self.label_dict = label_dict + '-'
#         # for i,char in enumerate(self.label_dict):
#         #     self.dict[char] = i+1

#         for img_file, label_file in zip(img_files, label_files):
#             with open(os.path.join(label_dir, label_file), 'r', encoding='gbk') as f:
#                 label = f.read()
#             self.img_names[img_file] = label
#             self.text_length.append(len(label))
#             # if len(label) > self.max_text_length:
#             #     self.max_text_length = len(label)
#         self.max_text_length = config.max_text_length
#         self.blank = config.blank
#         self.class_num = config.class_num
#         print(f'Finish loading {len(img_files)} images!')

#     def __len__(self):
#         return len(self.img_names)

#     def __getitem__(self, item):
#         img_name = self.img_list[item]
#         im = Image.open(os.path.join(self.img_dir, img_name))
#         im = im.convert("RGB")
#         r, g, b = im.split()
#         im = Image.merge("RGB", (b, g, r))
#         image = np.array(im)
#         label_str = self.img_names[img_name]
#         label = []
#         for c in label_str:
#             if c in label_dict:
#                 label.append(label_dict.index(c))
#         label.extend([int(config1.blank)] * (self.max_text_length - len(label)))
#         label = np.array(label)
#         return image, label

class Dataset:
    
    
    def __init__(self, img_dir, label_dir, device_target='Ascend'):
        """create train or evaluation dataset for crnn

        Args:
            img_dir ([type]): [description]
            label_dir ([type]): [description]
            device_target (str, optional): [description]. Defaults to 'Ascend'.

        Raises:
            RuntimeError: [description]
            RuntimeError: [description]
        """
        
        if not os.path.exists(img_dir):
            raise RuntimeError("the input image dir {} is invalid!".format(img_dir))
        if not os.path.exists(label_dir):
            raise RuntimeError("the label dir of input image {} is invalid!".format(label_dir))
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.target = device_target
        self.max_text_length = config1.max_text_length
        self.blank = config1.blank
        self.class_num = config1.class_num
        
        img_files = os.listdir(img_dir)
        label_files = os.listdir(label_dir)
        self.img_list = img_files
        self.img_names = {}
        label_length = []
        for img_file, label_file in zip(img_files, label_files):
            with open(os.path.join(label_dir, label_file), 'r', encoding='gbk') as f:
                label = f.read()
                label_length.append(len(label))
            self.img_names[img_file] = label
        
        print(f'Finish loading {len(img_files)} images!')
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, item):
        img_name = self.img_list[item]
        im = Image.open(os.path.join(self.img_dir, img_name))
        im = im.convert("RGB")
        r, g, b = im.split()
        im = Image.merge("RGB", (b, g, r))
        image = np.array(im)
        label_str = self.img_names[img_name]
        label = []
        for c in label_str:
            if c in label_dict:
                label.append(label_dict.index(c))
            else:
                label.append(int(config1.blank))
        label.extend([int(config1.blank)] * (self.max_text_length - len(label)))
        label = np.array(label)
        return image, label


def create_dataset(img_dir, label_dir, batch_size=1, num_shards=1, shard_id=0, is_training=True, config=config1):
    """
     create train or evaluation dataset for crnn

     Args:
        dataset_path(int): dataset path
        batch_size(int): batch size of generated dataset, default is 1
        num_shards(int): number of devices
        shard_id(int): rank id
        device_target(str): platform of training, support Ascend and GPU
     """
    dataset = Dataset(img_dir, label_dir)

    data_set = ds.GeneratorDataset(dataset, ["image", "label"], shuffle=True, num_shards=num_shards, shard_id=shard_id)
    image_trans = [
        vc.Resize((config.image_height, config.image_width)),
        vc.Normalize([127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
        vc.HWC2CHW()
    ]
    label_trans = [
        C.TypeCast(mstype.int32)
    ]
    data_set = data_set.map(operations=image_trans, input_columns=["image"], num_parallel_workers=8)
    data_set = data_set.map(operations=label_trans, input_columns=["label"], num_parallel_workers=8)

    data_set = data_set.batch(batch_size, drop_remainder=True)
    return data_set
#     return data_set, dataset.text_length


if __name__ == '__main__':
    img_dir = r'E:\program_lab\python\dataset\CASIA\textline\HWDB2.0Test_images'
    label_dir = r'E:\program_lab\python\dataset\CASIA\textline\HWDB2.0Test_label'
    Dataset(img_dir,label_dir)