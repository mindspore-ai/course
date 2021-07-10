"""
文件名：DatasetGenerator.py
作者：孙策
描述：数据集生成器类
修改人：〈修改人〉
修改时间：YYYY-MM-DD
修改内容：〈修改内容〉
"""
import numpy as np
import os
from PIL import Image
from mindspore import Tensor
from mindspore import dtype as mstype
import mindspore as ms
#import pandas as pd
#import csv
import cv2
import pathlib

class DatasetGenerator:
    """
    该类定义了一个数据集生成器。
    """
    def __init__(self, dataset_path, csv_path):
        """
        该函数是创建该类的一个实例时，会执行的初始化函数。
        该函数中，实现了根据传入的txt文件对数据集的图片文件和标签值进行读取。
        Args:
            dataset_path: 数据集路径
            csv_path: 描述数据集的txt文件所在路径
        """
        #print("DatasetGeneratora")
        #print(f"create_dataset: dataset_path={dataset_path} , csv_path={csv_path}")
        
        images = []  # 创建空列表用于保存数据集的图片数据
        labels = []  # 创建空列表用于保存数据集的标签数据
        #shots = []  # 创建空列表用于保存数据集的地址数据
        
        # 读取csv文件中的每一行并进行处理
        """
        with open(csv_path, 'r') as f:
            next(f)	
            lines = f.readlines()
            for line in lines:
                cols = line.strip().split(",")  # 根据逗号，拆分csv文件中一行文本的元素
                image_path = os.path.join(dataset_path,cols[imageRow])  # 得到一张图片的路径
                np_image = np.array(Image.open(image_path)).astype(np.float32)  # 读取图片，并保存为np格式
                labels.append(int(cols[labelRow]))  # 将当前图片的标签数据添加到labels列表
                shots.append(cols[imageRow])  # 将图片路径添加到shots列表
                #print("Image shape: {}".format(np_image.shape), ", Label: {}".format(expression))
                #print("Image address: {}".format(subDirectory_filePath), ", Label: {}".format(expression))
        f.close()
        """
        """
        with open(csv_path, 'r') as f:
            next(f)	
            lines = f.readlines()
            for line in lines:
                cols = line.strip().split(",")  # 根据逗号，拆分csv文件中一行文本的元素
                image_path = os.path.join(dataset_path,cols[imageRow])  # 得到一张图片的路径
                #print("Image address: {}".format(image_path))
                image_open = Image.open(image_path)
                #print("Image address: {}".format(image_open))
                init_image = np.array(image_open).resize(144,144,3) # 读取图片
                #print("Image address: {}".format(init_image.shape))
                init2_image = np.array(init_image)
                #print("Image address: {}".format(init2_image.shape))
                np_image = init2_image.astype(np.float32)  # 保存为np格式
                images.append(np_image)  # 将当前图片数据添加到images列表
                #images.append(image_path)  # 测试
                labels.append(int(cols[labelRow]))  # 将当前图片的标签数据添加到labels列表
                shots.append(cols[imageRow])  # 将图片路径添加到shots列表
                #print("Image shape: {}".format(np_image.shape), ", Label: {}".format(cols[labelRow]))
                #print("Image address: {}".format(image_path), ", Label: {}".format(cols[labelRow]))
        f.close()
        """
        with open(csv_path, 'r') as f:
            next(f)	
            lines = f.readlines()
            imageCount=0
            for line in lines:
                cols = line.strip().split(",")  # 根据逗号，拆分csv文件中一行文本的元素
                image_path = os.path.join(dataset_path,cols[0])  # 得到一张图片的路径
                img_type = cols[0].strip().split(".")
                image_path_judge = pathlib.Path(image_path)
                
                if (image_path_judge.exists() and (img_type[1]!='tif' and img_type[1]!='TIF')):
                    # 读取图片，并保存为np格式
                    #np3_image = Image.open(image_path)
                    #print("Image0 type: {}".format(np3_image.dtype), ", Label: {}".format(cols[1]))
                    #np2_image = np.array(np3_image)
                    #print("Image0 type: {}".format(np2_image.dtype), ", Label: {}".format(cols[1]))
                    #np0_image = np.float32(np2_image)
                    #print("Image0 type: {}".format(np0_image.dtype), ", Label: {}".format(cols[1]))
                    #np1_image = np.array(cv2resize.resize(np0_image, (144,144), interpolation = cv2resize.INTER_AREA))
                    #print("Image1 shape: {}".format(np1_image), ", Label: {}".format(cols[1]))
                    #np1_image = cv2resize.resize(np0_image, (144,144), interpolation = cv2resize.INTER_AREA)
                    #print("Image2 shape: {}".format(np1_image), ", Label: {}".format(cols[1]))
                    #np_image = np.float32(np1_image)
                    #print("Image type: {}".format(np_image.dtype), ", Label: {}".format(cols[1]))
                    #np_image = np.float32(np1_image)
                    #np_image = CV.GaussinBlur(np1_image,(48,48),0)
                    #np_image = np.array(Image.open(image_path)).astype(np.float32)
                    
                    print("Image count: {}".format(imageCount))
                    

                    np0_image = Image.open(image_path)
                    print(np0_image)
                    
                    np1_image = np.array(np0_image,dtype=np.float32)
                    print("Image dtype: {}".format(np1_image.dtype), ", shape: {}".format(np1_image.shape),)
                    
                    np_image = cv2.resize(np1_image, (128,128), interpolation=cv2.INTER_LINEAR)
                    
                    #print(np0_image.astype(np.float32).shape)
                    #np_image = np0_image.astype(np.float32)
                    #np_image = np.array(Image.open(image_path)).astype(np.float32)
                    
                    images.append(np_image)  # 将当前图片数据添加到images列表
                    labels.append(int(cols[1]))  # 将当前图片的标签数据添加到labels列表
                    imageCount+=1
                    #shots.append(cols[imageRow])  # 将图片路径添加到shots列表
                    print("Image dtype: {}".format(np_image.dtype), ", shape: {}".format(np_image.shape),)
                    
                    if imageCount==235000:
                        break


                    #print("Image address: {}".format(np_image.shape), ", Label: {}".format(cols[1]))
        f.close()

        self.images = images  # 将images存为该对象的images属性
        self.labels = labels  # 将labels存为该对象的labels属性
        #self.shots = shots  # 将shots存为该对象的shots属性

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.labels)