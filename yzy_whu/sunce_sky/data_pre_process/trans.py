#encoding:utf-8
import numpy as np
import scipy.misc as sm
import os
import csv
from PIL import Image

def getemotion(emotion_data):
    if emotion_data=='1':
        return 3
    if emotion_data=='2':
        return 1
    if emotion_data=='4':
        return 2
    if emotion_data=='6':
        return 2
    else:
        return 0


def translabel():
    imageCount=0
    
    #读取csv文件
    with open('training1.csv', 'r') as t:
        next(t)
        reader = csv.reader(t)
        
        #遍历csv文件内容
        trainimagelist=[]
        trainlabellist=[]        
        for row in reader:
            #解析每一行csv文件内容
            #cols = line.strip().split(",")  # 根据逗号，拆分csv文件中一行文本的元素
            emotion_data = row[6]
            image_data = row[0]
            
            emotion_key=getemotion(emotion_data)

            trainimagelist.append(image_data)
            trainlabellist.append(emotion_key)
            
            imageCount+=1

    t.close()
    
    with open('validation1.csv', 'r') as v:
        next(v)
        reader = csv.reader(v)
        
        #遍历csv文件内容
        evalimagelist=[]
        evallabellist=[]        
        for row in reader:
            #解析每一行csv文件内容
            #cols = line.strip().split(",")  # 根据逗号，拆分csv文件中一行文本的元素
            emotion_data = row[6]
            image_data = row[0]
            
            emotion_key=getemotion(emotion_data)

            evalimagelist.append(image_data)
            evallabellist.append(emotion_key)
            
            imageCount+=1

    v.close()
    
    with open('training.csv', mode='w', newline='') as csv_t:
        fieldnames = ['shot', 'label']
        writer = csv.DictWriter(csv_t, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in zip(trainimagelist, trainlabellist):
            writer.writerow({'shot':i[0], 'label':i[1]})
    csv_t.close()
    
    with open('validation.csv', mode='w', newline='') as csv_v:
        fieldnames = ['shot', 'label']
        writer = csv.DictWriter(csv_v, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in zip(evalimagelist, evallabellist):
            writer.writerow({'shot':i[0], 'label':i[1]})
    csv_v.close()
    
    print('总共有' + str(imageCount) + '张图片')
 
 
if __name__ == '__main__':
    translabel()

