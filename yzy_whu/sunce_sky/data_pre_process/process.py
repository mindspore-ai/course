#encoding:utf-8
import numpy as np
import scipy.misc as sm
import os
import csv
from PIL import Image

def getemotion(emotion_data):
    if emotion_data=='0':
        return 2
    if emotion_data=='3':
        return 3
    if emotion_data=='4':
        return 1
    else:
        return 0
        
#创建文件夹
def createDir(dir):
    if os.path.exists(dir) is False:
        os.makedirs(dir)
 
def saveImageFromFer2013():
 
 
    #读取csv文件
    with open('fer2013.csv', 'r') as f:
        next(f)
        reader = csv.reader(f)
        imageCount = 1
        fileCount = 1
        
        #遍历csv文件内容，并将图片数据按分类保存
        trainimagelist=[]
        trainlabellist=[]
        evalimagelist=[]
        evallabellist=[]
        
        for row in reader:
            #解析每一行csv文件内容
            #cols = line.strip().split(",")  # 根据逗号，拆分csv文件中一行文本的元素
            emotion_data = row[0]
            image_data = row[1]
            usage_data = row[2]
            #将图片数据转换成48*48
            data_array = list(map(float, image_data.split()))
            data_array = np.asarray(data_array)
            imagejpg = data_array.reshape(48, 48)
     
            emotion_key = getemotion(emotion_data)
            
            #图片要保存的文件夹
            imagePath = str(fileCount)
            createDir(imagePath)
     
            #图片文件名
            index = imageCount % 10
            if index==0 :
                fileCount+=1
            imageName = os.path.join(imagePath, '{}.jpg'.format(str(index)))
            imageCount += 1
            
            #sm.toimage(image).save(imageName)
            im = Image.fromarray(imagejpg).convert('L')
            im.save(imageName)
            
            if usage_data=='Training' :
                trainimagelist.append(imageName)
                trainlabellist.append(emotion_key)
            else:
                evalimagelist.append(imageName)
                evallabellist.append(emotion_key)
    f.close()
    
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
    saveImageFromFer2013()

