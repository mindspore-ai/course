import os
import argparse
#import random
from mindspore import context, Model, load_checkpoint, load_param_into_net, Tensor
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.communication.management import init
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.nn.optim.momentum import Momentum
from mindspore.context import ParallelMode
from resnet import resnet50
import cv2
from PIL import Image
import numpy as np
import mindspore as ms
import moxing as mox
import random

#from CreateDataset import create_dataset



# 定义一个参数接收器。用于读取运行时传入的参数
parser = argparse.ArgumentParser(description='face expression classification')
parser.add_argument('--run_distribute', type=bool, default=True, help='Run distribute.')
parser.add_argument('--device_num', type=int, default=24, help='Device num.')
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'])

args = parser.parse_args()

# 设置运行环境的参数
context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(device_id=device_id)

dataset_path = "./expression"    # 定义数据集所在路径
random.seed(1)


def get_img(data_path):
    # Getting image array from path:
    img0 = cv2.imread(data_path, 3)
    img1 = cv2.resize(img0, (128, 128))
    img2 = cv2.resize(img1, (224, 224))
    img = cv2.normalize(img2, 0, 255, cv2.NORM_MINMAX)
    return img

if __name__ == '__main__':
    
    data_obs_path='obs://sunce-demo/testdata/expression/'
    mox.file.copy_parallel(src_url=data_obs_path, dst_url='./expression')
    cpkt_obs_path='obs://sunce-demo/testdata/cpkt/'
    mox.file.copy_parallel(src_url=cpkt_obs_path, dst_url='./')
    
    #mox.file.copy_parallel(src_url='./cpkt', dst_url='./')
    #mox.file.copy_parallel(src_url='obs://sunce-demo/testdata/predict.csv', dst_url='./expression')
    
    # 自动并行运算
    if args.run_distribute:
        context.set_auto_parallel_context(device_num=args.device_num, parallel_mode=ParallelMode.DATA_PARALLEL)
        auto_parallel_context().set_all_reduce_fusion_split_indices([140])
        init()
    
    print("begin")
    
    net = resnet50(batch_size=32, num_classes=4)
    net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)
    #resnet = ResNet50()
    
    param_dict = load_checkpoint("train_resnet50-1_9792.ckpt")
    load_param_into_net(net, param_dict)
    model = Model(net, loss_fn=net_loss, optimizer=opt, metrics={'acc'})
    
    images=[]
    labels=[]
    
    csv_path=os.path.join(dataset_path,'predict.csv')
    
    print("begin predict")
    #with open(csv_path, 'r') as f:
    address = "./expression"
    count=0
    for root,dirs,files in os.walk(address): 
        #遍历 
    for file_name in dirs:
        #解析每一行csv文件内容
        pathq=os.path.join(address,file_name)
        txt_count=0
        for x, ys, txt_names in os.walk(os.path.join(address,file_name)):
        for txt_name in txt_names: 
            #cols = line.strip().split(",")  # 根据逗号，拆分csv文件中一行文本的元素
            image_path = os.path.join(pathq,txt_name)
            print(count)
            np0_image = get_img(image_path)
                
            np1_image = np.array(np0_image)
            #np1_image = np.transpose(np0_image,(2,0,1))
            #print("image: {}".format(np1_image))
                
            np2_image = np.transpose(np1_image,(2,0,1))
            #np2_image = np.array(np1_image)
            #print("shape: {}".format(np2_image.shape), ", dtype: {}".format(np2_image.dtype))
                
            np_image = np.array([np2_image], dtype=np.float32)
            #print("shape: {}".format(np_image.shape), ", dtype: {}".format(np_image.dtype))
                
            # 图像处理
            input_data = Tensor(np_image,ms.float32)
            pred = model.predict(input_data)
            print(pred)
            #print("label: {}".format( pred.argmax(axis=1) ) )
            a=pred[0][0]/5.39
            b=-pred[0][1]/0.61
            c=-pred[0][2]/0.75
            d=-pred[0][3]/3.24
                
            #a=a-random.random()*a
            #b=b-random.random()*b
            #c=c-random.random()*c
            #d=d-random.random()*d
                
            print(a,b,c,d)
            label_num=[a,b,c,d]
            #classes = {'0':a,'1':b,'2':c,'3':d}
            #print
            #classes = {a:'0',b:'1',:c,'3':d}
            label=label_num.index(max(label_num))
            #label=pred.index(max(pred))
            print("label: {}".format( label ) )
                
            labels.append(label)
            images.append(np_image)
            count+=1
            txt_count+=1
            if txt_count==10:
                break
    
    print("end predict")
    with open("result.csv", mode='w', newline='') as csv_p:
        fieldnames = ['label','shot']
        writer = csv.DictWriter(csv_p, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in zip(labellist, shotlist):
            writer.writerow({'shot':i[0], 'label':i[1]})
    
    out_obs_path='obs://sunce-demo/testdata/out/'
    mox.file.copy_parallel(src_url='result.csv', dst_url=out_obs_path)
    #mox.file.copy_parallel(src_url='./', dst_url='./out')
    
    