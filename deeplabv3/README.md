# Deeplabv3—语义分割

## 实验介绍
本实验主要介绍使用MindSpore深度学习框架在PASCAL VOC2012数据集上训练deeplabv3网络模型。本实验使用了MindSpore开源仓库model_zoo中的[deeplabv3](https://gitee.com/mindspore/mindspore/tree/r0.5/model_zoo/deeplabv3)模型案例。

## deeplabv3简要介绍
deeplabv1和deeplabv2，即带孔卷积(atrous convolution), 能够明确地调整filters的感受野，并决定DNN计算得到特征的分辨率。
deeplabv3中提出 Atrous Spatial Pyramid Pooling(ASPP)模块, 挖掘不同尺度的卷积特征，以及编码了全局内容信息的图像层特征，提升分割效果。
详细介绍参考论文：http://arxiv.org/abs/1706.05587 。

## 实验目的
* 了解如何使用MindSpore加载常用的PASCAL VOC2012数据集。
* 了解MindSpore的model_zoo模块，以及如何使用model_zoo中的模型。
* 了解deeplabv3这类语义分割模型的基本结构和编程方法。

## 预备知识
* 熟练使用Python，了解Shell及Linux操作系统基本知识。
* 具备一定的深度学习理论知识，如Encoder、Decoder、损失函数、优化器，训练策略、Checkpoint等。
* 了解华为云的基本使用方法，包括[OBS（对象存储）](https://www.huaweicloud.com/product/obs.html)、[ModelArts（AI开发平台）](https://www.huaweicloud.com/product/modelarts.html)、[训练作业](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0238.html)等功能。华为云官网：https://www.huaweicloud.com。
* 了解并熟悉MindSpore AI计算框架，MindSpore官网：https://www.mindspore.cn/。

## 实验环境
* MindSpore 1.0.0（MindSpore版本会定期更新，本指导也会定期刷新，与版本配套）。
* 华为云ModelArts（控制台左上角选择“华北-北京四”）：ModelArts是华为云提供的面向开发者的一站式AI开发平台，集成了昇腾AI处理器资源池，用户可以在该平台下体验MindSpore。。

## 实验准备

### 数据集准备

[Pascal VOC2012数据集](https://blog.csdn.net/haoji007/article/details/80361587)主要是针对视觉任务中监督学习提供标签数据，它有二十个类别。主要有四个大类别，分别是人、常见动物、交通车辆、室内家具用品。这里只说与图像分割（segmentation）有关的信息,本用例使用已去除分割标注的颜色，仅保留了分割任务的数据集。VOC2012[官网地址](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)，[官方下载地址](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)。

本实验指导的数据集可通过如下方式获取：
* 方式一，参考（推荐）[lenet5（手写数字识别）](../lenet5)或[checkpoint（模型的保存和加载）](../checkpoint)实验，拷贝他人共享的OBS桶中的数据集。
    ```
    import moxing
    moxing.file.copy_parallel(src_url="s3://share-course/dataset/voc2012/", dst_url='voc2012/')
    ```
* 方式二，从官网下载数据集

另外，本实验采用fine-tune的训练方式，为了节省训练时间，我们提前准备好了预训练的[checkpoint文件](https://share-course.obs.cn-north-4.myhuaweicloud.com/checkpoint/deeplabv3/deeplab_v3_s8-800_82.ckpt)，方便直接获取使用。

### 脚本准备

从[课程gitee仓库](https://gitee.com/mindspore/course)上下载本实验相关脚本。

### 上传文件

点击新建的OBS桶名，再打开“对象”标签页，通过“上传对象”、“新建文件夹”等功能，将脚本和数据集上传到OBS桶中，组织为如下形式：

```
deeplabv3
├── src # 包括数据集处理、网络定义等
│   └── *.py
└── main.ipynb # 执行脚本，包括训练和推理过程
```

## 实验步骤

### 代码梳理

代码文件说明：

- main.ipynb：代码入口文件；
- dataset.py：数据处理文件；
- loss：loss定义文件；
- deeplab_v3: deeplabv3网络定义文件；
- learning_rates.py: 学习率定义文件

实验流程：

1. 修改main.ipynb训练参数并运行，运行训练cell得到模型文件。
2. 修改main.ipynb测试1（test 1 cell）参数并运行，运行测试1单元得到mean iou结果。
3. 修改main.ipynb测试2（test 2 cell）参数并运行，运行测试2单元得到可视化结果。

### 数据处理（dataset.py）

数据处理流程如下所示：

1. 将语义标签转换为灰度图（dataset.py中SegDataset.get_gray_dataset）
2. 将图片和标签灰度图转换为mindrecord格式数据集（dataset.py中SegDataset.get_mindrecord_dataset）
3. 读取mindrecord数据集并预处理。（dataset.py中SegDataset.get_dataset。其中preprocess_为数据预处理。）

具体过程如下所示，见（main.ipynb）

```python
# dataset
dataset = data_generator.SegDataset(image_mean=args.image_mean,
                                    image_std=args.image_std,
                                    data_file=args.data_file,
                                    batch_size=args.batch_size,
                                    crop_size=args.crop_size,
                                    max_scale=args.max_scale,
                                    min_scale=args.min_scale,
                                    ignore_label=args.ignore_label,
                                    num_classes=args.num_classes,
                                    num_readers=2,
                                    num_parallel_calls=4,
                                    shard_id=args.rank,
                                    shard_num=args.group_size)
dataset.get_gray_dataset()
dataset.get_mindrecord_dataset(is_training=True)
dataset = dataset.get_dataset(repeat=1)
```


### 训练输入文件导入

```python
import moxing as mox
data_path = './VOC2012'
if not os.path.exists(data_path):
    mox.file.copy_parallel(src_url="s3://share-course/dataset/voc2012_raw/", dst_url=data_path)
cfg.data_file = data_path

ckpt_path = 'deeplab_s8.ckpt'
if not os.path.exists(ckpt_path):
    mox.file.copy_parallel(src_url="s3://share-course/checkpoint/deeplabv3/deeplab_v3_s8-800_82.ckpt", dst_url=ckpt_path)
cfg.ckpt_file = ckpt_path
```

### 训练输入文件导入

```python
import moxing as mox
data_path = './VOC2012'
if not os.path.exists(data_path):
    mox.file.copy_parallel(src_url="s3://share-course/dataset/voc2012_raw/", dst_url=data_path)
cfg.data_file = data_path
from src.data import dataset as data_generator
# dataset
dataset = data_generator.SegDataset(image_mean=cfg.image_mean,
                                    image_std=cfg.image_std,
                                    data_file=cfg.data_file)
dataset.get_gray_dataset()
cfg.data_lst = os.path.join(cfg.data_file,'ImageSets/Segmentation/val.txt')
cfg.voc_img_dir = os.path.join(cfg.data_file,'JPEGImages')
cfg.voc_anno_gray_dir = os.path.join(cfg.data_file,'SegmentationClassGray')

ckpt_path = './model'
if not os.path.exists(ckpt_path):
    mox.file.copy_parallel(src_url="s3://{user_obs}/model", dst_url=ckpt_path)   # if model had saved.
cfg.ckpt_file = os.path.join(ckpt_path,'deeplab_v3_s8-3_91.ckpt')  
print('loading checkpoing:',cfg.ckpt_file)
```

### 训练参数设定：

```python
cfg = edict({
    "batch_size": 16,
    "crop_size": 513,
    "image_mean": [103.53, 116.28, 123.675],
    "image_std": [57.375, 57.120, 58.395],
    "min_scale": 0.5,
    "max_scale": 2.0,
    "ignore_label": 255,
    "num_classes": 21,
    "train_epochs" : 3,

    "lr_type": 'cos',
    "base_lr": 0.0,

    "lr_decay_step": 3*91,
    "lr_decay_rate" :0.1,

    "loss_scale": 2048,      

    "model": 'deeplab_v3_s8',
    'rank': 0,
    'group_size':1,
    'keep_checkpoint_max':1,
    'train_dir': 'model',

    'is_distributed':False,
    'freeze_bn':True
})
```

### 测试参数设定：

```python
cfg = edict({
    "batch_size": 1,
    "crop_size": 513,
    "image_mean": [103.53, 116.28, 123.675],
    "image_std": [57.375, 57.120, 58.395],
    "scales": [1.0],           # [0.5,0.75,1.0,1.25,1.75]
    'flip': True,

    'ignore_label': 255,
    'num_classes':21,

    'model': 'deeplab_v3_s8',
    'freeze_bn': True,
    
    'if_png':False,
    'num_png':5
})
```

## 实验结果

### 训练日志结果

```
converting voc color png to gray png ...
converting done
creating mindrecord dataset...
number of samples: 1464
number of samples written: 1000
number of samples written: 1464
Create Mindrecord Done
epoch: 1 step: 91, loss is 0.004917805
Epoch time: 183256.301, per step time: 2013.806
epoch: 2 step: 91, loss is 0.00791893
Epoch time: 47812.316, per step time: 525.410
epoch: 3 step: 91, loss is 0.0061199386
Epoch time: 47803.087, per step time: 525.309
```
### 测试iou结果

```
the gray file is already exists！
loading checkpoing: ./model/deeplab_v3_s8-3_91.ckpt
processed 100 images
processed 200 images
processed 300 images
processed 400 images
processed 500 images
processed 600 images
processed 700 images
processed 800 images
processed 900 images
processed 1000 images
processed 1100 images
processed 1200 images
processed 1300 images
processed 1400 images
mean IoU 0.7709573541968988
```

### 测试图片输出结果

取其中一张图片结果如下所示：

![png](images/example.png)

```
prediction num: [ 0  2 15]
prediction color: ['background', 'bicycle', 'person']
prediction class: ['aliceblue', 'red', 'tan']
groundtruth num: [ 0  2 15]
groundtruth color: ['background', 'bicycle', 'person']
groundtruth class: ['aliceblue', 'red', 'tan']
```

**注解：** 以上三张图片，第左边为原始图片，中间为预测语义分割图，最右边为真实语义分割标签图。

## 结论

本实验主要介绍使用MindSpore实现deeplabv3网络，实现语义分割。分析原理和结果可得：

- deeplabv3网络对语义分割任务有效。
- deeplabv3网络对语义分割中细节效果较差。但是大概轮廓较好。