# 在Windows上运行LeNet_MNIST

## 实验介绍

LeNet5 + MINST被誉为深度学习领域的“Hello world”。本实验主要介绍使用MindSpore在Windows环境下MNIST数据集上开发和训练一个LeNet5模型，并验证模型精度。

## 实验目的

- 了解如何使用MindSpore进行简单卷积神经网络的开发。
- 了解如何使用MindSpore进行简单图片分类任务的训练。
- 了解如何使用MindSpore进行简单图片分类任务的验证。

## 预备知识

- 熟练使用Python，了解Shell及Linux操作系统基本知识。
- 具备一定的深度学习理论知识，如卷积神经网络、损失函数、优化器，训练策略等。
- 了解并熟悉MindSpore AI计算框架，MindSpore官网：[https://www.mindspore.cn](https://www.mindspore.cn/)

## 实验环境

- Windows-x64版本MindSpore 0.3.0；安装命令可见官网：

  [https://www.mindspore.cn/install](https://www.mindspore.cn/install)（MindSpore版本会定期更新，本指导也会定期刷新，与版本配套）。

## 实验准备

### 创建目录

创建一个experiment文件夹，用于存放实验所需的文件代码等。

### 数据集准备

MNIST是一个手写数字数据集，训练集包含60000张手写数字，测试集包含10000张手写数字，共10类。MNIST数据集的官网：[THE MNIST DATABASE](http://yann.lecun.com/exdb/mnist/)。

从MNIST官网下载如下4个文件到本地并解压：

```
train-images-idx3-ubyte.gz:  training set images (9912422 bytes)
train-labels-idx1-ubyte.gz:  training set labels (28881 bytes)
t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes)
t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)
```

### 脚本准备

从[课程gitee仓库](https://gitee.com/mindspore/course)上下载本实验相关脚本。

### 准备文件

将脚本和数据集放到到experiment文件夹中，组织为如下形式：

```
experiment
├── MNIST
│   ├── test
│   │   ├── t10k-images-idx3-ubyte
│   │   └── t10k-labels-idx1-ubyte
│   └── train
│       ├── train-images-idx3-ubyte
│       └── train-labels-idx1-ubyte
└── main.py
```

## 实验步骤

### 导入MindSpore模块和辅助模块

```python
import matplotlib.pyplot as plt

import mindspore as ms
import mindspore.context as context
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.transforms.vision.c_transforms as CV

from mindspore import nn
from mindspore.model_zoo.lenet import LeNet5
from mindspore.train import Model
from mindspore.train.callback import LossMonitor

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
```

### 数据处理

在使用数据集训练网络前，首先需要对数据进行预处理，如下：

```python
DATA_DIR_TRAIN = "MNIST/train"  # 训练集信息
DATA_DIR_TEST = "MNIST/test"  # 测试集信息

def create_dataset(training=True, num_epoch=1, batch_size=32, resize=(32, 32), rescale=1 / (255 * 0.3081), shift=-0.1307 / 0.3081, buffer_size=64):
    ds = ms.dataset.MnistDataset(DATA_DIR_TRAIN if training else DATA_DIR_TEST)

    ds = ds.map(input_columns="image", operations=[CV.Resize(resize), CV.Rescale(rescale, shift), CV.HWC2CHW()])
    ds = ds.map(input_columns="label", operations=C.TypeCast(ms.int32))
    ds = ds.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True).repeat(num_epoch)

    return ds
```

对其中几张图片进行可视化，可以看到图片中的手写数字，图片的大小为32x32。

```python
def show_dataset():
    ds = create_dataset(training=False)
    data = ds.create_dict_iterator().get_next()
    images = data['image']
    labels = data['label']

    for i in range(1, 5):
        plt.subplot(2, 2, i)
        plt.imshow(images[i][0])
        plt.title('Number: %s' % labels[i])
        plt.xticks([])
    plt.show()
```

![img](data:image/png;base64,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)

### 定义模型

MindSpore model_zoo中提供了多种常见的模型，可以直接使用。这里使用其中的LeNet5模型，模型结构如下图所示：

![img](https://www.mindspore.cn/tutorial/zh-CN/master/_images/LeNet_5.jpg)

图片来源于http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

### 训练

使用MNIST数据集对上述定义的LeNet5模型进行训练。训练策略如下表所示，可以调整训练策略并查看训练效果，要求验证精度大于95%。

| batch size | number of epochs | learning rate |    optimizer |
| ---------: | ---------------: | ------------: | -----------: |
|         32 |                3 |          0.01 | Momentum 0.9 |

```python
def test_train(lr=0.01, momentum=0.9, num_epoch=3, ckpt_name="a_lenet"):
    ds_train = create_dataset(num_epoch=num_epoch)
    ds_eval = create_dataset(training=False)

    net = LeNet5()
    loss = nn.loss.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction='mean')
    opt = nn.Momentum(net.trainable_params(), lr, momentum)

    loss_cb = LossMonitor(per_print_times=1)

    model = Model(net, loss, opt, metrics={'acc', 'loss'})
    model.train(num_epoch, ds_train, callbacks=[loss_cb], dataset_sink_mode=False)
    metrics = model.eval(ds_eval, dataset_sink_mode=False)
    print('Metrics:', metrics)
```

### 实验结果

1. 在训练日志中可以看到`epoch: 1 step: 1875, loss is 0.29772663`等字段，即训练过程的loss值；
2. 在训练日志中可以看到`Metrics: {'loss': 0.06830393138807267, 'acc': 0.9785657051282052}`字段，即训练完成后的验证精度。

```python
	...
>>> epoch: 1 step: 1875, loss is 0.29772663
    ...
>>> epoch: 2 step: 1875, loss is 0.049111396
    ...
>>> epoch: 3 step: 1875, loss is 0.08183163
>>> Metrics: {'loss': 0.06830393138807267, 'acc': 0.9785657051282052}
```

## 实验小结

本实验展示了如何使用MindSpore进行手写数字识别，以及开发和训练LeNet5模型。通过对LeNet5模型做几代的训练，然后使用训练后的LeNet5模型对手写数字进行识别，识别准确率大于95%。即LeNet5学习到了如何进行手写数字识别。