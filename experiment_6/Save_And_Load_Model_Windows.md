# 在Windows上运行训练时模型的保存和加载

## 实验介绍

本实验主要介绍在Windows环境下使用MindSpore实现训练时模型的保存和加载。建议先阅读MindSpore官网教程中关于模型参数保存和加载的内容。

在模型训练过程中，可以添加检查点（CheckPoint）用于保存模型的参数，以便进行推理及中断后再训练使用。使用场景如下：

- 训练后推理场景
  - 模型训练完毕后保存模型的参数，用于推理或预测操作。
  - 训练过程中，通过实时验证精度，把精度最高的模型参数保存下来，用于预测操作。
- 再训练场景
  - 进行长时间训练任务时，保存训练过程中的CheckPoint文件，防止任务异常退出后从初始状态开始训练。
  - Fine-tuning（微调）场景，即训练一个模型并保存参数，基于该模型，面向第二个类似任务进行模型训练。

## 实验目的

- 了解如何使用MindSpore实现训练时模型的保存。
- 了解如何使用MindSpore加载保存的模型文件并继续训练。
- 了解如何MindSpore的Callback功能。

## 预备知识

- 熟练使用Python，了解Shell及Linux操作系统基本知识。
- 具备一定的深度学习理论知识，如卷积神经网络、损失函数、优化器，训练策略、Checkpoint等。
- 了解并熟悉MindSpore AI计算框架，MindSpore官网：https://www.mindspore.cn/

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
import numpy as np

import mindspore as ms
import mindspore.context as context
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.transforms.vision.c_transforms as CV

from mindspore import nn, Tensor
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
```

### 数据处理

在使用数据集训练网络前，首先需要对数据进行预处理，如下：

```python
DATA_DIR_TRAIN = "MNIST/train"  # 训练集信息
DATA_DIR_TEST = "MNIST/test"  # 测试集信息


def create_dataset(training=True, num_epoch=1, batch_size=32, resize=(32, 32),
                   rescale=1 / (255 * 0.3081), shift=-0.1307 / 0.3081, buffer_size=64):
    ds = ms.dataset.MnistDataset(DATA_DIR_TRAIN if training else DATA_DIR_TEST)

    # define map operations
    resize_op = CV.Resize(resize)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()

    # apply map operations on images
    ds = ds.map(input_columns="image", operations=[resize_op, rescale_op, hwc2chw_op])
    ds = ds.map(input_columns="label", operations=C.TypeCast(ms.int32))

    ds = ds.shuffle(buffer_size=buffer_size)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.repeat(num_epoch)

    return ds
```

### 定义模型

定义LeNet5模型，模型结构如下图所示：

![img](https://www.mindspore.cn/tutorial/zh-CN/master/_images/LeNet_5.jpg)

图片来源于http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

```python
class LeNet5(nn.Cell):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, pad_mode='valid')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(400, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 10)

    def construct(self, input_x):
        output = self.conv1(input_x)
        output = self.relu(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.pool(output)
        output = self.flatten(output)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output
```

### 保存模型Checkpoint

MindSpore提供了Callback功能，可用于训练/测试过程中执行特定的任务。常用的Callback如下：

- `ModelCheckpoint`：保存网络模型和参数，用于再训练或推理；
- `LossMonitor`：监控loss值，当loss值为Nan或Inf时停止训练；
- `SummaryStep`：把训练过程中的信息存储到文件中，用于后续查看或可视化展示。

`ModelCheckpoint`会生成模型（.meta）和Chekpoint（.ckpt）文件，如每个epoch结束时，都保存一次checkpoint。

```python
class CheckpointConfig:
    """
    The config for model checkpoint.

    Args:
        save_checkpoint_steps (int): Steps to save checkpoint. Default: 1.
        save_checkpoint_seconds (int): Seconds to save checkpoint. Default: 0.
            Can't be used with save_checkpoint_steps at the same time.
        keep_checkpoint_max (int): Maximum step to save checkpoint. Default: 5.
        keep_checkpoint_per_n_minutes (int): Keep one checkpoint every n minutes. Default: 0.
            Can't be used with keep_checkpoint_max at the same time.
        integrated_save (bool): Whether to intergrated save in automatic model parallel scene. Default: True.
            Integrated save function is only supported in automatic parallel scene, not supported in manual parallel.

    Raises:
        ValueError: If the input_param is None or 0.
    """

class ModelCheckpoint(Callback):
    """
    The checkpoint callback class.

    It is called to combine with train process and save the model and network parameters after traning.

    Args:
        prefix (str): Checkpoint files names prefix. Default: "CKP".
        directory (str): Lolder path into which checkpoint files will be saved. Default: None.
        config (CheckpointConfig): Checkpoint strategy config. Default: None.

    Raises:
        ValueError: If the prefix is invalid.
        TypeError: If the config is not CheckpointConfig type.
    """
```

MindSpore提供了多种Metric评估指标，如`accuracy`、`loss`、`precision`、`recall`、`F1`。定义一个metrics字典/元组，里面包含多种指标，传递给`Model`，然后调用`model.eval`接口来计算这些指标。`model.eval`会返回一个字典，包含各个指标及其对应的值。

```python
def test_train(lr=0.01, momentum=0.9, num_epoch=2, check_point_name="b_lenet"):
    ds_train = create_dataset(num_epoch=num_epoch)
    ds_eval = create_dataset(training=False)
    steps_per_epoch = ds_train.get_dataset_size()

    net = LeNet5()
    loss = nn.loss.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction='mean')
    opt = nn.Momentum(net.trainable_params(), lr, momentum)

    ckpt_cfg = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=5)
    ckpt_cb = ModelCheckpoint(prefix=check_point_name, config=ckpt_cfg)
    loss_cb = LossMonitor(steps_per_epoch)

    model = Model(net, loss, opt, metrics={'acc', 'loss'})
    model.train(num_epoch, ds_train, callbacks=[ckpt_cb, loss_cb], dataset_sink_mode=False)
    metrics = model.eval(ds_eval, dataset_sink_mode=False)
    print('Metrics:', metrics)
```

### 加载Checkpoint继续训练

```python
def load_checkpoint(ckpoint_file_name, net=None):
    """
    Loads checkpoint info from a specified file.

    Args:
        ckpoint_file_name (str): Checkpoint file name.
        net (Cell): Cell network. Default: None

    Returns:
        Dict, key is parameter name, value is a Parameter.

    Raises:
        ValueError: Checkpoint file is incorrect.
    """

def load_param_into_net(net, parameter_dict):
    """
    Loads parameters into network.

    Args:
        net (Cell): Cell network.
        parameter_dict (dict): Parameter dict.

    Raises:
        TypeError: Argument is not a Cell, or parameter_dict is not a Parameter dict.
    """
```

使用load_checkpoint接口加载数据时，需要把数据传入给原始网络，而不能传递给带有优化器和损失函数的训练网络。

```python
CKPT = 'b_lenet-2_1875.ckpt'

def resume_train(lr=0.001, momentum=0.9, num_epoch=2, ckpt_name="b_lenet"):
    ds_train = create_dataset(num_epoch=num_epoch)
    ds_eval = create_dataset(training=False)
    steps_per_epoch = ds_train.get_dataset_size()

    net = LeNet5()
    loss = nn.loss.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction='mean')
    opt = nn.Momentum(net.trainable_params(), lr, momentum)

    param_dict = load_checkpoint(CKPT)
    load_param_into_net(net, param_dict)
    load_param_into_net(opt, param_dict)

    ckpt_cfg = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=5)
    ckpt_cb = ModelCheckpoint(prefix=ckpt_name, config=ckpt_cfg)
    loss_cb = LossMonitor(steps_per_epoch)

    model = Model(net, loss, opt, metrics={'acc', 'loss'})
    model.train(num_epoch, ds_train, callbacks=[ckpt_cb, loss_cb], dataset_sink_mode=False)
    metrics = model.eval(ds_eval, dataset_sink_mode=False)
    print('Metrics:', metrics)
```

### 加载Checkpoint进行推理

使用matplotlib定义一个将推理结果可视化的辅助函数，如下：

```python
def plot_images(pred_fn, ds, net):
    for i in range(1, 5):
        pred, image, label = pred_fn(ds, net)
        plt.subplot(2, 2, i)
        plt.imshow(np.squeeze(image))
        color = 'blue' if pred == label else 'red'
        plt.title("prediction: {}, truth: {}".format(pred, label), color=color)
        plt.xticks([])
    plt.show()
```

使用训练后的LeNet5模型对手写数字进行识别，可以看到识别结果基本上是正确的。

```python
CKPT = 'b_lenet_1-2_1875.ckpt'

def infer(ds, model):
    data = ds.get_next()
    images = data['image']
    labels = data['label']
    output = model.predict(Tensor(data['image']))
    pred = np.argmax(output.asnumpy(), axis=1)
    return pred[0], images[0], labels[0]

def test_infer():
    ds = create_dataset(training=False, batch_size=1).create_dict_iterator()
    net = LeNet5()
    param_dict = load_checkpoint(CKPT, net)
    model = Model(net)
    plot_images(infer, ds, model)
```

![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAUoAAAD7CAYAAAAMyN1hAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy8li6FKAAAcv0lEQVR4nO3de5RU5Znv8e9D03TLJUIraCMgXgDJMlEZguRoEuJl1IkecpxlJo7jQpfaIdE1OmO8xMl9NPHkmMvMMpmIE4TxbtBRYuIkSjQRZVCWgSSKCqMIKHJROgEEpJvn/FGbXbuaqn5316Wrqvv3WatXP7v27d3VTz+1330rc3dERKSwAdVugIhIrVOhFBEJUKEUEQlQoRQRCVChFBEJUKEUEQmoyUJpxjwzbozij5nxSpHL+bEZXylv6+qDGePNcDMGVrstkqXcLl01crsmC2WSO0+7Myk0nRkXmbG4y7yz3fnnyrUuXvexZvzSjC1m9OjCVDNmmLG+DG1YY8ZppS4nsbxPmvGkGX8yY025litZyu3Uy6l6ble8UPaTPZo9wAPAJZVYeJXewx3AXOCaKqy7Lii3S1c3ue3uPf4BXwP+JfCXwLeC3wHeHI2bAb4e/Drwt8HvjF4/G3w5eDv4s+AfTizvBPAXwLeB3w9+H/iNyeUlph0L/hD4ZvB3wG8Fnwy+C7wTfDt4ezTtvH3LiYYvA18N/i74QvDRiXEOPht8VbRNPwS3Hr4vR4N7D6YfAr4TfG/U7u3go8G/Dr4A/C7wP4Nfmmdb4vcF/M5oGTujZVwLPj7aplnga8G3gP9TEX/r08DXFJMn9fij3FZu5/spZY/yAuAM4ChgIvDlxLhDgRbgcKDNjClkKvjngIOA24CFZjSZMQh4GLgzmuenwF/nW6EZDcCjwBvAeOAw4D53VgKzgSXuDHVneJ55TwG+DXwGaI2WcV+Xyc4GPgIcF013RjTvODPazRiX9s1Jw50dwFnAW1G7h7rzVjR6JrAAGA7cHVjOhcBa4JxoGd9JjD4ZmAScCnzVjMnRNp1sRns5t6cPUW6XqK/ldimF8lZ31rnzLnATcH5i3F7ga+7sdmcncBlwmztL3el0Zz6wG5ge/TQCP3BnjzsLgOcLrHMaMBq4xp0d7uxyzz12040LgLnuvODObuBLwEfNGJ+Y5mZ32t1ZCzwJHA/gzlp3hkev95Yl7jzszt7oPSzWN9zZ6c4KYAWZfxTcWZzvn04A5Xal1V1ul1Io1yXiN8j8kffZ7M6uxPDhwNXRJ1d7VO3HRvOMBt50zzlQ/EaBdY4F3nCno4j2jk4u153twDtkPrn3eTsRvwcMLWI95bIuPEkqtbRN9UK5XVl1l9ulFMqxiXgcxLvVwH5nx9YBN0WfXPt+BrtzL7ABOMwM67K8fNYB4wocAA6dkXuLTFIDYMYQMl2lNwPzVVqhdnd9fQcwODF8aMrlSM8pt8ujz+R2KYXycjPGmNEC3ADc3820twOzzTjRDDNjiBmfMmMYsAToAP7ejIFmnEumG5LPc2SS7+ZoGc1mnBSN2wiMiY4L5XMPcLEZx5vRBHwLWOpe+qUv0TY1Q2bdUbuaEuPnmTGvwOwbgYPMODCwmuXAX5nRYsahwFV5lnNkURuQhxkDom1qhMz2dfPe9jXK7YhyO6OUQnkP8CvgtejnxkITurOMzLGcW4GtwGrgomjc+8C50fBW4G+AhwospxM4BziazAHe9dH0AL8GXgTeNmNLnnkXAV8BHiSTkEcBn02zodEB7+3dHPA+HNgZrZ8oTl5IPBZ4psA2vQzcC7wWdd1G55uOzAmBFcAaMu9713/ebwNfjpbxxcAm7bvYeXs3k3w82o5fkNkL2hmttz9QbmcptwHLnCbvmegizUvdeaLHM/cz0SfVCuDD7uypdnuke8rt9PpTbveHC2arKtqrmFztdoiUW3/K7Zq/hVFEpNqK6nqLiPQnJe1RmtmZZvaKma02s+vL1SiRalNuS1LRe5Rm1gC8CpxO5gzd88D57v5S+Zon0vuU29JVKSdzpgGr3f01ADO7j8w9nAWTaZA1eTNDSlillMs2tm5x95HVbkeNUm7XqV3s4H3fbeEpe6aUQnkYubcirQdO7DqRmbUBbQDNDOZEO7WEVUq5POELCt1KJ8rturXUF1VkuaUco8xXtffrx7v7HHef6u5TG7MX9IvUMuW25CilUK4n957YMeTeEytSr5TbkqOUQvk8MMHMjjCzQWRumVpYnmaJVJVyW3IUfYzS3TvM7Argl0ADMNfdXwzMJlLzlNvSVUm3MLr7L8jcWC7Spyi3JUm3MIqIBKhQiogEqFCKiASoUIqIBKhQiogE6MG9KQwYnP3eo7VXHp8zbm+KGzJGLs9+sd4BDz9XtnaJSO/QHqWISIAKpYhIgLreKdiw7PeqL/jcLTnjJg8a3HXy/Rzx2KVxPPHh8rVLJK2BR46P402faE01z8hHV8dx5+bN5W5SXdEepYhIgAqliEiACqWISICOUYr0A8njks/f9G+p5pnS9Pk4PvTB7Ov98Xil9ihFRAJUKEVEAtT1FpG8Xvhqtov+kd3ZbnjLHep6i4hIFyqUIiIBKpQiIgEqlCIiASqUIiIBOust0g8kH3CRvJA8eWZbCgvuUZrZXDPbZGZ/TLzWYmaPm9mq6PeIyjZTpPyU25JWmq73PODMLq9dDyxy9wnAomhYpN7MQ7ktKQS73u7+WzMb3+XlmcCMKJ4PPAVcV8Z2iVRcf8rt5P3ZB60c0+P5P37F0jhe3PnROB7+H0tKa1idKPZkziHuvgEg+j2q0IRm1mZmy8xs2R52F7k6kV6j3Jb9VPyst7vPcfep7j61kRTfxCVSJ5Tb/UexZ703mlmru28ws1ZgUzkbJVJFyu08vtv6QhxPPnJ6HA+vRmOqoNg9yoXArCieBTxSnuaIVJ1yW/aT5vKge4ElwCQzW29mlwA3A6eb2Srg9GhYpK4otyWtNGe9zy8w6tQyt0WkVym3JS3dwigiEqBCKSISoHu9C2gYOTKON5x3dBwPG7C3Gs0RkSrSHqWISIAKpYhIgLreBez5YPZ+2N/d8KPEmKG93xgRqSrtUYqIBKhQiogEqOudYE3ZBxu8/4HGkpa1qXNHdrm7GkpalohUl/YoRUQCVChFRALU9U5oP++EOL73plsSY3p+pvuku78Yx5Nuir+SBV2uLlJ/tEcpIhKgQikiEqBCKSISoGOUCZ2NFsdHNJZ2B87AXdll7d22raRliUh1aY9SRCRAhVJEJECFUkQkQIVSRCRAhVJEJECFUkQkIM33eo81syfNbKWZvWhmV0avt5jZ42a2Kvo9ovLNFSkf5baklWaPsgO42t0nA9OBy83sg8D1wCJ3nwAsioZF6olyW1IJXnDu7huADVG8zcxWAocBM4EZ0WTzgaeA6yrSygrqnDEljredtb2KLZHe1tdzW8qnR8cozWw8cAKwFDgkSrR9CTeqwDxtZrbMzJbtYXdprRWpEOW2dCd1oTSzocCDwFXu/ue087n7HHef6u5TG2kKzyDSy5TbEpLqXm8zaySTSHe7+0PRyxvNrNXdN5hZK7CpUo2spDdnNMfxyyfPrWJLpBr6cm5L+aQ5623AT4CV7v69xKiFwKwongU8Uv7miVSOclvSSrNHeRJwIfAHM1sevXYDcDPwgJldAqwFzqtME0UqRrktqaQ5670YsAKjTy1vc3pHw4Qj43hX654qtkSqqS/mtlSG7swREQlQoRQRCeiXTzhfeX1LHL9+1u1lW+76juwF6wN0WZ3UKOvwOH51z444PmrgATnTNVj+/aiO5uz8A4YNi+O+/CR/7VGKiASoUIqIBPTLrnelnPOda+N43Nzlcby3Go0RKWDA0j/G8VVnXRzHP3jsjpzpJjYOyTv/MxfcEsfTh/1jHE+4fGm5mlhztEcpIhKgQikiEqCud4mmfPPzcdz64Ko47nzvvWo0RyTIOzqyA++0x2GnF7r2PteohmyX3Js7y9auWqY9ShGRABVKEZEAdb1TSF5InjyzDV2625s391qbRKT3aI9SRCRAhVJEJECFUkQkoF8eoxz7s+znw+R1XwhOn3zARfKOG9BlQCL9gfYoRUQCVChFRAL6Zdf7gIefi+NxD/dsXj3gQvoSf29nHH/q5/+QOy7FXTcHL24se5tqkfYoRUQCVChFRAKCXW8zawZ+CzRF0y9w96+ZWQtwPzAeWAN8xt23Vq6pIuWl3M79+oa+/DzJUqXZo9wNnOLuxwHHA2ea2XTgemCRu08AFkXDIvVEuS2pBAulZ+y72bkx+nFgJjA/en0+8OmKtFCkQpTbklaqY5Rm1mBmy4FNwOPuvhQ4xN03AES/R1WumSKVodyWNFIVSnfvdPfjgTHANDM7Nu0KzKzNzJaZ2bI96DtcpbYotyWNHp31dvd24CngTGCjmbUCRL83FZhnjrtPdfepjTSV2FyRylBuS3eChdLMRprZ8Cg+ADgNeBlYCMyKJpsFPFKpRopUgnJb0kpzZ04rMN/MGsgU1gfc/VEzWwI8YGaXAGuB8yrYTpFKUG5LKsFC6e6/B07I8/o7wKmVaJRIb1BuS1rm7r23MrPNwBu9tkLpzuHuPrLajegrlNs1oyJ53auFUkSkHulebxGRABVKEZGAmiyUZswz48Yo/pgZrxS5nB+b8ZXytq4+mDHeDDfrn88crVXK7dJVI7drslAmufO0O5NC05lxkRmLu8w7251/rlzr4nUfa8YvzdhiRo8O+poxw4z1ZWjDGjNOK3U5ieV90ownzfiTGWvKtVzJqofcjtZ/pBmPmrEtyvHvpJyvJnM7WuYUM35rxnYzNppxZXfTV7xQ9pM9mj3AA8AllVh4ld7DHcBc4JoqrLsu9IfcNmMQ8Djwa+BQMrd63lXG5ff6e2jGwcB/AbcBBwFHA7/qdiZ37/EP+BrwL4G/BL4V/A7w5mjcDPD14NeBvw1+Z/T62eDLwdvBnwX/cGJ5J4C/AL4N/H7w+8BvTC4vMe1Y8IfAN4O/A34r+GTwXeCd4NvB26Np5+1bTjR8Gfhq8HfBF4KPToxz8Nngq6Jt+iG49fB9OTp6Jk3a6YeA7wTfG7V7O/ho8K+DLwC/C/zP4Jfm2Zb4fQG/M1rGzmgZ14KPj7ZpFvha8C3g/1TE3/o08DXF5Ek9/ii393s/2sCfLuJ9rNncBv/Wvr9d2p9S9igvAM4AjgImAl9OjDsUaAEOB9rMmEJm7+RzZCr4bcBCM5qiT6yHgTujeX4K/HW+FZrRADxK5nq18cBhwH3urARmA0vcGerO8DzzngJ8G/gMmTsy3gDu6zLZ2cBHgOOi6c6I5h1nRrsZ49K+OWm4swM4C3gravdQd96KRs8EFgDDgbsDy7mQzB0k50TLSHaNTgYmkbmA+qtmTI626WQz2su5PX2IcjtrOrDGjMeibvdTZnyowLSxGs/t6cC7ZjxrxiYzfhb63y6lUN7qzjp33gVuAs5PjNsLfM2d3e7sBC4DbnNnqTud7swn89DU6dFPI/ADd/a4swB4vsA6pwGjgWvc2eHOLvfcYzfduACY684L7uwGvgR81IzxiWludqfdnbXAk2Qe5oo7a90ZHr3eW5a487A7e6P3sFjfcGenOyuAFWT+UXBncb5/OgGU20ljgM8C/xq17+fAI9GHQLGqndtjyNzDfyUwDngduLe7FZVSKNcl4jfIvIn7bHZnV2L4cODq6JOrPar2Y6N5RgNvuuecBCl0h8NY4A13Oopo7+jkct3ZDrxD5pN7n7cT8XvA0CLWUy7rwpOkUkvbVC+U21k7gcXuPObO+8AtZPacJxfRzn2qnds7gf905/nob/kN4H+ZcWChGUoplGMT8TiId6uB/c78rgNuij659v0MdudeYANwmBnWZXn5rAPGFTgAHDrb/BaZpAbAjCFk/uBvBuartELt7vr6DmBwYvjQlMuRnlNuZ/0+xfoLqdXc7rpN+2LLMy1QWqG83IwxZrQAN5D5MqZCbgdmm3GiGWbGEDM+ZcYwYAnQAfy9GQPNOJdMNySf58gk383RMprNOCkatxEY002X4B7gYjOON6MJ+Baw1L30S1+ibWqGzLqjdjUlxs8zY16B2TcCB3X3aRZZDvyVGS1mHApclWc5Rxa1AXmYMSDapkbIbF+J3a16otzOuguYbsZp0XHUq4AtwEqoz9wG7gD+T/R+NQJfIbPXXPC4ZimF8h4yp9Rfi35uLDShO8vIHMu5FdgKrAYuisa9D5wbDW8F/gZ4qMByOoFzyJzOXwusj6aHzOULLwJvm7Elz7yLyLwhD5JJyKPIHHsJig54b+/mgO/hZHbnX4yGd0LOhcRjgWcKbNPLZI6PvBZ13Ubnm47MCYEVZL4V8Ffs/8/7beDL0TK+GNikfRc7b+9mko9H2/ELMntBOwldQtF3KLezy34F+Dvgx9E2zAT+d7RtUIe57c6vyXwA/pzMQ5mPBv6222VmTpf3jGUuQL7UnSd6PHM/E+0FrAA+7M6eardHuqfcTq8/5Xafv2C22qJP3lIOfIvUpP6U2zV/C6OISLUVVSjdGe/OE2Z2ppm9YmarzUxfEi91T7kt+RT94N7oe0ZeBU4nc+D5eeB8d3+pfM0T6X3KbemqlGOU04DV7v4agJndR+aMWMFkGmRN3syQElYp5bKNrVtcXwVRiHK7Tu1iB+/77oLXQxarlEJ5GLlX2K8HTuxuhmaGcKLpO5tqwRO+QN/vUphyu04t9UUVWW4phTJf1d6vH29mbUAbQHPOxfciNUu5LTlKOeu9ntxbvcaQe6sXAO4+x92nuvvUxuzNKiK1TLktOUoplM8DE8zsCDMbROZOgIXlaZZIVSm3JUfRXW937zCzK4BfAg3AXHd/MTCbSM1TbktXJd2Z4+6/IHMvsEifotyWJN2ZIyISoEIpIhKgQikiEqBCKSISoEIpIhKg51GK9FGdM6bE8ZszmvNOM2B3Nh73L8tzxu19772KtKseaY9SRCRAhVJEJEBd7xK9d272oTK7Dsz/uTPi1ex3vNszy/NOI9ITafJu21nZ79d6+eS5eadZ35Gd5pxt1+aMa71/VRx3bt5cVDv7Cu1RiogEqFCKiASoUIqIBOgYZQo2MPs27T3x2JxxF970szhuO3C/RxYCcMRjl8bxxLxfFS/SM2d8/Tdx/OWDXy56OWMGDo3j393wo5xxp//h4jge8BsdoxQRkW6oUIqIBKjrncKAg1ri+Pt353ZPJg/Sd6VIBVn263saDj44jpsGvFaN1vRb2qMUEQlQoRQRCVDXW6SGJbvbbc8uieOzBm9NTNXYiy3qn7RHKSISoEIpIhKgrrdIjbGp2ZsaPjkvf3e7ycLd7eOeOz+OR30///MouzPwhdVxvLfHc/ctwT1KM5trZpvM7I+J11rM7HEzWxX9HlHZZoqUn3Jb0krT9Z4HnNnlteuBRe4+AVgUDYvUm3kotyWFYNfb3X9rZuO7vDwTmBHF84GngOvK2K6qG3DsMXG891+3xfHYgTqs21fUam53Dsl2q69p+Z/EmHB3+5jFF8bxuH9piGN75nc9bkd/724nFftff4i7bwCIfo8qX5NEqkq5Lfup+MkcM2sD2gCa0e1+0ncot/uPYgvlRjNrdfcNZtYKbCo0obvPAeYAfMBavMj19bqOgw6I48ePuS8xpudnD6Wu1HVu20vDsvEzz1axJX1LsV3vhcCsKJ4FPFKe5ohUnXJb9pPm8qB7gSXAJDNbb2aXADcDp5vZKuD0aFikrii3Ja00Z73PLzDq1DK3paY0vvWnOD7q19knPS/7xA9zphvRkP/Y1MVrPxbHBy/Wvbi1qFZye8Bxk3OGX/3bnh0RO33lOXE8cnlHWdqUVueMKXH85oziD0uN/8+tOcN7V6wselmVoGtdREQCVChFRAJ0r3cBnauyT5CedHX2Urq3l+ZON6KBvP77vz4Ux+Pu0NlHKWzLlOE5w6+f8289mr/9rjFx3PLwkm6mLL9kd3tl24+6mbJ7R4xqyxk+5rbs4Yha6IZrj1JEJECFUkQkQF3vAgYMy164u2Pa+DhuNt0BK/1bw4Qj43hX656yLPP1T8/JGZ686QtxPG5FWVZREu1RiogEqFCKiASo611Ax5Sj4/g3tyW7BUN7vzEiVdYw/MA4fuVr2fj1U26vyPo6mrO3zicPg+3dti3f5BWnPUoRkQAVShGRAHW9RSTozXmj43jZXySfd1CZ53A+c8EtcTx92D/G8YTLl+abvOK0RykiEqBCKSISoEIpIhKgY5RldMzt2bsJjvrJ2jju3ScEipTHu49OjOOffujf43hEw5AeLef1Pdvj+KIrsscbL/jOo3HcduBbOfOMSqzDmzt7tL5K0B6liEiACqWISIC63mU0bE32boKOdeur2BKpJ6N+syFnOHkI5+XLws94/PgV2UtmFnd+NI6H/0dpz6acOe73cTyxsWfd7Qe2Z+/e+b/f/XwcN8zeEsczBq9KzNGz5fc27VGKiASoUIqIBKjrneAnHR/Hm/5hV6p5jvjZZXF8zAvtcaynVkpaHa+tyRk+6t+z10lMaMp2W5N3qyTPCn+39YU4vvgL2def+sTUOB720qA4bv1uuq8meXDOKXHcNDv73MlrWv4nOO+ru1qzbb0z+0DJtS3Z/7E1k7JfgTGxsTzPtayUNN/rPdbMnjSzlWb2opldGb3eYmaPm9mq6PeIyjdXpHyU25JWmq53B3C1u08GpgOXm9kHgeuBRe4+AVgUDYvUE+W2pBLserv7BmBDFG8zs5XAYcBMYEY02XzgKeC6irSyl2ydeEAcr5g2L9U8R9+T7SbVwrfFSXq1mtvJKyYmfP/9OH7nsxbHowp8++cd457ODiTi/3fiUXE8d9gZZWhl96YNznbP7772lG6mrA89OpljZuOBE4ClwCFRou1LuFGF5xSpbcpt6U7qQmlmQ4EHgavc/c89mK/NzJaZ2bI97C6mjSIVpdyWkFRnvc2skUwi3e3uD0UvbzSzVnffYGatwKZ887r7HGAOwAesxfNNI1ItNZ/bu7MF+LPLL4njn56Qvfc6zcXgyTPV17SFL2Iv1V8Ozp7FXtkL66u0NGe9DfgJsNLdv5cYtRCYFcWzgEfK3zyRylFuS1pp9ihPAi4E/mBmy6PXbgBuBh4ws0uAtcB5lWmiSMUotyWVNGe9FwNWYPSp5W2OSO+ph9zubP9THB/66Wx8waMXx3Hynuy/HPaHOJ7W1Fjh1pVP8t5wyL1gfdCG6m+HbmEUEQlQoRQRCdC93iJ1qOXsV+P4aZrjeO4Pr4jjn3/q+73aplJ875uzc4YPvOu/43g8pT0urhy0RykiEqBCKSISoK63SB8y6fqX4vjqb366ii3pmeHtv8sZrrU7U7RHKSISoEIpIhKgrrdIH7J327bsQDKWkmiPUkQkQIVSRCRAhVJEJECFUkQkQIVSRCRAhVJEJECFUkQkQIVSRCRAhVJEJEB35iSMeHVnHB/x2KWp5pn81rtx3Fn2FolILdAepYhIgAqliEiAut4J9szyOJ74TLp51N0W6fuCe5Rm1mxmz5nZCjN70cy+Eb3eYmaPm9mq6PeIyjdXpHyU25JWmq73buAUdz8OOB4408ymA9cDi9x9ArAoGhapJ8ptSSVYKD1jezTYGP04MBOYH70+H6if586LoNyW9FKdzDGzBjNbDmwCHnf3pcAh7r4BIPo9qnLNFKkM5bakkapQununux8PjAGmmdmxaVdgZm1mtszMlu1hd7HtFKkI5bak0aPLg9y9HXgKOBPYaGatANHvTQXmmePuU919aiNNJTZXpDKU29KdNGe9R5rZ8Cg+ADgNeBlYCMyKJpsFPFKpRopUgnJb0kpzHWUrMN/MGsgU1gfc/VEzWwI8YGaXAGuB8yrYTpFKUG5LKubee181bmabgTd6bYXSncPdfWS1G9FXKLdrRkXyulcLpYhIPdK93iIiASqUIiIBKpQiIgEqlCIiASqUIiIBKpQiIgEqlCIiASqUIiIBKpQiIgH/HyKwOLKinO/gAAAAAElFTkSuQmCC)

### 实验结果

1. 在训练日志中可以看到两阶段的loss值和验证精度打印，第一阶段为初始训练，第二阶段为加载Checkpoint继续训练；
2. 在训练目录里可以看到`b_lenet-graph.meta`、`b_lenet-2_1875.ckpt`等文件，即训练过程保存的Checkpoint。

```python
>>> epoch: 1 step: 1875, loss is 2.2984316
>>> epoch: 2 step: 1875, loss is 0.06388051
>>> Metrics: {'loss': 0.11160586341821517, 'acc': 0.9637419871794872}
        
>>> epoch: 1 step: 1875, loss is 0.008898618
>>> epoch: 2 step: 1875, loss is 0.05747048
>>> Metrics: {'loss': 0.07453688951276351, 'acc': 0.9767628205128205}
```

## 实验小结

本实验展示了MindSpore的Checkpoint、断点继续训练等高级特性：

1. 使用MindSpore的ModelCheckpoint接口每个epoch保存一次Checkpoint，训练2个epoch并终止。
2. 使用MindSpore的load_checkpoint和load_param_into_net接口加载上一步保存的Checkpoint继续训练2个epoch。
3. 观察训练过程中Loss的变化情况，加载Checkpoint继续训练后loss进一步下降。