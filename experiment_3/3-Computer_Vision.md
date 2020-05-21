<h1 style="text-align:center">计算机视觉应用</h1>


## 实验介绍

本实验主要介绍使用MindSpore在CIFAR10数据集上训练ResNet50。本实验建议使用MindSpore model_zoo中提供的ResNet50。

## 实验目的

- 了解如何使用MindSpore加载常用的CIFAR-10图片分类数据集。
- 了解MindSpore的model_zoo模块，以及如何使用model_zoo中的模型。
- 了解ResNet50这类大模型的基本结构和编程方法。

## 预备知识

- 熟练使用Python，了解Shell及Linux操作系统基本知识。
- 具备一定的深度学习理论知识，如卷积神经网络、损失函数、优化器，训练策略、Checkpoint等。
- 了解华为云的基本使用方法，包括[OBS（对象存储）](https://www.huaweicloud.com/product/obs.html)、[ModelArts（AI开发平台）](https://www.huaweicloud.com/product/modelarts.html)、[训练作业](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0046.html)等功能。华为云官网：https://www.huaweicloud.com
- 了解并熟悉MindSpore AI计算框架，MindSpore官网：https://www.mindspore.cn/

## 实验环境

- MindSpore 0.2.0（MindSpore版本会定期更新，本指导也会定期刷新，与版本配套）；
- 华为云ModelArts：ModelArts是华为云提供的面向开发者的一站式AI开发平台，集成了昇腾AI处理器资源池，用户可以在该平台下体验MindSpore。ModelArts官网：https://www.huaweicloud.com/product/modelarts.html

## 实验准备

### 创建OBS桶

本实验需要使用华为云OBS存储脚本和数据集，可以参考[快速通过OBS控制台上传下载文件](https://support.huaweicloud.com/qs-obs/obs_qs_0001.html)了解使用OBS创建桶、上传文件、下载文件的使用方法。

> **提示：** 华为云新用户使用OBS时通常需要创建和配置“访问密钥”，可以在使用OBS时根据提示完成创建和配置。也可以参考[获取访问密钥并完成ModelArts全局配置](https://support.huaweicloud.com/prepare-modelarts/modelarts_08_0002.html)获取并配置访问密钥。

创建OBS桶的参考配置如下：

- 区域：华北-北京四
- 数据冗余存储策略：单AZ存储
- 桶名称：如ms-course
- 存储类别：标准存储
- 桶策略：公共读
- 归档数据直读：关闭
- 企业项目、标签等配置：免

### 数据集准备

CIFAR-10是一个图片分类数据集，包含60000张32x32的彩色物体图片，训练集50000张，测试集10000张，共10类，每类6000张。CIFAR-10数据集的官网：[THE MNIST DATABASE](http://www.cs.toronto.edu/~kriz/cifar.html)。

从CIFAR-10官网下载“CIFAR-10 binary version (suitable for C programs)”到本地并解压。

### 脚本准备

从[MindSpore tutorial仓库](https://gitee.com/mindspore/docs/tree/r0.2/tutorials/tutorial_code/sample_for_cloud/)里下载相关脚本。

### 上传文件

将脚本和数据集上传到OBS桶中，组织为如下形式：

```
experiment_3
├── 脚本等文件
└── cifar10
    ├── batches.meta.txt
    ├── eval
    │   └── test_batch.bin
    └── train
        ├── data_batch_1.bin
        ├── data_batch_2.bin
        ├── data_batch_3.bin
        ├── data_batch_4.bin
        └── data_batch_5.bin
```

## 实验步骤

参考MindSpore官网[计算机视觉应用](https://www.mindspore.cn/tutorial/zh-CN/0.1.0-alpha/advanced_use/computer_vision_application.html)教程，使用MindSpore在CIFAR10数据集上训练ResNet50，并进行验证。建议：

- 使用单卡训练即可；
- 理解并熟悉教程中涉及的源码；
- 使用MindSpore model_zoo中提供的ResNet50。

### 代码梳理

- resnet50_train.py：主脚本，包含性能测试`PerformanceCallback`、动态学习率`get_lr`、执行函数`resnet50_train`等函数；
- dataset.py：数据处理脚本。

`PerformanceCallback`继承MindSpore Callback类，并统计每个训练step的时延：

```python
class PerformanceCallback(Callback):
    """
    Training performance callback.

    Args:
        batch_size (int): Batch number for one step.
    """
    def __init__(self, batch_size):
        super(PerformanceCallback, self).__init__()
        self.batch_size = batch_size
        self.last_step = 0
        self.epoch_begin_time = 0

    def step_begin(self, run_context):
        self.epoch_begin_time = time.time()

    def step_end(self, run_context):
        params = run_context.original_args()
        cost_time = time.time() - self.epoch_begin_time
        train_steps = params.cur_step_num -self.last_step
        print(f'epoch {params.cur_epoch_num} cost time = {cost_time}, train step num: {train_steps}, '
              f'one step time: {1000*cost_time/train_steps} ms, '
              f'train samples per second of cluster: {device_num*train_steps*self.batch_size/cost_time:.1f}\n')
        self.last_step = run_context.original_args().cur_step_num
```

`get_lr`生成学习率数组，其中每个元素对应每个step的学习率，这里学习率下降采用二次曲线的形式：

```python
def get_lr(global_step,
           total_epochs,
           steps_per_epoch,
           lr_init=0.01,
           lr_max=0.1,
           warmup_epochs=5):
    """
    Generate learning rate array.

    Args:
        global_step (int): Initial step of training.
        total_epochs (int): Total epoch of training.
        steps_per_epoch (float): Steps of one epoch.
        lr_init (float): Initial learning rate. Default: 0.01.
        lr_max (float): Maximum learning rate. Default: 0.1.
        warmup_epochs (int): The number of warming up epochs. Default: 5.

    Returns:
        np.array, learning rate array.
    """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    if warmup_steps != 0:
        inc_each_step = (float(lr_max) - float(lr_init)) / float(warmup_steps)
    else:
        inc_each_step = 0
    for i in range(int(total_steps)):
        if i < warmup_steps:
            lr = float(lr_init) + inc_each_step * float(i)
        else:
            base = ( 1.0 - (float(i) - float(warmup_steps)) / (float(total_steps) - float(warmup_steps)) )
            lr = float(lr_max) * base * base
            if lr < 0.0:
                lr = 0.0
        lr_each_step.append(lr)

    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]

    return learning_rate
```

MindSpore支持直接读取cifar10数据集：

```python
if device_num == 1 or not do_train:
    ds = de.Cifar10Dataset(dataset_path, num_parallel_workers=8, shuffle=do_shuffle)
else:
    ds = de.Cifar10Dataset(dataset_path, num_parallel_workers=8, shuffle=do_shuffle,
                           num_shards=device_num, shard_id=device_id)
```

使用数据增强，如随机裁剪、随机水平反转：

```python
# define map operations
random_crop_op = C.RandomCrop((32, 32), (4, 4, 4, 4))
random_horizontal_flip_op = C.RandomHorizontalFlip(device_id / (device_id + 1))
```

导入并使用model_zoo里的resnet50模型：

```python
from mindspore.model_zoo.resnet import resnet50
# create model
net = resnet50(class_num = class_num)
```

`model_zoo.resnet`中resnet50定义如下：

```python
def resnet50(class_num=10):
    return ResNet(ResidualBlock,
                  [3, 4, 6, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 2],
                  class_num)
```

ResNet类定义如下：

```python
class ResNet(nn.Cell):
    """
    ResNet architecture.

    Args:
        block (Cell): Block for network.
        layer_nums (list): Numbers of block in different layers.
        in_channels (list): Input channel in each layer.
        out_channels (list): Output channel in each layer.
        strides (list):  Stride size in each layer.
        num_classes (int): The number of classes that the training images are belonging to.
    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResNet(ResidualBlock,
        >>>        [3, 4, 6, 3],
        >>>        [64, 256, 512, 1024],
        >>>        [256, 512, 1024, 2048],
        >>>        [1, 2, 2, 2],
        >>>        10)
    """
```

ResNet的不同版本均由5个阶段（stage）组成，其中ResNet50结构为Convx1 -> ResidualBlockx3 -> ResidualBlockx4 -> ResidualBlockx6 -> ResidualBlockx5 -> Pooling+FC。

`ResidualBlock`为残差模块，相比传统卷积多了一个short-cut支路，用于将浅层的信息直接传递到深层，使得网络可以很深，而不会出现训练时梯度消失/爆炸的问题：

```python
class ResidualBlock(nn.Cell):
    expansion = 4

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1):
        super(ResidualBlock, self).__init__()

        channel = out_channel // self.expansion
        self.conv1 = _conv1x1(in_channel, channel, stride=1)
        self.bn1 = _bn(channel)

        self.conv2 = _conv3x3(channel, channel, stride=stride)
        self.bn2 = _bn(channel)

        self.conv3 = _conv1x1(channel, out_channel, stride=1)
        self.bn3 = _bn_last(out_channel)

        self.relu = nn.ReLU()

        # 如果in
        self.down_sample = False
        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None
        if self.down_sample:
            self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel, stride),
                                                        _bn(out_channel)])
        self.add = P.TensorAdd()

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample:
            identity = self.down_sample_layer(identity)

        # output为残差支路，identity为short-cut支路
        out = self.add(out, identity)
        out = self.relu(out)

        return out
```

创建训练作业时，运行参数会通过脚本传参的方式输入给脚本代码，脚本必须解析传参才能在代码中使用相应参数。如data_url和train_url，分别对应数据存储路径(OBS路径)和训练输出路径(OBS路径)。脚本对传参进行解析后赋值到`args`变量里，在后续代码里可以使用。

```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
parser.add_argument('--train_url', required=True, default=None, help='Location of training outputs.')
parser.add_argument('--num_epochs', type=int, default=1, help='Number of training epochs.')
args, unknown = parser.parse_known_args()
```

MindSpore暂时没有提供直接访问OBS数据的接口，需要通过MoXing提供的API与OBS交互。将OBS中存储的数据拷贝至执行容器：

```python
import moxing as mox
mox.file.copy_parallel(src_url=args.data_url, dst_url='cifar10/')
```

如需将训练输出（如模型Checkpoint）从执行容器拷贝至OBS，请参考：

```python
import moxing as mox
mox.file.copy_parallel(src_url='output', dst_url='s3://OBS/PATH')
```

### 创建训练作业

可以参考[使用常用框架训练模型](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0238.html)来创建并启动训练作业。

创建训练作业的参考配置：

- 算法来源：常用框架->Ascend-Powered-Engine->MindSpore
- 代码目录：选择上述新建的OBS桶中的experiment_3目录
- 启动文件：选择上述新建的OBS桶中的experiment_3目录下的`resnet50_train.py`
- 数据来源：数据存储位置->选择上述新建的OBS桶中的experiment_3文件夹下的cifar10目录
- 训练输出位置：选择上述新建的OBS桶中的experiment_3目录并在其中创建output目录
- 作业日志路径：同训练输出位置
- 规格：Ascend:1*Ascend 910
- 其他均为默认

启动并查看训练过程：

1. 点击提交以开始训练；
2. 在训练作业列表里可以看到刚创建的训练作业，在训练作业页面可以看到版本管理；
3. 点击运行中的训练作业，在展开的窗口中可以查看作业配置信息，以及训练过程中的日志，日志会不断刷新，等训练作业完成后也可以下载日志到本地进行查看；
4. 在训练日志中可以看到`epoch 90 cost time = 27.963477849960327, train step num: 1562, one step time: 17.90235457743939 ms, train samples per second of cluster: 1787.5`等字段，即训练过程的性能数据；
5. 在训练日志中可以看到`epoch: 90 step: 1562, loss is 0.00250402`等字段，即训练过程的loss数据；
6. 在训练日志里可以看到`Evaluation result: {'acc': 0.9182692307692307}.`字段，即训练完成后的验证精度。

## 实验结论

本实验主要介绍使用MindSpore在CIFAR10数据集上训练ResNet50，了解了以下知识点：

- 性能测试
- 动态学习率
- model_zoo：resnet50
- cifar10数据集、数据增强
