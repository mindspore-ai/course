# Unet网络

## 实验介绍

## 实验目的

- 了解如何使用MindSpore进行简单卷积神经网络的开发。
- 了解如何使用MindSpore进行简单图片分割任务的训练。
- 了解如何使用MindSpore进行简单图片分割任务的验证。

## 预备知识

- 熟练使用Python，了解Shell及Linux操作系统基本知识。
- 具备一定的深度学习理论知识，如卷积神经网络、图像分割、损失函数、训练策略等。
- 了解华为云的基本使用方法，包括[OBS（对象存储）](https://www.huaweicloud.com/product/obs.html)、[ModelArts（AI开发平台）](https://www.huaweicloud.com/product/modelarts.html)、[Notebook（开发工具）](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0032.html)、[训练作业](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0238.html)等服务。华为云官网：https://www.huaweicloud.com
- 了解并熟悉MindSpore AI计算框架，MindSpore官网：https://www.mindspore.cn

## 实验环境

- MindSpore 1.0.0（MindSpore版本会定期更新，本指导也会定期刷新，与版本配套）；
- 华为云ModelArts（控制台左上角选择“华北-北京四”）：ModelArts是华为云提供的面向开发者的一站式AI开发平台，集成了昇腾AI处理器资源池，用户可以在该平台下体验MindSpore；

## 实验准备

### 数据集准备

ISBI 挑战是训练和测试数据是果蝇第一龄幼虫腹侧腹侧神经索（VNC）的连续切片透射电子显微镜（ssTEM）数据集的两个部分，每组30个切片的集合。微立方体的尺寸约为2 x 2 x 1.5微米，分辨率为4x4x50 nm /像素。

数据集大小：22.5M

- 训练集：15M，30张图像（训练数据包含2个多页TIF文件，每个文件包含30张2D图像。train-volume.tif和train-labels.tif分别包含数据和标签。）
- 测试集：7.5M，30张图像（测试数据包含1个多页TIF文件，每个文件包含30张2D图像。test-volume.tif包含数据。）

从[ISBI Challenge](http://brainiac2.mit.edu/isbi_challenge/home)下载如下3个文件到当前目录的data文件夹下：

```
train-labels.tif
train-volume.tif
test-volume.tif
```

### 脚本准备

从[课程gitee仓库](https://gitee.com/mindspore/course)上下载本实验相关脚本。将脚本和数据集组织为如下形式：

```
unet
├── data
│   ├── test-volume.tif
│   ├── train-labels.tif
│   └── train-volume.tif
├── src
│   ├── config.py
│   ├── data_loader.py
│   ├── loss.py
│   ├── utils.py
│   └── unet
│   	├── __init__.py
│   	├── unet_model.py
│   	└── unet_parts.py
├── main.py
└── README.md
```

### 创建OBS桶

使用ModelArts训练作业/Notebook时，需要使用华为云OBS存储实验脚本和数据集，可以参考[快速通过OBS控制台上传下载文件](https://support.huaweicloud.com/qs-obs/obs_qs_0001.html)了解使用OBS创建桶、上传文件、下载文件的使用方法（下文给出了操作步骤）。

> **提示：** 华为云新用户使用OBS时通常需要创建和配置“访问密钥”，可以在使用OBS时根据提示完成创建和配置。也可以参考[获取访问密钥并完成ModelArts全局配置](https://support.huaweicloud.com/prepare-modelarts/modelarts_08_0002.html)获取并配置访问密钥。

打开[OBS控制台](https://storage.huaweicloud.com/obs/?region=cn-north-4&locale=zh-cn#/obs/manager/buckets)，点击右上角的“创建桶”按钮进入桶配置页面，创建OBS桶的参考配置如下：

- 区域：华北-北京四
- 数据冗余存储策略：单AZ存储
- 桶名称：全局唯一的字符串
- 存储类别：标准存储
- 桶策略：公共读
- 归档数据直读：关闭
- 企业项目、标签等配置：免

### 上传文件

点击新建的OBS桶名，再打开“对象”标签页，通过“上传对象”、“新建文件夹”等功能，将脚本和数据集上传到OBS桶中。上传文件后，查看页面底部的“任务管理”状态栏（正在运行、已完成、失败），确保文件均上传完成。若失败请：

- 参考[上传对象大小限制/切换上传方式](https://support.huaweicloud.com/qs-obs/obs_qs_0008.html)，
- 参考[上传对象失败常见原因](https://support.huaweicloud.com/obs_faq/obs_faq_0134.html)。
- 若无法解决请[新建工单](https://console.huaweicloud.com/ticket/?region=cn-north-4&locale=zh-cn#/ticketindex/createIndex)，产品类为“对象存储服务”，问题类型为“桶和对象相关”，会有技术人员协助解决。

## 实验步骤（ModelArts训练作业）

ModelArts提供了训练作业服务，训练作业资源池大，且具有作业排队等功能，适合大规模并发使用。使用训练作业时，如果有修改代码和调试的需求，有如下三个方案：

1. 在本地修改代码后重新上传；

2. 使用[PyCharm ToolKit](https://support.huaweicloud.com/tg-modelarts/modelarts_15_0001.html)配置一个本地Pycharm+ModelArts的开发环境，便于上传代码、提交训练作业和获取训练日志。

3. 在ModelArts上创建Notebook，然后设置[Sync OBS功能](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0038.html)，可以在线修改代码并自动同步到OBS中。因为只用Notebook来编辑代码，所以创建CPU类型最低规格的Notebook就行。

### 适配训练作业

创建训练作业时，运行参数会通过脚本传参的方式输入给脚本代码，脚本必须解析传参才能在代码中使用相应参数。如data_url和train_url，分别对应数据存储路径(OBS路径)和训练输出路径(OBS路径)。脚本对传参进行解析后赋值到`args`变量里，在后续代码里可以使用。

```python
import argparse
parser = argparse.ArgumentParser(description='UNET')
parser.add_argument('--data_url', required=True, help='Location of data.')
parser.add_argument('--train_url', required=True, default=None, help='Location of training outputs.')
args_opt = parser.parse_args()
```

MindSpore暂时没有提供直接访问OBS数据的接口，需要通过ModelArts自带的moxing框架与OBS交互。拷贝自己账户下或他人共享的OBS桶内的数据集至执行容器。

```python
import moxing as mox
# src_url形如's3://OBS/PATH'，为OBS桶中数据集的路径，dst_url为执行容器中的路径
mox.file.copy_parallel(src_url=args_opt.data_url, dst_url='./data')
```

### 创建训练作业

可以参考[使用常用框架训练模型](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0238.html)来创建并启动训练作业（下文给出了操作步骤）。

打开[ModelArts控制台-训练管理-训练作业](https://console.huaweicloud.com/modelarts/?region=cn-north-4#/trainingJobs)，点击“创建”按钮进入训练作业配置页面，创建训练作业的参考配置：

- 算法来源：常用框架->Ascend-Powered-Engine->MindSpore
- 代码目录：选择上述新建的OBS桶中的unet目录
- 启动文件：选择上述新建的OBS桶中的unet目录下的`main.py`
- 数据来源：数据存储位置->选择上述新建的OBS桶中的unet目录下的data目录
- 训练输出位置：选择上述新建的OBS桶中的unet目录并在其中创建output目录
- 作业日志路径：同训练输出位置
- 规格：Ascend:1*Ascend 910
- 其他均为默认

启动并查看训练过程：

1. 点击提交以开始训练；
2. 在训练作业列表里可以看到刚创建的训练作业，在训练作业页面可以看到版本管理；
3. 点击运行中的训练作业，在展开的窗口中可以查看作业配置信息，以及训练过程中的日志，日志会不断刷新，等训练作业完成后也可以下载日志到本地进行查看；
4. 参考实验步骤（ModelArts Notebook），在日志中找到对应的打印信息，检查实验是否成功。

## 实验步骤（ModelArts Notebook）

ModelArts Notebook资源池较小，且每个运行中的Notebook会一直占用Device资源不释放，不适合大规模并发使用（不使用时需停止实例，以释放资源）。

### 创建Notebook

可以参考[创建并打开Notebook](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0034.html)来创建并打开Notebook（下文给出了操作步骤）。

打开[ModelArts控制台-开发环境-Notebook](https://console.huaweicloud.com/modelarts/?region=cn-north-4#/notebook)，点击“创建”按钮进入Notebook配置页面，创建Notebook的参考配置：

- 计费模式：按需计费
- 名称：unet
- 工作环境：公共镜像->Ascend-Powered-Engine 1.0 (python3)
- 资源池：公共资源
- 类型：Ascend
- 规格：单卡1*Ascend 910
- 存储位置：对象存储服务（OBS）->选择上述新建的OBS桶中的unet文件夹
- 自动停止：打开->选择1小时后（后续可在Notebook中随时调整）

> **注意：**
>
> - 在Jupyter Notebook/JupyterLab文件列表里，展示的是关联的OBS桶里的文件，并不在当前Notebook工作环境（容器）中，Notebook中的代码无法直接访问这些文件。
> - 打开Notebook前，选中文件列表里的所有文件/文件夹（实验脚本和数据集），并点击列表上方的“Sync OBS”按钮，使OBS桶中的所有文件同时同步到Notebook执行容器中，这样Notebook中的代码才能访问数据集。
>   - 使用Notebook时，可参考[与OBS同步文件](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0038.html)；
>   - 使用JupyterLab时，可参考[与OBS同步文件](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0336.html)。
>   - 同步文件的大小和数量超过限制时，请参考[MoXing常用操作示例](https://support.huaweicloud.com/moxing-devg-modelarts/modelarts_11_0005.html#section5)中的拷贝操作，将大文件（如数据集）拷贝到Notebook容器中。
> - Notebook/JupyterLab文件列表页面的“Upload/上传”功能，会将文件上传至OBS桶中，而不是Notebook执行容器中，仍需额外同步/拷贝。
> - 在Notebook里通过代码/命令（如`wget, git`、python`urllib, requests`等）获取的文件，存在于Notebook执行容器中，但不会显示在文件列表里。
> - 每个Notebook实例仅被分配了1个Device，如果在一个实例中打开多个Notebook页面（即多个进程），运行其中一个页面上的MindSpore代码时，请关闭其他页面的kernel，否则会出现Device被占用的错误。
> - Notebook运行中一直处于计费状态，不使用时，在Notebook控制台页面点击实例右侧的“停止”，以停止计费。停止后，Notebook里的内容不会丢失（已同步至OBS）。下次需要使用时，点击实例右侧的“启动”即可。可参考[启动或停止Notebook实例](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0041.html)。

打开Notebook后，选择MindSpore环境作为Kernel。

> **提示：** 
>
> - 上述数据集和脚本的准备工作也可以在Notebook环境中完成，在Jupyter Notebook文件列表页面，点击右上角的"New"->"Terminal"，进入Notebook环境所在终端，进入`work`目录，可以使用常用的linux shell命令，如`wget, gzip, tar, mkdir, mv`等，完成数据集和脚本的下载和准备。
> - 可将如下每段代码拷贝到Notebook代码框/Cell中，从上至下阅读提示并执行代码框进行体验。代码框执行过程中左侧呈现[\*]，代码框执行完毕后左侧呈现如[1]，[2]等。请等上一个代码框执行完毕后再执行下一个代码框。

### 导入模块

导入MindSpore模块和辅助模块，设置MindSpore上下文，如执行模式、设备等。

```python
import os
import ast
import numpy as np

import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as F
from mindspore import Model, context
from mindspore.nn.loss.loss import _Loss
from mindspore.communication.management import init, get_group_size
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.unet import UNet
from src.data_loader import create_dataset
from src.loss import CrossEntropyWithLogits
from src.utils import StepLossTimeMonitor
from src.config import cfg_unet
from scipy.special import softmax

device_id = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False, device_id=device_id)

mindspore.set_seed(1)
```

###  定义损失函数

```python
class CrossEntropyWithLogits(_Loss):
    def __init__(self):
        super(CrossEntropyWithLogits, self).__init__()
        self.transpose_fn = F.Transpose()
        self.reshape_fn = F.Reshape()
        self.softmax_cross_entropy_loss = nn.SoftmaxCrossEntropyWithLogits()
        self.cast = F.Cast()
    def construct(self, logits, label):
        # NCHW->NHWC
        logits = self.transpose_fn(logits, (0, 2, 3, 1))
        logits = self.cast(logits, mindspore.float32)
        label = self.transpose_fn(label, (0, 2, 3, 1))

        loss = self.reduce_mean(self.softmax_cross_entropy_loss(self.reshape_fn(logits, (-1, 2)), 
                                                                self.reshape_fn(label, (-1, 2))))
        return self.get_loss(loss)
```

### 定义验证

```python
class dice_coeff(nn.Metric):
    def __init__(self):
        super(dice_coeff, self).__init__()
        self.clear()
    def clear(self):
        self._dice_coeff_sum = 0
        self._samples_num = 0

    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError('Mean dice coeffcient need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))

        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        self._samples_num += y.shape[0]
        y_pred = y_pred.transpose(0, 2, 3, 1)
        y = y.transpose(0, 2, 3, 1)
        y_pred = softmax(y_pred, axis=3)

        inter = np.dot(y_pred.flatten(), y.flatten())
        union = np.dot(y_pred.flatten(), y_pred.flatten()) + np.dot(y.flatten(), y.flatten())

        single_dice_coeff = 2*float(inter)/float(union+1e-6)
        print("single dice coeff is:", single_dice_coeff)
        self._dice_coeff_sum += single_dice_coeff

    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return self._dice_coeff_sum / float(self._samples_num)
```

### 定义训练过程

参数传入训练数据集（`data_dir`）和训练参数，构建网络、损失函数、优化器等，并配置好`CheckPoint`生成信息，然后使用`model.train`接口，进行模型训练。

```python
def train_net(data_dir, cross_valid_ind=1, epochs=400, batch_size=16, lr=0.0001, run_distribute=False, cfg=None):

    if run_distribute:
        init()
        group_size = get_group_size()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode,
                                          device_num=group_size,
                                          gradients_mean=False)
    net = UNet(n_channels=cfg['num_channels'], n_classes=cfg['num_classes'])

    if cfg['resume']:
        param_dict = load_checkpoint(cfg['resume_ckpt'])
        load_param_into_net(net, param_dict)

    criterion = CrossEntropyWithLogits()
    train_dataset, _ = create_dataset(data_dir, epochs, batch_size, True, cross_valid_ind, run_distribute)
    train_data_size = train_dataset.get_dataset_size()
    print("dataset length is:", train_data_size)
    ckpt_config = CheckpointConfig(save_checkpoint_steps=train_data_size,
                                   keep_checkpoint_max=cfg['keep_checkpoint_max'])
    ckpoint_cb = ModelCheckpoint(prefix='ckpt_unet_medical_adam',
                                 directory='./ckpt_{}/'.format(device_id),
                                 config=ckpt_config)

    optimizer = nn.Adam(params=net.trainable_params(), learning_rate=lr, weight_decay=cfg['weight_decay'],
                        loss_scale=cfg['loss_scale'])

    loss_scale_manager = mindspore.train.loss_scale_manager.FixedLossScaleManager(cfg['FixedLossScaleManager'], False)

    model = Model(net, loss_fn=criterion, loss_scale_manager=loss_scale_manager, optimizer=optimizer, amp_level="O3")

    print("============== Starting Training ==============")
    model.train(1, train_dataset, callbacks=[StepLossTimeMonitor(batch_size=batch_size), ckpoint_cb],
                dataset_sink_mode=False)
    print("============== End Training ==============")
```

### 定义模型验证

```python
def test_net(data_dir, ckpt_path, cross_valid_ind=1, cfg=None):

    net = UNet(n_channels=cfg['num_channels'], n_classes=cfg['num_classes'])
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)

    criterion = CrossEntropyWithLogits()
    _, valid_dataset = create_dataset(data_dir, 1, 1, False, cross_valid_ind, False)
    model = Model(net, loss_fn=criterion, metrics={"dice_coeff": dice_coeff()})

    print("============== Starting Evaluating ============")
    dice_score = model.eval(valid_dataset, dataset_sink_mode=False)
    print("Cross valid dice coeff is:", dice_score)
```

### 运行训练验证

定义数据集路径以及保存的ckpt文件路径以用于模型的训练和验证。

```python
data_url = './data'
ckpt_path = './ckpt_0/ckpt_unet_medical_adam-1_600.ckpt'
run_distribute = False
epoch_size = cfg_unet['epochs'] if not run_distribute else cfg_unet['distribute_epochs']
    
train_net(data_dir=data_url,
          cross_valid_ind=cfg_unet['cross_valid_ind'],
          epochs=epoch_size,
          batch_size=cfg_unet['batchsize'],
          lr=cfg_unet['lr'],
          run_distribute=run_distribute,
          cfg=cfg_unet)
    
print('*'*60)
    
test_net(data_dir=data_url,
         ckpt_path=ckpt_path,
         cross_valid_ind=cfg_unet['cross_valid_ind'],
         cfg=cfg_unet)
```

```python
dataset length is: 600
============== Starting Training ==============
step: 1, loss is 0.7064769, fps is 0.53297180539365
step: 2, loss is 0.6923242, fps is 60.30013460159545
step: 3, loss is 0.68789375, fps is 60.394233157544235
step: 4, loss is 0.6804908, fps is 60.43714691491556
step: 5, loss is 0.6662822, fps is 60.47554353213432
...
step: 596, loss is 0.20853733, fps is 60.4342623311036
step: 597, loss is 0.19242267, fps is 60.52064689919999
step: 598, loss is 0.2004529, fps is 60.450539343959576
step: 599, loss is 0.19381239, fps is 60.491842792461064
step: 600, loss is 0.20040925, fps is 60.42767780364049
============== End Training ==============
************************************************************
============== Starting Evaluating ============
single dice coeff is: 0.9010871252916888
single dice coeff is: 0.9027840532821939
single dice coeff is: 0.9233969994354486
single dice coeff is: 0.9211319096981847
single dice coeff is: 0.9118504078664812
single dice coeff is: 0.902567889358818
Cross valid dice coeff is: {'dice_coeff': 0.9104697308221358}
```

## 实验小结

本实验展示了如何使用MindSpore进行图片分割任务的训练和验证，以及开发和训练Unet模型。通过对Unet模型做几代的训练，然后使用训练后的Unet模型对数据集进行评估验证，准确率大于90%。即Unet模型学习到了如何进行图片分割。