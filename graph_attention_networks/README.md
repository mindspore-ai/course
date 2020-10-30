#  Graph Attention Networks

## 实验介绍

图注意力网络（Graph Attention Networks）由PetarVeličković等人于2017年提出。通过利用屏蔽的自我关注层来解决基于现有图的方法的缺点，GAT在转导数据集（例如Cora）和归纳数据集（例如PPI）上都达到或匹配了最新技术。

本实验主要介绍在下载的Cora和Citeseer数据集上使用MindSpore进行图注意力网络的训练。

## 实验目的

- 了解GAT相关知识。
- 在MindSpore中使用Cora和Citeseer数据集训练GAT示例。
- 了解MindSpore的model_zoo模块，以及如何使用model_zoo中的模型。

## 预备知识

- 熟练使用Python，了解Shell及Linux操作系统基本知识。
- 具备一定的深度学习理论知识，如卷积神经网络、图卷积网络、图注意力网络等。
- 了解并熟悉MindSpore AI计算框架，MindSpore官网：[https://www.mindspore.cn](https://www.mindspore.cn/)

## 实验环境

- MindSpore 0.5.0（MindSpore版本会定期更新，本指导也会定期刷新，与版本配套）；
- 华为云ModelArts：ModelArts是华为云提供的面向开发者的一站式AI开发平台，集成了昇腾AI处理器资源池，用户可以在该平台下体验MindSpore。ModelArts官网：https://www.huaweicloud.com/product/modelarts.html

## 实验准备

### 数据集准备

在[59.36.11.51](59.36.11.51)服务器上下载已经转换为MindRecord格式的Cora和Citeseer数据集文件。

**Cora：**dataset/workspace/mindspore_dataset/cora/cora_mr/

**Citeseer：**dataset/workspace/mindspore_dataset/citeseer/citeseer_mr/

将数据集放置到data_mr文件夹下，该文件夹应包含以下文件：

```
└─data_mr 
    ├─citeseer_mr
    ├─citeseer_mr.db
    ├─cora_mr
    └─cora_mr.db
```

### 脚本准备

从[课程gitee仓库](https://gitee.com/mindspore/course)上下载本实验相关脚本。将脚本和数据集组织为如下形式：

```
gat
├── data_mr
│   ├── citeseer_mr
│   ├── citeseer_mr.db
│   ├── cora_mr
│   └── cora_mr.db
├── src
│   ├── utils.py
│   ├── gat.py
│   ├── dataset.py
│   └── config.py
│── main.py
└── README.md
```

### 创建OBS桶

本实验需要使用华为云OBS存储实验脚本和数据集，可以参考[快速通过OBS控制台上传下载文件](https://support.huaweicloud.com/qs-obs/obs_qs_0001.html)了解使用OBS创建桶、上传文件、下载文件的使用方法。

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

点击新建的OBS桶名，再打开“对象”标签页，通过“上传对象”、“新建文件夹”等功能，将脚本和数据集上传到OBS桶中。

## 实验步骤(ModelArts Notebook)

ModelArts Notebook资源池较小，且每个运行中的Notebook会一直占用Device资源不释放，不适合大规模并发使用（不使用时需停止实例，以释放资源）。

### 创建Notebook

可以参考[创建并打开Notebook](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0034.html)来创建并打开本实验的Notebook脚本。

打开[ModelArts控制台-开发环境-Notebook](https://console.huaweicloud.com/modelarts/?region=cn-north-4#/notebook)，点击“创建”按钮进入Notebook配置页面，创建Notebook的参考配置：

- 计费模式：按需计费
- 名称：gat
- 工作环境：Python3
- 资源池：公共资源
- 类型：Ascend
- 规格：单卡1*Ascend 910
- 存储位置：对象存储服务（OBS）->选择上述新建的OBS桶中的gat文件夹
- 自动停止：打开->选择1小时后（后续可在Notebook中随时调整）

> **注意：**
>
> - 在Jupyter Notebook/JupyterLab文件列表里，展示的是关联的OBS桶里的文件，并不在当前Notebook工作环境（容器）中，Notebook中的代码无法直接访问这些文件。
> - 打开Notebook前，选中文件列表里的所有文件/文件夹（实验脚本和数据集），并点击列表上方的“Sync OBS”按钮，使OBS桶中的所有文件同时同步到Notebook执行容器中，这样Notebook中的代码才能访问数据集。
>   - 使用Jupyter Notebook时，可参考[与OBS同步文件](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0038.html)；
>   - 使用JupyterLab时，可参考[与OBS同步文件](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0336.html)。
>   - 同步文件的大小和数量超过限制时，请参考[MoXing常用操作示例](https://support.huaweicloud.com/moxing-devg-modelarts/modelarts_11_0005.html#section5)中的拷贝操作，将大文件（如数据集）拷贝到Notebook容器中。
> - 打开Notebook后，选择MindSpore环境作为Kernel。
> - Notebook运行中一直处于计费状态，不使用时，在Notebook控制台页面点击实例右侧的“停止”，以停止计费。停止后，Notebook里的内容不会丢失（已同步至OBS）。下次需要使用时，点击实例右侧的“启动”即可。可参考[启动或停止Notebook实例](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0041.html)。

> **提示：** 
>
> - 上述数据集和脚本的准备工作也可以在Notebook环境中完成，在Jupyter Notebook文件列表页面，点击右上角的"New"->"Terminal"，进入Notebook环境所在终端，进入`work`目录，可以使用常用的linux shell命令，如`wget, gzip, tar, mkdir, mv`等，完成数据集和脚本的下载和准备。
> - 可将如下每段代码拷贝到Notebook代码框/Cell中，从上至下阅读提示并执行代码框进行体验。代码框执行过程中左侧呈现[\*]，代码框执行完毕后左侧呈现如[1]，[2]等。请等上一个代码框执行完毕后再执行下一个代码框。

### 导入模块

导入MindSpore模块和辅助模块，设置MindSpore上下文，如执行模式、设备等。

```python
import os
import numpy as np

from easydict import EasyDict as edict
from mindspore import context

from src.gat import GAT
from src.config import GatConfig
from src.dataset import load_and_process
from src.utils import LossAccuracyWrapper, TrainGAT
from mindspore.train.serialization import load_checkpoint, _exec_save_checkpoint

# os.environ['DEVICE_ID']='0'
context.set_context(mode=context.GRAPH_MODE,device_target="Ascend", save_graphs=False)
```

### 参数配置

训练参数可以在config.py中设置。

```python
"learning_rate": 0.005,            # Learning rate
"num_epochs": 200,                 # Epoch sizes for training
"hid_units": [8],                  # Hidden units for attention head at each layer
"n_heads": [8, 1],                 # Num heads for each layer
"early_stopping": 100,             # Early stop patience
"l2_coeff": 0.0005                 # l2 coefficient
"attn_dropout": 0.6                # Attention dropout ratio
"feature_dropout":0.6              # Feature dropout ratio
```

### 定义训练

```python
def train(args_opt):
    """Train GAT model."""

    if not os.path.exists("ckpts"):
        os.mkdir("ckpts")

    # train parameters
    hid_units = GatConfig.hid_units
    n_heads = GatConfig.n_heads
    early_stopping = GatConfig.early_stopping
    lr = GatConfig.lr
    l2_coeff = GatConfig.l2_coeff
    num_epochs = GatConfig.num_epochs
    feature, biases, y_train, train_mask, y_val, eval_mask, y_test, test_mask = load_and_process(args_opt.data_dir,
                                                                                                 args_opt.train_nodes_num,
                                                                                                 args_opt.eval_nodes_num,
                                                                                                 args_opt.test_nodes_num)
    feature_size = feature.shape[2]
    num_nodes = feature.shape[1]
    num_class = y_train.shape[2]

    gat_net = GAT(feature,
                  biases,
                  feature_size,
                  num_class,
                  num_nodes,
                  hid_units,
                  n_heads,
                  attn_drop=GatConfig.attn_dropout,
                  ftr_drop=GatConfig.feature_dropout)
    gat_net.add_flags_recursive(fp16=True)

    eval_net = LossAccuracyWrapper(gat_net,
                                   num_class,
                                   y_val,
                                   eval_mask,
                                   l2_coeff)

    train_net = TrainGAT(gat_net,
                         num_class,
                         y_train,
                         train_mask,
                         lr,
                         l2_coeff)

    train_net.set_train(True)
    val_acc_max = 0.0
    val_loss_min = np.inf
    for _epoch in range(num_epochs):
        train_result = train_net()
        train_loss = train_result[0].asnumpy()
        train_acc = train_result[1].asnumpy()

        eval_result = eval_net()
        eval_loss = eval_result[0].asnumpy()
        eval_acc = eval_result[1].asnumpy()

        print("Epoch:{}, train loss={:.5f}, train acc={:.5f} | val loss={:.5f}, val acc={:.5f}".format(
            _epoch, train_loss, train_acc, eval_loss, eval_acc))
        if eval_acc >= val_acc_max or eval_loss < val_loss_min:
            if eval_acc >= val_acc_max and eval_loss < val_loss_min:
                val_acc_model = eval_acc
                val_loss_model = eval_loss
                if os.path.exists('ckpts/gat.ckpt'):
                    os.remove('ckpts/gat.ckpt')
                _exec_save_checkpoint(train_net.network, "ckpts/gat.ckpt")
            val_acc_max = np.max((val_acc_max, eval_acc))
            val_loss_min = np.min((val_loss_min, eval_loss))
            curr_step = 0
        else:
            curr_step += 1
            if curr_step == early_stopping:
                print("Early Stop Triggered!, Min loss: {}, Max accuracy: {}".format(val_loss_min, val_acc_max))
                print("Early stop model validation loss: {}, accuracy{}".format(val_loss_model, val_acc_model))
                break
    gat_net_test = GAT(feature,
                       biases,
                       feature_size,
                       num_class,
                       num_nodes,
                       hid_units,
                       n_heads,
                       attn_drop=0.0,
                       ftr_drop=0.0)
    load_checkpoint("ckpts/gat.ckpt", net=gat_net_test)
    gat_net_test.add_flags_recursive(fp16=True)

    test_net = LossAccuracyWrapper(gat_net_test,
                                   num_class,
                                   y_test,
                                   test_mask,
                                   l2_coeff)
    test_result = test_net()
    print("Test loss={}, test acc={}".format(test_result[0], test_result[1]))
```

### 运行训练

使用不同的数据集训练操作起来非常方便，只需要将参数`dataname`修改为需要训练的数据集名称即可。

```python
#------------------------定义变量------------------------------
dataname = 'cora_mr'
datadir_save = './data_mr'
datadir = os.path.join(datadir_save, dataname)

cfg = edict({
    'SRC_PATH': './data',
    'MINDRECORD_PATH': datadir_save,
    'DATASET_NAME': dataname,  # citeseer,cora
    'mindrecord_partitions':1,
    'mindrecord_header_size_by_bit' : 18,
    'mindrecord_page_size_by_bit' : 20,

    'data_dir': datadir,
    'seed' : 123,
    'train_nodes_num':140,
    'eval_nodes_num':500,
    'test_nodes_num':1000
})

#训练
print("============== Starting Training ==============")
train(cfg)
```

### 实验结果

训练结果将打印如下结果：

```python
============== Starting Training ==============
Epoch:0, train loss=1.98498 train acc=0.17143 | val loss=1.97946 val acc=0.27200
Epoch:1, train loss=1.98345 train acc=0.15000 | val loss=1.97233 val acc=0.32600
Epoch:2, train loss=1.96968 train acc=0.21429 | val loss=1.96747 val acc=0.37400
Epoch:3, train loss=1.97061 train acc=0.20714 | val loss=1.96410 val acc=0.47600
Epoch:4, train loss=1.96864 train acc=0.13571 | val loss=1.96066 val acc=0.59600
...
Epoch:195, train loss=1.45111 train_acc=0.56429 | val_loss=1.44325 val_acc=0.81200
Epoch:196, train loss=1.52476 train_acc=0.52143 | val_loss=1.43871 val_acc=0.81200
Epoch:197, train loss=1.35807 train_acc=0.62857 | val_loss=1.43364 val_acc=0.81400
Epoch:198, train loss=1.47566 train_acc=0.51429 | val_loss=1.42948 val_acc=0.81000
Epoch:199, train loss=1.56411 train_acc=0.55000 | val_loss=1.42632 val_acc=0.80600
Test loss=1.5366285, test acc=0.84199995
```

下表显示了Cora数据集上的结果：

|                              | MindSpore + Ascend910 | Tensorflow + V100 |
| ---------------------------- | --------------------- | :---------------- |
| 精度                         | 0.830933271           | 0.828649968       |
| 训练耗时（200 epochs）       | 27.62298311 s         | 36.711862 s       |
| 端到端训练耗时（200 epochs） | 39.074 s              | 50.894 s          |

