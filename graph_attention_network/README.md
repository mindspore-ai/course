#  Graph Attention Network

## 实验介绍

图神经网络（Graph Neural Network, GNN）把深度学习应用到图结构（Graph）中，其中的图卷积网络（Graph Convolutional Network，GCN）可以在Graph上进行卷积操作。但是GCN存在一些缺陷：依赖拉普拉斯矩阵，不能直接用于有向图；模型训练依赖于整个图结构，不能用于动态图；卷积的时候没办法为邻居节点分配不同的权重。
[图注意力网络（Graph Attention Networks）](https://arxiv.org/pdf/1710.10903.pdf)由Petar Veličković等人于2018年提出。GAT采用了Attention机制，可以为不同节点分配不同权重，训练时依赖于成对的相邻节点，而不依赖具体的网络结构，可以用于inductive任务。

[1] https://baijiahao.baidu.com/s?id=1671028964544884749

本实验主要介绍在Cora和Citeseer数据集上使用MindSpore进行图注意力网络的训练和验证。

## 实验目的

- 了解GAT相关知识。
- 在MindSpore中使用Cora和Citeseer数据集训练和验证GAT。
- 了解MindSpore的model_zoo模块，以及如何使用model_zoo中的模型。

## 预备知识

- 熟练使用Python，了解Shell及Linux操作系统基本知识。
- 具备一定的深度学习理论知识，如图卷积网络、图注意力网络、损失函数、优化器，训练策略等。
- 了解华为云的基本使用方法，包括[OBS（对象存储）](https://www.huaweicloud.com/product/obs.html)、[ModelArts（AI开发平台）](https://www.huaweicloud.com/product/modelarts.html)、[Notebook（开发工具）](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0032.html)、[训练作业](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0238.html)等服务。华为云官网：https://www.huaweicloud.com
- 了解并熟悉MindSpore AI计算框架，MindSpore官网：https://www.mindspore.cn

## 实验环境

- MindSpore 1.0.0（MindSpore版本会定期更新，本指导也会定期刷新，与版本配套）；
- 华为云ModelArts：ModelArts是华为云提供的面向开发者的一站式AI开发平台，集成了昇腾AI处理器资源池，用户可以在该平台下体验MindSpore。ModelArts官网：https://www.huaweicloud.com/product/modelarts.html

## 实验准备

### 数据集准备

Cora和CiteSeer是图神经网络常用的数据集，数据集官网[LINQS Datasets](https://linqs.soe.ucsc.edu/data)。

Cora数据集包含2708个科学出版物，分为七个类别。 引用网络由5429个链接组成。 数据集中的每个出版物都用一个0/1值的词向量描述，0/1指示词向量中是否出现字典中相应的词。 该词典包含1433个独特的单词。 数据集中的README文件提供了更多详细信息。

CiteSeer数据集包含3312种科学出版物，分为六类。 引用网络由4732个链接组成。 数据集中的每个出版物都用一个0/1值的词向量描述，0/1指示词向量中是否出现字典中相应的词。 该词典包含3703个独特的单词。 数据集中的README文件提供了更多详细信息。

本实验使用Github上[kimiyoung/planetoid](https://github.com/kimiyoung/planetoid/tree/master/data)预处理和划分好的数据集。

将数据集放置到所需的路径下，该文件夹应包含以下文件：

```
data 
├── ind.cora.allx 
├── ind.cora.ally 
├── ...
├── ind.cora.test.index 
├── trans.citeseer.tx
├── trans.citeseer.ty
├── ...
└── trans.pubmed.y
```

inductive模型的输入包含：

- `x`，已标记的训练实例的特征向量，
- `y`，已标记的训练实例的one-hot标签，
- `allx`，标记的和未标记的训练实例（`x`的超集）的特征向量，
- `graph`，一个`dict`，格式为`{index: [index_of_neighbor_nodes]}.`

令n为标记和未标记训练实例的数量。在`graph`中这n个实例的索引应从0到n-1，其顺序与`allx`中的顺序相同。

除了`x`，`y`，`allx`，和`graph`如上所述，预处理的数据集还包括：

- `tx`，测试实例的特征向量，
- `ty`，测试实例的one-hot标签，
- `test.index`，`graph`中测试实例的索引，
- `ally`，是`allx`中实例的标签。

### 脚本准备

从[课程gitee仓库](https://gitee.com/mindspore/course)上下载本实验相关脚本。将脚本和数据集组织为如下形式：

```
gat
├── data
├── graph_to_mindrecord 
│   ├── citeseer
│   ├── cora
│   ├── graph_map_schema.py
│   └── writer.py
├── src
│   ├── utils.py
│   ├── gat.py
│   ├── dataset.py
│   └── config.py
│── main.py
└── README.md
```

### 创建OBS桶

本实验需要使用华为云OBS存储实验脚本和数据集，可以参考[快速通过OBS控制台上传下载文件](https://support.huaweicloud.com/qs-obs/obs_qs_0001.html)了解使用OBS创建桶、上传文件、下载文件的使用方法（下文给出了操作步骤）。

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

###  适配训练作业

创建训练作业时，运行参数会通过脚本传参的方式输入给脚本代码，脚本必须解析传参才能在代码中使用相应参数。如data_url和train_url，分别对应数据存储路径(OBS路径)和训练输出路径(OBS路径)。脚本对传参进行解析后赋值到`args`变量里，在后续代码里可以使用。

```python
import argparse
parser = argparse.ArgumentParser(description='GAT')
parser.add_argument('--data_url', required=True, help='Location of data.')
parser.add_argument('--train_url', required=True, help='Location of training outputs.')
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
- 代码目录：选择上述新建的OBS桶中的gat目录
- 启动文件：选择上述新建的OBS桶中的gat目录下的`main.py`
- 数据来源：数据存储位置->选择上述新建的OBS桶中的gat目录下的data目录
- 训练输出位置：选择上述新建的OBS桶中的gat目录并在其中创建output目录
- 作业日志路径：同训练输出位置
- 规格：Ascend:1*Ascend 910
- 其他均为默认

启动并查看训练过程：

1. 点击提交以开始训练；
2. 在训练作业列表里可以看到刚创建的训练作业，在训练作业页面可以看到版本管理；
3. 点击运行中的训练作业，在展开的窗口中可以查看作业配置信息，以及训练过程中的日志，日志会不断刷新，等训练作业完成后也可以下载日志到本地进行查看；
4. 参考实验步骤（ModelArts Notebook），在日志中找到对应的打印信息，检查实验是否成功。

## 实验步骤（ModelArts Notebook）

推荐使用ModelArts训练作业进行实验，适合大规模并发使用。若使用ModelArts Notebook，请参考[LeNet5](../lenet5)及[Checkpoint](../checkpoint)实验案例，了解Notebook的使用方法和注意事项。

### 导入模块

导入MindSpore模块和辅助模块，设置MindSpore上下文，如执行模式、设备等。

```python
import os

import argparse
import numpy as np

from easydict import EasyDict as edict
from mindspore import context

from src.gat import GAT
from src.config import GatConfig
from src.dataset import load_and_process
from src.utils import LossAccuracyWrapper, TrainGAT
from graph_to_mindrecord.writer import run
from mindspore.train.serialization import load_checkpoint, save_checkpoint

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
                save_checkpoint(train_net.network, "ckpts/gat.ckpt")
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
dataname = 'cora'
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

# 转换数据格式
print("============== Graph To Mindrecord ==============")
run(cfg)
    
#训练
print("============== Starting Training ==============")
train(cfg)
```

### 实验结果

训练结果将打印如下结果：

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

下表显示了Cora数据集上的结果：

|                              | MindSpore + Ascend910 | Tensorflow + V100 |
| ---------------------------- | --------------------- | :---------------- |
| 精度                         | 0.830933271           | 0.828649968       |
| 训练耗时（200 epochs）       | 27.62298311 s         | 36.711862 s       |
| 端到端训练耗时（200 epochs） | 39.074 s              | 50.894 s          |
