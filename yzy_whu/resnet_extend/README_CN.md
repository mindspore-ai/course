# 目录

<!-- TOC -->

- [实验概述](#实验概述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [实验内容](#实验内容)
- [运行](# 运行)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
- [ModelZoo主页](#ModelZoo主页)

<!-- /TOC -->

# 实验概述

## 概述

残差神经网络（ResNet）由微软研究院何凯明等五位华人提出，通过ResNet单元，成功训练152层神经网络，赢得了ILSVRC2015冠军。ResNet前五项的误差率为3.57%，参数量低于VGGNet，因此效果非常显著。传统的卷积网络或全连接网络或多或少存在信息丢失的问题，还会造成梯度消失或爆炸，导致深度网络训练失败，ResNet则在一定程度上解决了这个问题。通过将输入信息传递给输出，确保信息完整性。整个网络只需要学习输入和输出的差异部分，简化了学习目标和难度。ResNet的结构大幅提高了神经网络训练的速度，并且大大提高了模型的准确率。正因如此，ResNet十分受欢迎，甚至可以直接用于ConceptNet网络。

值得一提的是，本实验旨在通过其modelzoo提供的官方源码进行魔改，以达到在这个过程中提升对于mindspore框架的使用熟练度和对深度学习知识的动手能力。下面以魔改代码使其能够运行resnet18为例展开

## 参考资料

1. [model_zoo · MindSpore/mindspore - 码云 - 开源中国 (gitee.com)](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)：mindspore modelazoo的官方示例代码

2. [course: MindSpore实验。](https://gitee.com/mindspore/course/tree/master/resnet50)：这位小兄弟参考其官方源码运行的示例

3. [rishivar/Resnet-18](https://github.com/rishivar/Resnet-18)：基于其源码再创造的resnet18网络

# 模型架构

ResNet的总体网络架构如下：
[链接](https://arxiv.org/pdf/1512.03385.pdf)

# 数据集

使用的数据集：[CIFAR-10](<http://www.cs.toronto.edu/~kriz/cifar.html>)

- 数据集大小：共10个类、60,000个32*32彩色图像
    - 训练集：50,000个图像
    - 测试集：10,000个图像
- 数据格式：二进制文件
    - 注：数据在dataset.py中处理。
- 下载数据集。目录结构如下：

```text
├─cifar-10-batches-bin
│
└─cifar-10-verify-bin
```

# 环境要求

- 硬件(Ascend/GPU)
    - 准备Ascend或GPU处理器搭建硬件环境。如需试用昇腾处理器，请发送[申请表](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/file/other/Ascend%20Model%20Zoo%E4%BD%93%E9%AA%8C%E8%B5%84%E6%BA%90%E7%94%B3%E8%AF%B7%E8%A1%A8.docx)至ascend@huawei.com，审核通过即可获得资源。
    - 本实验中采用的是ascend处理器的notebook环境，可以通过华为云的modelarts服务购买，搭配obs使用。具体使用方法建议参考[参考资料2](https://gitee.com/mindspore/course/tree/master/resnet50)其中的readme的做法，此处不再赘述。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# 实验内容

实验主要针对train、src/config、src/resnet等文件进行了更改，通过学习resnet网络结构，观察已经给出了resnet50、resnet101等网络结构的代码，在源代码基础上进行修改，使其能够以resnet18的结构进行训练。具体文件的作用请移步[脚本说明](脚本说明)。

值得一提的是，resnet常用的结构包括resnet18、34、50、101、152等，modelzoo源代码给出了50和101，本实验扩写了18和152（由于只下载了cifar10数据集，其种类较少，因此仅对18进行了运行测试。对于更深层的网络建议使用ImageNet2012数据集运行。）

# 运行

通过sh的运行在参考资料1中已有详细说明，此处仅对本实验的运行做出说明：

1、建立obs桶，将项目文件和数据上传到obs桶中的一个文件夹。

2、在modelarts中建立一个notebook环境，使用ascend处理器，关联上面提到的文件夹。

3、打开notebook对应的jupyterlab，在左侧文件管理中选中所有文件进行syncobs，这样代码就可以访问到相对应的文件了。

4、打开train.py，更改其args设定的默认参数（此处为了方便在notebook中运行做了一定更改，以默认值传参）。

5、点击运行，即可开始训练

# 脚本说明

## 脚本及样例代码

```shell
.
└──resnet
  ├── README.md
  ├── scripts
    ├── run_distribute_train.sh            # 启动Ascend分布式训练（8卡）
    ├── run_parameter_server_train.sh      # 启动Ascend参数服务器训练(8卡)
    ├── run_eval.sh                        # 启动Ascend评估
    ├── run_standalone_train.sh            # 启动Ascend单机训练（单卡）
    ├── run_distribute_train_gpu.sh        # 启动GPU分布式训练（8卡）
    ├── run_parameter_server_train_gpu.sh  # 启动GPU参数服务器训练（8卡）
    ├── run_eval_gpu.sh                    # 启动GPU评估
    └── run_standalone_train_gpu.sh        # 启动GPU单机训练（单卡）
  ├── src
    ├── config.py                          # 参数配置
    ├── dataset.py                         # 数据预处理
    ├── CrossEntropySmooth.py             # ImageNet2012数据集的损失定义
    ├── lr_generator.py                    # 生成每个步骤的学习率
    └── resnet.py                          # ResNet骨干网络，包括ResNet50、ResNet101和SE-ResNet50
  ├── data           
    ├── cifar-10-batches-bin               #训练集数据
    └── cifar-10-verify-bin                #测试数据
  ├── eval.py                              # 评估网络
  ├── gpu_resnet_benchmark.py              #网络在gpu运行的benchmark
  ├── mindspore_hub_conf.py                #mindspore的配置
  ├── export.py                            #数据导入配置
  ├── train.ipynb                          #在motebook中使用的train.py代码
  ├── eval.ipynb                           #在notebook中使用的
  └── train.py                             # 训练网络
  
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置ResNet50和CIFAR-10数据集。

```text
"class_num":10,                  # 数据集类数
"batch_size":32,                 # 输入张量的批次大小
"loss_scale":1024,               # 损失等级
"momentum":0.9,                  # 动量
"weight_decay":1e-4,             # 权重衰减
"epoch_size":90,                 # 此值仅适用于训练；应用于推理时固定为1
"pretrain_epoch_size":0,         # 加载预训练检查点之前已经训练好的模型的周期大小；实际训练周期大小等于epoch_size减去pretrain_epoch_size
"save_checkpoint":True,          # 是否保存检查点
"save_checkpoint_epochs":5,      # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一步完成后保存
"keep_checkpoint_max":10,        # 只保留最后一个keep_checkpoint_max检查点
"save_checkpoint_path":"./",     # 检查点保存路径
"warmup_epochs":5,               # 热身周期数
"lr_decay_mode":"poly”           # 衰减模式可为步骤、策略和默认
"lr_init":0.01,                  # 初始学习率
"lr_end":0.0001,                  # 最终学习率
"lr_max":0.1,                    # 最大学习率
```

- 配置ResNet50和ImageNet2012数据集。

```text
"class_num":1001,                # 数据集类数
"batch_size":256,                # 输入张量的批次大小
"loss_scale":1024,               # 损失等级
"momentum":0.9,                  # 动量优化器
"weight_decay":1e-4,             # 权重衰减
"epoch_size":90,                 # 此值仅适用于训练；应用于推理时固定为1
"pretrain_epoch_size":0,         # 加载预训练检查点之前已经训练好的模型的周期大小；实际训练周期大小等于epoch_size减去pretrain_epoch_size
"save_checkpoint":True,          # 是否保存检查点
"save_checkpoint_epochs":5,      # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max":10,        # 只保存最后一个keep_checkpoint_max检查点
"save_checkpoint_path":"./",     # 检查点相对于执行路径的保存路径
"warmup_epochs":0,               # 热身周期数
"lr_decay_mode":"Linear",        # 用于生成学习率的衰减模式
"use_label_smooth":True,         # 标签平滑
"label_smooth_factor":0.1,       # 标签平滑因子
"lr_init":0,                     # 初始学习率
"lr_max":0.8,                    # 最大学习率
"lr_end":0.0,                    # 最小学习率
```

- 配置ResNet101和ImageNet2012数据集。

```text
"class_num":1001,                # 数据集类数
"batch_size":32,                 # 输入张量的批次大小
"loss_scale":1024,               # 损失等级
"momentum":0.9,                  # 动量优化器
"weight_decay":1e-4,             # 权重衰减
"epoch_size":120,                # 训练周期大小
"pretrain_epoch_size":0,         # 加载预训练检查点之前已经训练好的模型的周期大小；实际训练周期大小等于epoch_size减去pretrain_epoch_size
"save_checkpoint":True,          # 是否保存检查点
"save_checkpoint_epochs":5,      # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max":10,        # 只保存最后一个keep_checkpoint_max检查点
"save_checkpoint_path":"./",     # 检查点相对于执行路径的保存路径
"warmup_epochs":0,               # 热身周期数
"lr_decay_mode":"cosine”         # 用于生成学习率的衰减模式
"use_label_smooth":True,         # 标签平滑
"label_smooth_factor":0.1,       # 标签平滑因子
"lr":0.1                         # 基础学习率
```

- 配置SE-ResNet50和ImageNet2012数据集。

```text
"class_num":1001,                # 数据集类数
"batch_size":32,                 # 输入张量的批次大小
"loss_scale":1024,               # 损失等级
"momentum":0.9,                  # 动量优化器
"weight_decay":1e-4,             # 权重衰减
"epoch_size":28,                 # 创建学习率的周期大小
"train_epoch_size":24            # 实际训练周期大小
"pretrain_epoch_size":0,         # 加载预训练检查点之前已经训练好的模型的周期大小；实际训练周期大小等于epoch_size减去pretrain_epoch_size
"save_checkpoint":True,          # 是否保存检查点
"save_checkpoint_epochs":4,      # 两个检查点之间的周期间隔；默认情况下，最后一个检查点将在最后一个周期完成后保存
"keep_checkpoint_max":10,        # 只保存最后一个keep_checkpoint_max检查点
"save_checkpoint_path":"./",     # checkpoint相对于执行路径的保存路径
"warmup_epochs":3,               # 热身周期数
"lr_decay_mode":"cosine”         # 用于生成学习率的衰减模式
"use_label_smooth":True,         # 标签平滑
"label_smooth_factor":0.1,       # 标签平滑因子
"lr_init":0.0,                   # 初始学习率
"lr_max":0.3,                    # 最大学习率
"lr_end":0.0001,                 # 最终学习率
```



# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
