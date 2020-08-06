# Graph_Convolutional_Network

## Graph_Convolutional_Network原理

### 图卷积神经网络由来

**为什么传统的卷积神经网络不能直接运用到图上？还需要设计专门的图卷积网络？**

简单来说，卷积神经网络的研究的对象是限制在Euclidean domains的数据。Euclidean data最显著的特征就是有规则的空间结构，比如图片是规则的正方形栅格，或者语音是规则的一维序列。而这些数据结构能够用一维、二维的矩阵表示，卷积神经网络处理起来很高效。但是，我们的现实生活中有很多数据并不具备规则的空间结构，称为Non Euclidean data。比如推荐系统、电子交易、计算几何、脑信号、分子结构等抽象出的图谱。这些图谱结构每个节点连接都不尽相同，有的节点有三个连接，有的节点有两个连接，是不规则的数据结构。

如下图所示左图是欧式空间数据，右图为非欧式空间数据。其中绿色节点为卷积核。

> **观察可以发现**
>
> 1. 在图像为代表的欧式空间中，结点的邻居数量都是固定的。比如说绿色结点的邻居始终是8个。在图这种非欧空间中，结点有多少邻居并不固定。目前绿色结点的邻居结点有2个，但其他结点也会有5个邻居的情况。
> 2. 欧式空间中的卷积操作实际上是用固定大小可学习的卷积核来抽取像素的特征。对于非欧式空间因为邻居结点不固定，所以传统的卷积核不能直接用于抽取图上结点的特征。

![GCN](images/gcn1.png)

为了解决传统卷积不能直接应用在邻居结点数量不固定的非欧式空间数据上的问题。

1. 提出一种方式把非欧空间的图转换成欧式空间。
2. 找出一种可处理变长邻居结点的卷积核在图上抽取特征。（图卷积属于这一种，后边又将其分为基于空间域和基于频域）

### 图卷积神经网络概述

GCN的本质目的就是用来提取拓扑图的空间特征。 图卷积神经网络主要有两类，一类是基于空间域（spatial domain）或顶点域（vertex domain）的，另一类则是基于频域或谱域（spectral domain）的。GCN属于频域图卷积神经网络。

基于空域卷积的方法直接将卷积操作定义在每个结点的连接关系上，它跟传统的卷积神经网络中的卷积更相似一些。在这个类别中比较有代表性的方法有 Message Passing Neural Networks(MPNN), GraphSage, Diffusion Convolution Neural Networks(DCNN), PATCHY-SAN等。

频域方法希望借助图谱的理论来实现拓扑图上的卷积操作。从整个研究的时间进程来看：首先研究GSP（graph signal processing）的学者定义了graph上的傅里叶变化（Fourier Transformation），进而定义了graph上的卷积，最后与深度学习结合提出了Graph Convolutional Network（GCN）。

### GCN网络过程

过程：

1. 定义graph上的Fourier Transformation傅里叶变换
2. 定义graph上的convolution卷积

#### 图上的傅里叶变换

传统的傅里叶变换：
$$
F(w) = \int f(t)e^{-jwt}dt
$$

其中$ e^{-jwt} $为基函数、特征向量。

注：特征向量定义：特征方程$ AV= \lambda V $中V为特征向量，$\lambda $为特征值

由于图节点为离散的，我们将上面傅里叶变换离散化，使用有限项分量来近似F(w) ，得到图上的傅里叶变换：

​                                                              $ F(\lambda _l) = \hat {f(\lambda _l)} = \sum_{i=1}^{N}{f(i)}u_l(i)i $

其中$u_l(i)$为基函数、特征向量，$\lambda _l$为$u_l(i)$对应的特征值，$ f(i）$为特i征值$\lambda _l$下的$f$ 。即：特征值$\lambda _l$下的$f$ 的傅里叶变换是该分量与$\lambda _l$对应的特征向量$u_l$进行内积运算。这里用拉普拉斯向量$u_l$替换傅里叶变换的基$ e^{-jwt} $，后边我们将定义拉普拉斯矩阵。

利用矩阵乘法将Graph上的傅里叶变换推广到矩阵形式：


$$
\left[ \begin{matrix} \hat {f(\lambda _1)} \\ \hat {f(\lambda _2)} \\ ...  \\  \hat {f(\lambda _N)} \end{matrix} \right] = 
\left[ \begin{matrix} u_1(1) & u_1(2) & ... & u_1(N) \\ u_2(1) & u_2(2) & ... & u_2(N) \\ ... & ... & ... & ...  \\  u_N(1) & u_N(2) & ... & u_N(N) \end{matrix} \right]
\left[ \begin{matrix} {f(1)} \\  {f(2)} \\ ...  \\  {f(N)} \end{matrix} \right]
$$

即 f 在Graph上傅里叶变换的矩阵形式为：$\hat f = U^Tf$

 f 在Graph上傅里叶逆变换的矩阵形式为：$f = U\hat f$

其中$U$为f的特征矩阵，拉普拉斯矩阵的正交基（后边我们会提到拉普拉斯矩阵）。它是一个对称正交矩阵，故$U^T=U^{-1}$

#### 图卷积

传统卷积:

卷积定理：函数$f(t)$、$h(t)$（h为卷积核）的卷积是其傅里叶变换的逆运算，所以传统卷积可以定义为：
$$
f*h = F^{-1}[\hat {f(w)}\hat {h(w)}] = \frac{1}{2\pi}\int \hat {f(w)}\hat {h(w)}e^jwtdw
$$
图卷积:

根据卷积定理和前面得到的Graph上的傅里叶逆变换可以得到图卷积为：
$$
(f*h)_G = F^{-1}[U^Tf F(h)]=UF(h)U^Tf
$$
其中h为卷积核，F(h)为h的傅里叶变换。

将F(h）写成对角矩阵形式得到：
$$
H = F(h)=\left[ \begin{matrix} \hat {h(\lambda _1)} & & \\ & ... & \\ & & \hat {h(\lambda _N)} \end{matrix} \right]
$$

图卷积可以写成如下形式：

$$
(f*h)_G = U\left[ \begin{matrix} \hat {h(\lambda _1)} & & \\ & ... & \\ & & \hat {h(\lambda _N)} \end{matrix} \right]U^Tf = Lf
$$

其中L为拉普拉斯矩阵。U为L的正交化矩阵。上面式子也是GCN算法计算公式，通俗的解释，我们可以把GCN看成是基于拉普拉斯矩阵的特征分解。

### 拉普拉斯矩阵

GCN的核心基于拉普拉斯矩阵的谱分解（特征分解）。所以GCN算法的关键在于定义拉普拉斯矩阵。

本文的拉普拉斯矩阵定义如下：
$$
L = \widetilde U^{-\frac{1}{2}}\widetilde H\widetilde U^{-\frac{1}{2}}
$$
其中

$$
\widetilde U^{-\frac{1}{2}}\widetilde H\widetilde U^{-\frac{1}{2}} = I + U^{-\frac{1}{2}}HU^{-\frac{1}{2}}
$$

$$
\widetilde H= H + I \\
\widetilde U_{ii} = \sum_{j}\widetilde H_{ij}
$$

U取图的度矩阵（如下图中的degree matrix）,H取图的邻接矩阵（如下图中的adjacency matrix）

![GCN](images/gcn2.png)

[Graph_Convolutional_Network原理引用于论文]: https://arxiv.org/pdf/1609.02907.pdf

## 实验介绍

图卷积网络（Graph Convolutional Network，GCN）是近年来逐渐流行的一种神经网络结构。不同于只能用于网格结构（grid-based）数据的传统网络模型 LSTM 和 CNN，图卷积网络能够处理具有广义拓扑图结构的数据，并深入发掘其特征和规律。

本实验主要介绍在下载的Cora和Citeseer数据集上使用MindSpore进行图卷积网络的训练。

## 实验目的

- 了解GCN相关知识；
- 在MindSpore中使用Cora和Citeseer数据集训练GCN示例。

## 预备知识

- 熟练使用Python，了解Shell及Linux操作系统基本知识。
- 具备一定的深度学习理论知识，如前馈神经网络、卷积神经网络、图卷积网络等。
- 了解并熟悉MindSpore AI计算框架，MindSpore官网：[https://www.mindspore.cn](https://www.mindspore.cn/)

## 实验环境

- MindSpore 0.5.0（MindSpore版本会定期更新，本指导也会定期刷新，与版本配套）；
- 华为云ModelArts：ModelArts是华为云提供的面向开发者的一站式AI开发平台，集成了昇腾AI处理器资源池，用户可以在该平台下体验MindSpore。ModelArts官网：https://www.huaweicloud.com/product/modelarts.html

## 实验准备

### 数据集准备

从[github](https://github.com/kimiyoung/planetoid)下载/kimiyoung/planetoid提供的数据集Cora或Citeseer。这是Planetoid的一种实现，以下论文提出了一种基于图的半监督学习方法：[图嵌入技术在半监督学习中的应用](https://arxiv.org/abs/1603.08861)

将数据集放置到所需的路径下，该文件夹应包含以下文件：

```
└─data 
    ├─ind.cora.allx 
    ├─ind.cora.ally 
    ├─...
    ├─ind.cora.test.index 
    ├─trans.citeseer.tx
    ├─trans.citeseer.ty
    ├─...
    └─trans.pubmed.y
```

其中模型的输入包含：

- `x`，标记训练实例的特征向量，
- `y`，即已标记的训练实例的热门标签，
- `allx`，标记的和未标记的训练实例（的超集`x`）的特征向量，
- `graph`，`dict`格式为`{index: [index_of_neighbor_nodes]}.`

令n为标记和未标记训练实例的数量。这n个实例的索引应从0到n-1 `graph`，其顺序与中的顺序相同`allx`。

除了`x`，`y`，`allx`，和`graph`如上所述，预处理的数据集还包括：

- `tx`，测试实例的特征向量，
- `ty`，测试实例的热门标签，
- `test.index`，`graph`对于归纳设置，中的测试实例的索引，
- `ally`，是中实例的标签`allx`。

`graph`转换设置中测试实例的索引是从`#x`到`#x + #tx - 1`，与中的顺序相同`tx`。

### 脚本准备

从[MindSpore model_zoo](https://gitee.com/mindspore/mindspore/tree/r0.5/model_zoo/gcn)中下载GCN代码；[课程gitee仓库](https://gitee.com/mindspore/course)上下载本实验相关脚本。

### 上传文件

将脚本和数据集放到到experiment文件夹中，组织为如下形式：

```
experiment
├── data
├── graph_to_mindrecord 
│   ├── citeseer 
│   ├── cora
│   ├── graph_map_schema.py
│   ├── writer.py
│── src
│   ├── config.py 
│   ├── dataset.py 
│   ├── gcn.py 
│   ├── metrics.py
│── README.md
└── main.py
```

## 实验步骤

### 代码梳理

#### 数据处理

将Cora或Citeseer生成mindrecord格式的数据集。（可在cfg中设置`DATASET_NAME`为cora或者citeseer来转换不同的数据集）

```python
def run(cfg):
    args = read_args()
    #建立输出文件夹
    cur_path = os.getcwd()
    M_PATH = os.path.join(cur_path, cfg.MINDRECORD_PATH)
    if os.path.exists(M_PATH):
        shutil.rmtree(M_PATH)  # 删除文件夹
    os.mkdir(M_PATH)
    cfg.SRC_PATH = os.path.join(cur_path, cfg.SRC_PATH)
    #参数
    args.mindrecord_script= cfg.DATASET_NAME
    args.mindrecord_file=os.path.join(cfg.MINDRECORD_PATH,cfg.DATASET_NAME)
    args.mindrecord_partitions=cfg.mindrecord_partitions
    args.mindrecord_header_size_by_bit=cfg.mindrecord_header_size_by_bit
    args.mindrecord_page_size_by_bit=cfg.mindrecord_header_size_by_bit
    args.graph_api_args=cfg.SRC_PATH

    start_time = time.time()
    # pass mr_api arguments
    os.environ['graph_api_args'] = args.graph_api_args

    try:
        mr_api = import_module('graph_to_mindrecord.'+args.mindrecord_script + '.mr_api')
    except ModuleNotFoundError:
        raise RuntimeError("Unknown module path: {}".format(args.mindrecord_script + '.mr_api'))

    # init graph schema
    graph_map_schema = GraphMapSchema()

    num_features, feature_data_types, feature_shapes = mr_api.node_profile
    graph_map_schema.set_node_feature_profile(num_features, feature_data_types, feature_shapes)

    num_features, feature_data_types, feature_shapes = mr_api.edge_profile
    graph_map_schema.set_edge_feature_profile(num_features, feature_data_types, feature_shapes)

    graph_schema = graph_map_schema.get_schema()
```

### 参数配置

训练参数可以在config.py中设置。

```shell
"learning_rate": 0.01,            # Learning rate
"epochs": 200,                    # Epoch sizes for training
"hidden1": 16,                    # Hidden size for the first graph convolution layer
"dropout": 0.5,                   # Dropout ratio for the first graph convolution layer
"weight_decay": 5e-4,             # Weight decay for the parameter of the first graph convolution layer
"early_stopping": 10,             # Tolerance for early stopping
```

### 运行训练

```shell
def train(args_opt):
    """Train model."""
    np.random.seed(args_opt.seed)
    config = ConfigGCN()
    adj, feature, label = get_adj_features_labels(args_opt.data_dir)

    nodes_num = label.shape[0]
    train_mask = get_mask(nodes_num, 0, args_opt.train_nodes_num)
    eval_mask = get_mask(nodes_num, args_opt.train_nodes_num, args_opt.train_nodes_num + args_opt.eval_nodes_num)
    test_mask = get_mask(nodes_num, nodes_num - args_opt.test_nodes_num, nodes_num)

    class_num = label.shape[1]
    gcn_net = GCN(config, adj, feature, class_num)
    gcn_net.add_flags_recursive(fp16=True)

    eval_net = LossAccuracyWrapper(gcn_net, label, eval_mask, config.weight_decay)
    test_net = LossAccuracyWrapper(gcn_net, label, test_mask, config.weight_decay)
    train_net = TrainNetWrapper(gcn_net, label, train_mask, config)

    loss_list = []
    for epoch in range(config.epochs):
        t = time.time()

        train_net.set_train()
        train_result = train_net()
        train_loss = train_result[0].asnumpy()
        train_accuracy = train_result[1].asnumpy()

        eval_net.set_train(False)
        eval_result = eval_net()
        eval_loss = eval_result[0].asnumpy()
        eval_accuracy = eval_result[1].asnumpy()

        loss_list.append(eval_loss)
        if epoch%10==0:
            print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(train_loss),
                "train_acc=", "{:.5f}".format(train_accuracy), "val_loss=", "{:.5f}".format(eval_loss),
                "val_acc=", "{:.5f}".format(eval_accuracy), "time=", "{:.5f}".format(time.time() - t))

        if epoch > config.early_stopping and loss_list[-1] > np.mean(loss_list[-(config.early_stopping+1):-1]):
            print("Early stopping...")
            break

    t_test = time.time()
    test_net.set_train(False)
    test_result = test_net()
    test_loss = test_result[0].asnumpy()
    test_accuracy = test_result[1].asnumpy()
    print("Test set results:", "loss=", "{:.5f}".format(test_loss),
          "accuracy=", "{:.5f}".format(test_accuracy), "time=", "{:.5f}".format(time.time() - t_test))
          
if __name__ == '__main__':
    #------------------------定义变量------------------------------
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument('--data_url', type=str, default='./data', help='Dataset directory')
    args_opt = parser.parse_args()

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

    #转换数据格式
    print("============== Graph To Mindrecord ==============")
    run(cfg)
    #训练
    print("============== Starting Training ==============")
    train(cfg)       
```

MindSpore暂时没有提供直接访问OBS数据的接口，需要通过MoXing提供的API与OBS交互。将OBS中存储的数据拷贝至执行容器：

```python
import moxing as mox
mox.file.copy_parallel(args.data_url, dst_url='./data')
```

将训练模型Checkpoint从执行容器拷贝至OBS：

```python
mox.file.copy_parallel(src_url='data_mr', dst_url=cfg.MINDRECORD_PATH)
```

### 实验结果

训练结果将存储在脚本路径中，该路径的文件夹名称以“ train”开头。可以在日志中找到类似以下结果。

```shell
Epoch: 0000 train_loss= 1.95401 train_acc= 0.12143 val_loss= 1.94917 val_acc= 0.31400 time= 36.95478
Epoch: 0010 train_loss= 1.86495 train_acc= 0.85000 val_loss= 1.90644 val_acc= 0.50200 time= 0.00491
Epoch: 0020 train_loss= 1.75353 train_acc= 0.88571 val_loss= 1.86284 val_acc= 0.53000 time= 0.00525
Epoch: 0030 train_loss= 1.59934 train_acc= 0.87857 val_loss= 1.80850 val_acc= 0.55400 time= 0.00517
Epoch: 0040 train_loss= 1.45166 train_acc= 0.91429 val_loss= 1.74404 val_acc= 0.59400 time= 0.00502
Epoch: 0050 train_loss= 1.29577 train_acc= 0.94286 val_loss= 1.67278 val_acc= 0.67200 time= 0.00491
Epoch: 0060 train_loss= 1.13297 train_acc= 0.97857 val_loss= 1.59820 val_acc= 0.72800 time= 0.00482
Epoch: 0070 train_loss= 1.05231 train_acc= 0.95714 val_loss= 1.52455 val_acc= 0.74800 time= 0.00506
Epoch: 0080 train_loss= 0.97807 train_acc= 0.97143 val_loss= 1.45385 val_acc= 0.76800 time= 0.00519
Epoch: 0090 train_loss= 0.85581 train_acc= 0.97143 val_loss= 1.39556 val_acc= 0.77400 time= 0.00476
Epoch: 0100 train_loss= 0.81426 train_acc= 0.98571 val_loss= 1.34453 val_acc= 0.78400 time= 0.00479
Epoch: 0110 train_loss= 0.74759 train_acc= 0.97143 val_loss= 1.28945 val_acc= 0.78400 time= 0.00516
Epoch: 0120 train_loss= 0.70512 train_acc= 0.99286 val_loss= 1.24538 val_acc= 0.78600 time= 0.00517
Epoch: 0130 train_loss= 0.69883 train_acc= 0.98571 val_loss= 1.21186 val_acc= 0.78200 time= 0.00531
Epoch: 0140 train_loss= 0.66174 train_acc= 0.98571 val_loss= 1.19131 val_acc= 0.78400 time= 0.00481
Epoch: 0150 train_loss= 0.57727 train_acc= 0.98571 val_loss= 1.15812 val_acc= 0.78600 time= 0.00475
Epoch: 0160 train_loss= 0.59659 train_acc= 0.98571 val_loss= 1.13203 val_acc= 0.77800 time= 0.00553
Epoch: 0170 train_loss= 0.59405 train_acc= 0.97143 val_loss= 1.12650 val_acc= 0.78600 time= 0.00555
Epoch: 0180 train_loss= 0.55484 train_acc= 1.00000 val_loss= 1.09338 val_acc= 0.78000 time= 0.00542
Epoch: 0190 train_loss= 0.52347 train_acc= 0.99286 val_loss= 1.07537 val_acc= 0.78800 time= 0.00510
Test set results: loss= 1.01702 accuracy= 0.81400 time= 6.51215
```

运行main.py会在当前目录下生成一个关于Cora训练数据的动态图t-SNE_visualization_on_Cora.gif。

![](images/t-SNE_visualization_on_Cora.gif)

### 适配训练作业

创建训练作业时，运行参数会通过脚本传参的方式输入给脚本代码，脚本必须解析传参才能在代码中使用相应参数。如data_url和train_url，分别对应数据存储路径(OBS路径)和训练输出路径(OBS路径)。脚本对传参进行解析后赋值到`args`变量里，在后续代码里可以使用。

```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_url', required=False, default=None, help='Location of data.')
parser.add_argument('--train_url', required=False, default=None, help='Location of training outputs.')
args = parser.parse_args()
dataset = args.dataset
```

MindSpore暂时没有提供直接访问OBS数据的接口，需要通过MoXing提供的API与OBS交互。将OBS中存储的数据拷贝至执行容器（见`start.py`）：

```python
import moxing as mox
mox.file.copy_parallel(src_url=args.data_url, dst_url='data/')
```

### 创建训练作业

可以参考[使用常用框架训练模型](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0238.html)来创建并启动训练作业。

创建训练作业的参考配置：

- 算法来源：常用框架->Ascend-Powered-Engine->MindSpore
- 代码目录：选择上述新建的OBS桶中的experiment目录
- 启动文件：选择上述新建的OBS桶中的experiment目录下的`start.py`
- 数据来源：数据存储位置->选择上述新建的OBS桶中的experiment目录下的data目录
- 训练输出位置：选择上述新建的OBS桶中的experiment目录并在其中创建output目录
- 作业日志路径：同训练输出位置
- 规格：Ascend:1*Ascend 910
- 其他均为默认

启动并查看训练过程：

1. 点击提交以开始训练；
2. 在训练作业列表里可以看到刚创建的训练作业，在训练作业页面可以看到版本管理；
3. 点击运行中的训练作业，在展开的窗口中可以查看作业配置信息，以及训练过程中的日志，日志会不断刷新，等训练作业完成后也可以下载日志到本地进行查看；
4. 参考上述代码梳理，在日志中找到对应的打印信息，检查实验是否成功。

