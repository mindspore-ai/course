# 构建语义分割网络模型应用
 
## 实验介绍
本实验主要介绍使用MindSpore深度学习框架在PASCAL VOC 2012数据集上训练deeplabv3网络模型。本实验参考MindSpore开源仓库model_zoo中的[deeplabv3 Example](https://gitee.com/mindspore/mindspore/tree/r0.5/model_zoo/deeplabv3) 模型案例。

## deeplabv3简要介绍
deeplabv1和deeplabv2，即带孔卷积(atrous convolution), 能够明确地调整filters的感受野，并决定DNN计算得到特征的分辨率。
deeplabv3中提出 Atrous Spatial Pyramid Pooling(ASPP)模块, 挖掘不同尺度的卷积特征，以及编码了全局内容信息的图像层特征，提升分割效果。
详细介绍参考论文：http://arxiv.org/abs/1706.05587 。

## 实验目的
* 了解如何使用MindSpore加载常用的PASCAL VOC 2012数据集。
* 了解MindSpore的model_zoo模块，以及如何使用model_zoo中的模型。
* 了解deeplabv3这类语义分割模型的基本结构和编程方法。

## 预备知识
* 熟练使用Python，了解Shell及Linux操作系统基本知识。
* 具备一定的深度学习理论知识，如Encoder、Decoder、损失函数、优化器，训练策略、Checkpoint等。
* 了解华为云的基本使用方法，包括[OBS（对象存储）](https://www.huaweicloud.com/product/obs.html) 、[ModelArts（AI开发平台](https://www.huaweicloud.com/product/modelarts.html) 、[训练作业](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0046.html) 等功能。华为云官网：https://www.huaweicloud.com。
* 了解并熟悉MindSpore AI计算框架，MindSpore官网：https://www.mindspore.cn/。

## 实验环境
* MindSpore 0.5.0（MindSpore版本会定期更新，本指导也会定期刷新，与版本配套）。
* 华为云ModelArts：ModelArts是华为云提供的面向开发者的一站式AI开发平台，集成了昇腾AI处理器资源池，用户可以在该平台下体验MindSpore。ModelArts官网：https://www.huaweicloud.com/product/modelarts.html。

## 实验准备
### 创建OBS桶
本实验需要使用华为云OBS存储脚本和数据集，可以参考[快速通过OBS控制台上传下载文件](https://support.huaweicloud.com/qs-obs/obs_qs_0001.html) 了解使用OBS创建桶、上传文件、下载文件的使用方法。当数据集大时，可以使用[OBS Browser+](https://support.huaweicloud.com/browsertg-obs/obs_03_1000.html) 。

> 提示： 华为云新用户使用OBS时通常需要创建和配置“访问密钥”，可以在使用OBS时根据提示完成创建和配置。也可以[参考获取访问密钥并完成ModelArts全局配置](https://support.huaweicloud.com/prepare-modelarts/modelarts_08_0002.html) 获取并配置访问密钥。

创建OBS桶的参考配置如下：

* 区域：华北-北京四
* 数据冗余存储策略：单AZ存储
* 桶名称：如ms-course
* 存储类别：标准存储
* 桶策略：公共读
* 归档数据直读：关闭
* 企业项目、标签等配置：免

## 数据集准备
[Pascal VOC2012数据集](https://blog.csdn.net/haoji007/article/details/80361587) 主要是针对视觉任务中监督学习提供标签数据，它有二十个类别。主要有四个大类别，分别是人、常见动物、交通车辆、室内家具用品。这里只说与图像分割（segmentation）有关的信息,本用例使用已去除分割标注的颜色，仅保留了分割任务的数据集。VOC2012[官网地址](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html) ，[官方下载地址](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) 。

本实验指导的数据集可通过如下方式获取：
* 方式一：针对教学使用的[实验指导](https://gitee.com/mindspore/course)和 [模型案例](https://gitee.com/mindspore/mindspore/tree/r0.5/model_zoo) ，为了节省下载和处理数据集的时间，我们提前准备好了数据集,可直接通过上述的[华为云OBS](https://share-course.obs.cn-north-4.myhuaweicloud.com/dataset/voc2012.zip) （已去除分割标注的颜色，仅保留了分割任务的数据）获取。
* 方式二：使用moxing接口拷贝数据集，即在ModelArts上使用moxing的拷贝功能直接拷贝共享的数据集到执行容器中：
    ```
    import moxing
    # set moxing/obs auth info, ak:Access Key Id, sk:Secret Access Key, server:endpoint of obs bucket
    moxing.file.set_auth(ak='VCT2GKI3GJOZBQYJG5WM', sk='t1y8M4Z6bHLSAEGK2bCeRYMjo2S2u0QBqToYbxzB', server="obs.cn-north-4.myhuaweicloud.com")
    
    # copy dataset from obs to container/cache
    moxing.file.copy_parallel(src_url="s3://share-course/dataset/voc2012/", dst_url='/cache/data_path')
    ```
    
另外，本实验采用fine-tune的训练方式，为了节省训练时间，我们提前准备好了预训练的[checkpoint文件](https://share-course.obs.myhuaweicloud.com/checkpoint/deeplabv3/deeplabv3_train_14-1_1.ckpt) ,方便直接获取使用。

## 脚本准备
从MindSpore开源仓库model_zoo中下载[deeplabv3模型案例](https://gitee.com/mindspore/mindspore/tree/r0.5/model_zoo/deeplabv3) 。从[课程gitee仓库](https://gitee.com/mindspore/course) 中下载相关执行脚本。

## 上传文件
将脚本和数据集上传到OBS桶中，可参考如下组织形式：
```
deeplabv3_example
├── voc2012 # 数据集
├── checkpoint # ckpt文件存放路径
└── deeplabv3  # 执行脚本存放路径
    ├── src # 包括数据集处理、网络定义等
    └── main.py # 执行脚本，包括训练和推理过程
```

## 实验步骤
### 代码梳理
`main.py`：执行脚本，包含训练和推理过程。主要包括创建数据集、网络定义、网络模型fine_tune等函数。

#### 创建数据集:
```python
def create_dataset(args, data_url, epoch_num=1, batch_size=1, usage="train", shuffle=True):
   """
   Create Dataset for deeplabv3.

   Args:
       args (dict): Train parameters.
       data_url (str): Dataset path.
       epoch_num (int): Epoch of dataset (default=1).
       batch_size (int): Batch size of dataset (default=1).
       usage (str): Whether is use to train or eval (default='train').

   Returns:
       Dataset.
   """
   # create iter dataset
   dataset = HwVocRawDataset(data_url, usage=usage)
   dataset_len = len(dataset)
 
   # wrapped with GeneratorDataset
   dataset = de.GeneratorDataset(dataset, ["image", "label"], sampler=None)
   dataset.set_dataset_size(dataset_len)
   dataset = dataset.map(input_columns=["image", "label"], operations=DataTransform(args, usage=usage))

   channelswap_op = C.HWC2CHW()
   dataset = dataset.map(input_columns="image", operations=channelswap_op)

   # 1464 samples / batch_size 8 = 183 batches
   # epoch_num is num of steps
   # 3658 steps / 183 = 20 epochs
   if usage == "train" and shuffle:
       dataset = dataset.shuffle(1464)
   dataset = dataset.batch(batch_size, drop_remainder=(usage == "train"))
   dataset = dataset.repeat(count=epoch_num)
   dataset.map_model = 4

   return dataset 
``` 

#### 定义deeplabv3网络模型：
```python
def deeplabv3_resnet50(num_classes, feature_shape, image_pyramid,
                       infer_scale_sizes, atrous_rates=None, decoder_output_stride=None,
                       output_stride=16, fine_tune_batch_norm=False):
   """
   ResNet50 based deeplabv3 network.

   Args:
       num_classes (int): Class number.
       feature_shape (list): Input image shape, [N,C,H,W].
       image_pyramid (list): Input scales for multi-scale feature extraction.
       atrous_rates (list): Atrous rates for atrous spatial pyramid pooling.
       infer_scale_sizes (list): 'The scales to resize images for inference.
       decoder_output_stride (int): 'The ratio of input to output spatial resolution'
       output_stride (int): 'The ratio of input to output spatial resolution.'
       fine_tune_batch_norm (bool): 'Fine tune the batch norm parameters or not'

   Returns:
       Cell, cell instance of ResNet50 based deeplabv3 neural network.

   Examples:
       >>> deeplabv3_resnet50(100, [1,3,224,224],[1.0],[1.0])
   """
   return deeplabv3(num_classes=num_classes,
                    feature_shape=feature_shape,
                    backbone=resnet50_dl(fine_tune_batch_norm),
                    channel=2048,
                    depth=256,
                    infer_scale_sizes=infer_scale_sizes,
                    atrous_rates=atrous_rates,
                    decoder_output_stride=decoder_output_stride,
                    output_stride=output_stride,
                    fine_tune_batch_norm=fine_tune_batch_norm,
                    image_pyramid=image_pyramid)
```
#### 模型训练过程
定义LossCallBack类，用于监测模型训练过程的loss值：
```python
class LossCallBack(Callback):
    """
    Monitor the loss in training.
    Note:
        if per_print_times is 0 do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """
    def __init__(self, per_print_times=1):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0")
        self._per_print_times = per_print_times
    def step_end(self, run_context):
        cb_params = run_context.original_args()
        print("epoch: {}, step: {}, outputs are {}".format(cb_params.cur_epoch_num, cb_params.cur_step_num,
                                                           str(cb_params.net_outputs)))

```

定义model_fine_tune函数，用于对网络模型进行微调：
```python
  def model_fine_tune(flags, train_net, fix_weight_layer):
      path = flags.checkpoint_url
      if path is None:
         return
      path = checkpoint_path
      param_dict = load_checkpoint(path)
      load_param_into_net(train_net, param_dict)
      for para in train_net.trainable_params():
          if fix_weight_layer in para.name:
              para.requires_grad = False
```

网络模型的完整训练过程：
```python
    train_dataset = create_dataset(args_opt, data_path, config.epoch_size, config.batch_size, usage="train")
    dataset_size = train_dataset.get_dataset_size()
    time_cb = TimeMonitor(data_size=dataset_size)
    callback = [time_cb, LossCallBack()]
    if config.enable_save_ckpt:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                                     keep_checkpoint_max=config.save_checkpoint_num)
        ckpoint_cb = ModelCheckpoint(prefix='checkpoint_deeplabv3', config=config_ck)
        callback.append(ckpoint_cb)
    net = deeplabv3_resnet50(config.seg_num_classes, [config.batch_size, 3, args_opt.crop_size, args_opt.crop_size],
                             infer_scale_sizes=config.eval_scales, atrous_rates=config.atrous_rates,
                             decoder_output_stride=config.decoder_output_stride, output_stride=config.output_stride,
                             fine_tune_batch_norm=config.fine_tune_batch_norm, image_pyramid=config.image_pyramid)
    net.set_train()
    model_fine_tune(args_opt, net, 'layer')
    loss = OhemLoss(config.seg_num_classes, config.ignore_label)
    opt = Momentum(filter(lambda x: 'beta' not in x.name and 'gamma' not in x.name and 'depth' not in x.name and 'bias' not in x.name, net.trainable_params()), learning_rate=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    model = Model(net, loss, opt)
    model.train(config.epoch_size, train_dataset, callback)

```
>提示：训练过程中，可通过修改上述示例代码路径下的deeplabv3_example/deeplabv3/src/config.py文件的相关参数来提升训练精度，本实验指导采用默认配置。

训练结果示例：
```
epoch: 1, step: 732, outputs are 0.64453894
Epoch time: 91362.341, per step time: 124.812
epoch: 2, step: 1464, outputs are 0.13636473
Epoch time: 25760.597, per step time: 35.192
epoch: 3, step: 2196, outputs are 0.11666249
Epoch time: 25503.751, per step time: 34.841
epoch: 4, step: 2928, outputs are 0.33679807
Epoch time: 25438.145, per step time: 34.752
epoch: 5, step: 3660, outputs are 0.7013806
Epoch time: 25304.372, per step time: 34.569
epoch: 6, step: 4392, outputs are 0.9661154
Epoch time: 25466.854, per step time: 34.791
```

#### 推理过程
定义mIou指标进行推理性能评估：
```python
class MiouPrecision(Metric):
    """Calculate miou precision."""
    def __init__(self, num_class=21):
        super(MiouPrecision, self).__init__()
        if not isinstance(num_class, int):
            raise TypeError('num_class should be integer type, but got {}'.format(type(num_class)))
        if num_class < 1:
            raise ValueError('num_class must be at least 1, but got {}'.format(num_class))
        self._num_class = num_class
        self._mIoU = []
        self.clear()

    def clear(self):
        self._hist = np.zeros((self._num_class, self._num_class))
        self._mIoU = []

    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError('Need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))
        predict_in = self._convert_data(inputs[0])
        label_in = self._convert_data(inputs[1])
        if predict_in.shape[1] != self._num_class:
            raise ValueError('Class number not match, last input data contain {} classes, but current data contain {} '
                             'classes'.format(self._num_class, predict_in.shape[1]))
        pred = np.argmax(predict_in, axis=1)
        label = label_in
        if len(label.flatten()) != len(pred.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}'.format(len(label.flatten()), len(pred.flatten())))
            raise ValueError('Class number not match, last input data contain {} classes, but current data contain {} '
                             'classes'.format(self._num_class, predict_in.shape[1]))
        self._hist = confuse_matrix(label.flatten(), pred.flatten(), self._num_class)
        mIoUs = iou(self._hist)
        self._mIoU.append(mIoUs)

    def eval(self):
        """
        Computes the mIoU categorical accuracy.
        """
        mIoU = np.nanmean(self._mIoU)
        print('mIoU = {}'.format(mIoU))
        return mIoU

```

模型完整推理过程：
```python
eval_dataset = create_dataset(args_opt, data_path, config.epoch_size, config.batch_size, usage="eval")
    net = deeplabv3_resnet50(config.seg_num_classes, [config.batch_size, 3, args_opt.crop_size, args_opt.crop_size],
                             infer_scale_sizes=config.eval_scales, atrous_rates=config.atrous_rates,
                             decoder_output_stride=config.decoder_output_stride, output_stride=config.output_stride,
                             fine_tune_batch_norm=config.fine_tune_batch_norm, image_pyramid=config.image_pyramid)

    param_dict = load_checkpoint(eval_checkpoint_path)
    load_param_into_net(net, param_dict)
    mIou = MiouPrecision(config.seg_num_classes)
    metrics = {'mIou': mIou}
    loss = OhemLoss(config.seg_num_classes, config.ignore_label)
    model = Model(net, loss, metrics=metrics)
    model.eval(eval_dataset)
```
>提示：将上述训练完的checkpoint文件进行加载推理，本实验采用训练完的最后一个checkpoint文件，即checkpoint_deeplabv3-6_732.ckpt。

推理结果示例：
```
mIoU = 0.6148479926928656
```

由于ModelArts创建训练作业时，运行参数会通过脚本传参的方式输入给脚本代码，脚本必须解析传参才能在代码中使用相应参数。如data_url和train_url，分别对应数据存储路径(OBS路径)和训练输出路径(OBS路径)。脚本需对传参进行解析后赋值到args_opt变量里，在后续代码里可以使用。
```python
parser = argparse.ArgumentParser(description="deeplabv3 training")
parser.add_argument("--distribute", type=str, default="false", help="Run distribute, default is false.")
parser.add_argument('--data_url', required=True, default=None, help='Train data url')
parser.add_argument('--train_url', required=True, default=None, help='Train data output url')
parser.add_argument('--checkpoint_url', default=None, help='Checkpoint path')
args_opt = parser.parse_args()

```

MindSpore暂时没有提供直接访问OBS数据的接口，需要通过MoXing提供的API与OBS交互。将OBS中存储的数据拷贝至执行容器,可参考本实验：
```python
import moxing as mox
mox.file.copy_parallel(src_url=args_opt.data_url, dst_url='voc2012/')
mox.file.copy_parallel(src_url=args_opt.checkpoint_url, dst_url='checkpoint/')
```
模型训练使用的是拷贝至当前执行容器路径下的相应文件：
```python
data_path = "./voc2012"
train_checkpoint_path = "./checkpoint/deeplabv3_train_14-1_1.ckpt" #预训练的ckpt
```

>提示：如若需将训练输出（如模型Checkpoint文件）从执行容器拷贝至OBS，请参考：
>```python
>import moxing
># dst_url形如's3://OBS/PATH'，将ckpt目录拷贝至OBS后，可在OBS的`args_opt.train_url`目录下看到ckpt目录
>moxing.file.copy_parallel(src_url='ckpt', dst_url=os.path.join(args_opt.train_url, 'ckpt'))
>```

## 创建训练作业
可以参考[使用常用框架训练模型](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0238.html)来创建并启动训练作业。

创建训练作业的参考配置：

* 算法来源：常用框架->Ascend-Powered-Engine->MindSpore
* 代码目录：如选择上述新建的OBS桶中的deeplabv3_example/deeplabv3/
* 启动文件：如选择上述新建的OBS桶中的deeplabv3_example/deeplabv3/下的main.py
* 数据来源：数据存储位置->选择上述新建的OBS桶中的deeplabv3_example/的voc2012目录
* 训练输出位置：选择上述新建的OBS桶中的deeplabv3_example/目录，并在其中创建output目录
* 运行参数：点击增加运行参数，分别输入checkpoint_url参数和对应具体路径值的参数，如本实验输入为s3://ms-course(桶名称)/deeplabv3_example/checkpoint/。
* 作业日志路径：选择上述新建的OBS桶中的deeplabv3_example/目录，并在其中创建log目录
* 规格：Ascend:1*Ascend 910
* 其他均为默认

点击提交以开始训练，查看训练过程：
1. 在训练作业列表里可以看到刚创建的训练作业，在训练作业页面可以看到版本管理。
2. 点击运行中的训练作业，在展开的窗口中可以查看作业配置信息，以及训练过程中的日志，日志会不断刷新，等训练作业完成后也可以下载日志到本地进行查看。

> 提示：ModelArts提供了[PyCharm ToolKit工具](https://support.huaweicloud.com/tg-modelarts/modelarts_15_0003.html) ，方便基于MindSpore框架的脚本开发和调试；
> 在使用PyCharm ToolKit工具进行传参训练时，注意参数key-value的书写格式，如本实验设置：checkpoint_url=s3://ms-course(桶名称)/deeplabv3_example/checkpoint/ 。
> 或者可用ModelArts下的开发环境[Notebook](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0034.html) 进行基于MindSpore框架的脚本开发和调试。

## 实验结论
本实验主要介绍使用MindSpore在voc2012数据集上训练和推理deeplabv3网络模型，了解以下知识点：
* 加载VOC2012数据集并进行相关数据增强等预处理操作；
* 了解deeplabv3网络模型结构及其在MindSpore框架下的实现；
* 使用fine-tune功能对模型进行微调；
* 使用自定义Callback实现性能监测；
* 使用自定义的Miou指标进行模型推理性能评估。











