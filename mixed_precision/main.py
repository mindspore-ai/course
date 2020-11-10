import os

import math
import time
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

import mindspore.nn as nn
import mindspore.dataset.engine as de
import mindspore.common.dtype as mstype
import mindspore.common.initializer as weight_init
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.vision.c_transforms as C

from mindspore import context
from mindspore.train.model import Model
from mindspore.nn.loss.loss import _Loss
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.tensor import Tensor
from mindspore.train.callback import Callback
from mindspore.nn.optim.momentum import Momentum
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor

context.set_context(mode=context.GRAPH_MODE, enable_auto_mixed_precision=False, device_target="Ascend")

train_path = "./datasets/cifar10/train"
ds = de.Cifar10Dataset(train_path, num_parallel_workers=8, shuffle=True)
print("the cifar dataset size is :", ds.get_dataset_size())
dict1 = ds.create_dict_iterator()
datas = dict1.get_next()
image = datas["image"].asnumpy()
print("the tensor of image is:", image.shape)
plt.imshow(np.array(image))
plt.show()


def create_dataset(dataset_path, do_train, repeat_num=10, batch_size=32):
    ds = de.Cifar10Dataset(dataset_path, num_parallel_workers=8, shuffle=True)

    # define map operations
    trans = []
    if do_train:
        trans += [
            C.RandomCrop((32, 32), (4, 4, 4, 4)),
            C.RandomHorizontalFlip(prob=0.5)
        ]

    trans += [
        C.Resize((224, 224)),
        C.Rescale(1.0 / 255.0, 0.0),
        C.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        C.HWC2CHW()
    ]

    type_cast_op = C2.TypeCast(mstype.int32)

    ds = ds.map(input_columns="label", num_parallel_workers=8, operations=type_cast_op)
    ds = ds.map(input_columns="image", num_parallel_workers=8, operations=trans)

    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)
    # apply dataset repeat operation
    ds = ds.repeat(repeat_num)

    return ds


ds = create_dataset(train_path, do_train=True, repeat_num=10, batch_size=32)
print("the cifar dataset size is:", ds.get_dataset_size())
dict1 = ds.create_dict_iterator()
datas = dict1.get_next()
image = datas["image"].asnumpy()
single_pic = np.transpose(image[0], (1,2,0))
print("the tensor of image is:", image.shape)
plt.imshow(np.array(single_pic))
plt.show()


def get_lr(lr_init, lr_end, lr_max, warmup_epochs, total_epochs, steps_per_epoch, lr_decay_mode):
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    if lr_decay_mode == 'steps':
        decay_epoch_index = [0.3 * total_steps, 0.6 * total_steps, 0.8 * total_steps]
        for i in range(total_steps):
            if i < decay_epoch_index[0]:
                lr = lr_max
            elif i < decay_epoch_index[1]:
                lr = lr_max * 0.1
            elif i < decay_epoch_index[2]:
                lr = lr_max * 0.01
            else:
                lr = lr_max * 0.001
            lr_each_step.append(lr)

    elif lr_decay_mode == 'poly':
        if warmup_steps != 0:
            inc_each_step = (float(lr_max) - float(lr_init)) / float(warmup_steps)
        else:
            inc_each_step = 0
        for i in range(total_steps):
            if i < warmup_steps:
                lr = float(lr_init) + inc_each_step * float(i)
            else:
                base = (1.0 - (float(i) - float(warmup_steps)) / (float(total_steps) - float(warmup_steps)))
                lr = float(lr_max) * base * base
                if lr < 0.0:
                    lr = 0.0
            lr_each_step.append(lr)
    else:
        for i in range(total_steps):
            if i < warmup_steps:
                lr = lr_init + (lr_max - lr_init) * i / warmup_steps
            else:
                lr = lr_max - (lr_max - lr_end) * (i - warmup_steps) / (total_steps - warmup_steps)
            lr_each_step.append(lr)

    lr_each_step = np.array(lr_each_step).astype(np.float32)

    return lr_each_step


def linear_warmup_lr(current_step, warmup_steps, base_lr, init_lr):
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    lr = float(init_lr) + lr_inc * current_step
    return lr


def warmup_cosine_annealing_lr(lr, steps_per_epoch, warmup_epochs, max_epoch=120, global_step=0):
    base_lr = lr
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    decay_steps = total_steps - warmup_steps

    lr_each_step = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr = linear_warmup_lr(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            linear_decay = (total_steps - i) / decay_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * 2 * 0.47 * i / decay_steps))
            decayed = linear_decay * cosine_decay + 0.00001
            lr = base_lr * decayed
        lr_each_step.append(lr)

    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[global_step:]
    return learning_rate


class CrossEntropy(_Loss):
    def __init__(self, smooth_factor=0., num_classes=1001):
        super(CrossEntropy, self).__init__()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.mean = P.ReduceMean(False)

    def construct(self, logit, label):
        one_hot_label = self.onehot(label, F.shape(logit)[1], self.on_value, self.off_value)
        loss = self.ce(logit, one_hot_label)
        loss = self.mean(loss, 0)
        return loss


def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)


def _conv3x3(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 3, 3)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=3, stride=stride, padding=0, pad_mode='same', weight_init=weight)


def _conv1x1(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 1, 1)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=1, stride=stride, padding=0, pad_mode='same', weight_init=weight)


def _conv7x7(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 7, 7)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=7, stride=stride, padding=0, pad_mode='same', weight_init=weight)


def _bn(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _bn_last(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=0, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _fc(in_channel, out_channel):
    weight_shape = (out_channel, in_channel)
    weight = _weight_variable(weight_shape)
    return nn.Dense(in_channel, out_channel, has_bias=True, weight_init=weight, bias_init=0)


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

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class ResNet(nn.Cell):

    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides,
                 num_classes):
        super(ResNet, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of layer_num, in_channels, out_channels list must be 4!")

        self.conv1 = _conv7x7(3, 64, stride=2)
        self.bn1 = _bn(64)
        self.relu = P.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0])
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1])
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2])
        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[3])

        self.mean = P.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()
        self.end_point = _fc(out_channels[3], num_classes)

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride):

        layers = []

        resnet_block = block(in_channel, out_channel, stride=stride)
        layers.append(resnet_block)

        for _ in range(1, layer_num):
            resnet_block = block(out_channel, out_channel, stride=1)
            layers.append(resnet_block)

        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        out = self.mean(c5, (2, 3))
        out = self.flatten(out)
        out = self.end_point(out)

        return out


def resnet50(class_num=10):
    return ResNet(ResidualBlock,
                  [3, 4, 6, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 2],
                  class_num)


def resnet101(class_num=1001):
    return ResNet(ResidualBlock,
                  [3, 4, 23, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 2],
                  class_num)

random.seed(1)
np.random.seed(1)
de.config.set_seed(1)
AMP_LEVEL = "O3"

if __name__ == '__main__':

    """train ResNet-50"""
    parser = argparse.ArgumentParser(description='Mixed Precision')
    parser.add_argument('--data_url', required=True, help='Location of data.')
    parser.add_argument('--train_url', required=True, default=None, help='Location of training outputs.')
    parser.add_argument('--net', type=str, default="resnet50", help='Resnet Model, either resnet50 or resnet101')
    parser.add_argument('--dataset', type=str, default="cifar10", help='Dataset, either cifar10 or imagenet2012')
    parser.add_argument('--run_distribute', type=bool, default=False, help='Run distribute')
    parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
    parser.add_argument('--pre_trained', type=str, default=None, help='Pretrained checkpoint path')
    args_opt = parser.parse_known_args()[0]

    # import moxing as mox
    
    # mox.file.copy_parallel(src_url=args_opt.data_url, dst_url='./datasets/')

    ckpt_save_dir = "./resnet_ckpt"
    batch_size = 32
    epoch_size = 10
    dataset_path = "./datasets/cifar10/train"
    test_path = "./datasets/cifar10/test"

    # create dataset
    dataset = create_dataset(dataset_path=dataset_path, do_train=True, repeat_num=1,
                             batch_size=batch_size)
    step_size = dataset.get_dataset_size()
    # define net
    net = resnet50(class_num=10)

    # init weight
    if args_opt.pre_trained:
        param_dict = load_checkpoint(args_opt.pre_trained)
        load_param_into_net(net, param_dict)
    else:
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data = weight_init.initializer(weight_init.XavierUniform(),
                                                                    cell.weight.data.shape,
                                                                    cell.weight.data.dtype).to_tensor()
            if isinstance(cell, nn.Dense):
                cell.weight.set_data = weight_init.initializer(weight_init.TruncatedNormal(),
                                                                    cell.weight.data.shape,
                                                                    cell.weight.data.dtype).to_tensor()
    # init lr
    warmup_epochs = 5
    lr_init = 0.01
    lr_end = 0.00001
    lr_max = 0.1
    lr = get_lr(lr_init=lr_init, lr_end=lr_end, lr_max=lr_max,
                warmup_epochs=warmup_epochs, total_epochs=epoch_size, steps_per_epoch=step_size,
                lr_decay_mode='poly')
    lr = Tensor(lr)

    # define opt
    loss_scale = 1024
    momentum = 0.9
    weight_decay = 1e-4

    # define loss, model
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, momentum)
    
    eval_net = nn.WithEvalCell(net, loss, AMP_LEVEL in ["O2", "O3"])
    model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'},amp_level=AMP_LEVEL, eval_network=eval_net, 
              eval_indexes=[0, 1, 2], keep_batchnorm_fp32=False)

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()

    cb = [time_cb, loss_cb]
    save_checkpoint = 5
    if save_checkpoint:
        save_checkpoint_epochs = 5
        keep_checkpoint_max = 10
        config_ck = CheckpointConfig(save_checkpoint_steps=save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="resnet", directory=ckpt_save_dir, config=config_ck)
        cb += [ckpt_cb]

    # train model
    model.train(epoch_size, dataset, callbacks=cb, dataset_sink_mode=True)

    # Eval model
    eval_dataset_path = "./datasets/cifar10/test"
    eval_data = create_dataset(eval_dataset_path,do_train=False)
    acc = model.eval(eval_data,dataset_sink_mode=True)
    print("Accuracy:",acc)