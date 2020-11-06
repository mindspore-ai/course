import csv
import os
import time

import numpy as np
from easydict import EasyDict as edict
from matplotlib import pyplot as plt

import mindspore
from mindspore import nn
from mindspore import context
from mindspore import dataset
from mindspore.train.callback import TimeMonitor, LossMonitor
from mindspore import Tensor
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

# 解析执行本脚本时的传参
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
parser.add_argument('--train_url', required=True, default=None, help='Location of training outputs.')
args, unknown = parser.parse_known_args()

import moxing as mox
mox.file.copy_parallel(src_url=os.path.join(args.data_url, 'iris.data'), dst_url='iris.data')  # 将OBS桶中数据拷贝到容器中

cfg = edict({
    'data_size': 150,
    'train_size': 120,  # 训练集大小
    'test_size': 30,  # 测试集大小
    'feature_number': 4,  # 输入特征数
    'num_class': 3,  # 分类类别
    'batch_size': 30,
    'data_dir': 'iris.data',  # 执行容器中数据集所在路径
    'save_checkpoint_steps': 5,  # 多少步保存一次模型
    'keep_checkpoint_max': 1,  # 最多保存多少个模型
    'out_dir_no_opt': './model_iris/no_opt',  # 保存模型路径，无优化器模型
    'out_dir_sgd': './model_iris/sgd',  # 保存模型路径,SGD优化器模型
    'out_dir_momentum': './model_iris/momentum',  # 保存模型路径，momentum模型
    'out_dir_adam': './model_iris/adam',  # 保存模型路径，adam优化器模型
    'output_prefix': "checkpoint_fashion_forward"  # 保存模型文件名
})


# 读取数据并预处理:
with open(cfg.data_dir) as csv_file:
    data = list(csv.reader(csv_file, delimiter=','))

# 共150条数据，将数据集的4个属性作为自变量X。将数据集的3个类别映射为{0, 1，2}，作为因变量Y。
label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
X = np.array([[float(x) for x in s[:-1]] for s in data[:cfg.data_size]], np.float32)
Y = np.array([label_map[s[-1]] for s in data[:150]], np.int32)

# 将数据集分为训练集120条，测试集30条。
train_idx = np.random.choice(cfg.data_size, cfg.train_size, replace=False)
test_idx = np.array(list(set(range(cfg.data_size)) - set(train_idx)))
X_train, Y_train = X[train_idx], Y[train_idx]
X_test, Y_test = X[test_idx], Y[test_idx]
print('训练数据x尺寸：', X_train.shape)
print('训练数据y尺寸：', Y_train.shape)
print('测试数据x尺寸：', X_test.shape)
print('测试数据y尺寸：', Y_test.shape)

# 使用MindSpore GeneratorDataset接口将numpy.ndarray类型的数据转换为Dataset。
def gen_data(X_train, Y_train, epoch_size):
    XY_train = list(zip(X_train, Y_train))
    ds_train = dataset.GeneratorDataset(XY_train, ['x', 'y'])
    ds_train = ds_train.shuffle(buffer_size=cfg.train_size).batch(cfg.batch_size, drop_remainder=True)
    XY_test = list(zip(X_test, Y_test))
    ds_test = dataset.GeneratorDataset(XY_test, ['x', 'y'])
    ds_test = ds_test.shuffle(buffer_size=cfg.test_size).batch(cfg.test_size, drop_remainder=True)
    return ds_train, ds_test


# 训练
def train(network, net_opt, ds_train, prefix, directory, print_times):
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    model = Model(network, loss_fn=net_loss, optimizer=net_opt, metrics={"acc"})
    loss_cb = LossMonitor(per_print_times=print_times)
    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                 keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix=prefix, directory=directory, config=config_ck)
    print("============== Starting Training ==============")
    model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, loss_cb], dataset_sink_mode=False)
    return model


# 评估预测
def eval_predict(model, ds_test):
    # 使用测试集评估模型，打印总体准确率
    metric = model.eval(ds_test, dataset_sink_mode=False)
    print(metric)
    # 预测
    test_ = ds_test.create_dict_iterator().get_next()
    test = Tensor(test_['x'], mindspore.float32)
    predictions = model.predict(test)
    predictions = predictions.asnumpy()
    for i in range(10):
        p_np = predictions[i, :]
        p_list = p_np.tolist()
        print('第' + str(i) + '个sample预测结果：', p_list.index(max(p_list)), '   真实结果：', test_['y'][i])


# --------------------------------------------------无优化器-----------------------------------
epoch_size = 20
print('------------------无优化器--------------------------')
# 数据
ds_train, ds_test = gen_data(X_train, Y_train, epoch_size)
# 定义网络并训练
network = nn.Dense(cfg.feature_number, cfg.num_class)
model = train(network, None, ds_train, "checkpoint_no_opt", cfg.out_dir_no_opt, 4)
# 评估预测
eval_predict(model, ds_test)

# ---------------------------------------------------SGD-------------------------------------
epoch_size = 200
lr = 0.01
print('-------------------SGD优化器-----------------------')
# 数据
ds_train, ds_test = gen_data(X_train, Y_train, epoch_size)
# 定义网络并训练、测试、预测
network = nn.Dense(cfg.feature_number, cfg.num_class)
net_opt = nn.SGD(network.trainable_params(), lr)
model = train(network, net_opt, ds_train, "checkpoint_sgd", cfg.out_dir_sgd, 40)
# 评估预测
eval_predict(model, ds_test)

# ----------------------------------------------------Momentum-------------------------------
epoch_size = 20
lr = 0.01
print('-------------------Momentum优化器-----------------------')
# 数据
ds_train, ds_test = gen_data(X_train, Y_train, epoch_size)
# 定义网络并训练
network = nn.Dense(cfg.feature_number, cfg.num_class)
net_opt = nn.Momentum(network.trainable_params(), lr, 0.9)
model = train(network, net_opt, ds_train, "checkpoint_momentum", cfg.out_dir_momentum, 4)
# 评估预测
eval_predict(model, ds_test)

# ----------------------------------------------------Adam-----------------------------------
epoch_size = 15
lr = 0.1
print('------------------Adam优化器--------------------------')
# 数据
ds_train, ds_test = gen_data(X_train, Y_train, epoch_size)
# 定义网络并训练
network = nn.Dense(cfg.feature_number, cfg.num_class)
net_opt = nn.Adam(network.trainable_params(), learning_rate=lr)
model = train(network, net_opt, ds_train, "checkpoint_adam", cfg.out_dir_adam, 4)
# 评估预测
eval_predict(model, ds_test)

# -------------------------------------------------------------------------------------------
mox.file.copy_parallel(src_url='model_iris', dst_url=args.train_url)  # 将容器输出拷贝到OBS桶中
