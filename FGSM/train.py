# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""train"""

import os

from mindspore import context, Model
import mindspore.nn as nn
from mindspore.train.callback import LossMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.common import set_seed

from src.data.dataset import create_dataset
from src.net.lenet import LeNet5


def test_net(model, data_path):
    """test_net"""
    ds_eval = create_dataset(os.path.join(data_path, "test"))
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    print("{}".format(acc))


def train_net(model, epoch_size, data_path, ckpoint_cb, sink_mode):
    """train_net"""
    ds_train = create_dataset(os.path.join(data_path, "train"), 32)
    model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor(125)], dataset_sink_mode=sink_mode)


if __name__ == '__main__':
    set_seed(1)

    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')

    net = LeNet5()
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_opt = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)

    config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
    ckpoint = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)

    train_epoch = 1
    mnist_path = "./datasets/MNIST_Data/"
    model = Model(net, net_loss, net_opt, metrics={"Accuracy": nn.Accuracy()})
    train_net(model, train_epoch, mnist_path, ckpoint, False)
    test_net(model, mnist_path)
