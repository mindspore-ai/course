# Copyright 2020 Huawei Technologies Co., Ltd
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
"""
#################train gru########################
"""
import argparse
import os

import numpy as np
from src.dataset import create_dataset
from src.seq2seq import Seq2Seq, WithLossCell
from src.config import cfg
from mindspore import Tensor, nn, Model, context
from mindspore.train.callback import LossMonitor, CheckpointConfig, ModelCheckpoint, TimeMonitor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MindSpore LSTM Example')
    parser.add_argument('--dataset_path', type=str, default='./preprocess', help='dataset path.')
    parser.add_argument('--ckpt_save_path', type=str, default='./', help='checkpoint save path.')
    args = parser.parse_args()

    context.set_context(
        mode=context.GRAPH_MODE,
        save_graphs=False,
        device_target='Ascend')

    ds_train = create_dataset(args.dataset_path, cfg.batch_size)

    network = Seq2Seq(cfg)
    network = WithLossCell(network, cfg)
    optimizer = nn.Adam(network.trainable_params(), learning_rate=cfg.learning_rate, beta1=0.9, beta2=0.98)
    model = Model(network, optimizer=optimizer)

    loss_cb = LossMonitor()
    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps, keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="gru", directory=args.ckpt_save_path, config=config_ck)
    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    callbacks = [time_cb, ckpoint_cb, loss_cb]

    model.train(cfg.num_epochs, ds_train, callbacks=callbacks, dataset_sink_mode=False)
