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
#################train lstm example on aclImdb########################
"""
import argparse
import os

import numpy as np

from src.config import lstm_cfg, lstm_cfg_ascend
from src.dataset import convert_to_mindrecord
from src.dataset import lstm_create_dataset
from src.lr_schedule import get_lr
from src.lstm import SentimentNet
from mindspore import Tensor, nn, Model, context
from mindspore.nn import Accuracy
from mindspore.train.callback import LossMonitor, CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train.serialization import load_param_into_net, load_checkpoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MindSpore LSTM Example')
    parser.add_argument('--preprocess', type=str, default='false', choices=['true', 'false'],
                        help='whether to preprocess data.')
    parser.add_argument('--data_url', type=str, default="./aclImdb",
                        help='path where the dataset is stored.')
    parser.add_argument('--glove_path', type=str, default="./glove",
                        help='path where the GloVe is stored.')
    parser.add_argument('--preprocess_path', type=str, default="./preprocess",
                        help='path where the pre-process data is stored.')
    parser.add_argument('--ckpt_path', type=str, default="./",
                        help='the path to save the checkpoint file.')
    parser.add_argument('--pre_trained', type=str, default=None,
                        help='the pretrained checkpoint file path.')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['GPU', 'CPU', 'Ascend'],
                        help='the target device to run, support "GPU", "CPU". Default: "Ascend".')
    parser.add_argument('--train_url', required=True, default=None, help='Location of training outputs.')
    args = parser.parse_args()

    context.set_context(
        mode=context.GRAPH_MODE,
        save_graphs=False,
        device_target=args.device_target)

    # import moxing as mox

    # mox.file.copy_parallel(src_url='s3://zhengnj-course/lstm/glove', dst_url=args.glove_path)

    if args.device_target == 'Ascend':
        cfg = lstm_cfg_ascend
    else:
        cfg = lstm_cfg

    if args.preprocess == "true":
        print("============== Starting Data Pre-processing ==============")
        convert_to_mindrecord(cfg.embed_size, args.data_url, args.preprocess_path, args.glove_path)

    embedding_table = np.loadtxt(os.path.join(args.preprocess_path, "weight.txt")).astype(np.float32)
    # DynamicRNN in this network on Ascend platform only support the condition that the shape of input_size
    # and hiddle_size is multiples of 16, this problem will be solved later.
    if args.device_target == 'Ascend':
        pad_num = int(np.ceil(cfg.embed_size / 16) * 16 - cfg.embed_size)
        if pad_num > 0:
            embedding_table = np.pad(embedding_table, [(0, 0), (0, pad_num)], 'constant')
        cfg.embed_size = int(np.ceil(cfg.embed_size / 16) * 16)
    network = SentimentNet(vocab_size=embedding_table.shape[0],
                           embed_size=cfg.embed_size,
                           num_hiddens=cfg.num_hiddens,
                           num_layers=cfg.num_layers,
                           bidirectional=cfg.bidirectional,
                           num_classes=cfg.num_classes,
                           weight=Tensor(embedding_table),
                           batch_size=cfg.batch_size)
    # pre_trained
    if args.pre_trained:
        load_param_into_net(network, load_checkpoint(args.pre_trained))

    ds_train = lstm_create_dataset(args.preprocess_path, cfg.batch_size, 1)

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    if cfg.dynamic_lr:
        lr = Tensor(get_lr(global_step=cfg.global_step,
                           lr_init=cfg.lr_init, lr_end=cfg.lr_end, lr_max=cfg.lr_max,
                           warmup_epochs=cfg.warmup_epochs,
                           total_epochs=cfg.num_epochs,
                           steps_per_epoch=ds_train.get_dataset_size(),
                           lr_adjust_epoch=cfg.lr_adjust_epoch))
    else:
        lr = cfg.learning_rate

    opt = nn.Momentum(network.trainable_params(), lr, cfg.momentum)
    loss_cb = LossMonitor()

    model = Model(network, loss, opt, {'acc': Accuracy()})

    print("============== Starting Training ==============")
    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                 keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="lstm", directory=args.ckpt_path, config=config_ck)
    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    if args.device_target == "CPU":
        model.train(cfg.num_epochs, ds_train, callbacks=[time_cb, ckpoint_cb, loss_cb], dataset_sink_mode=False)
    else:
        model.train(cfg.num_epochs, ds_train, callbacks=[time_cb, ckpoint_cb, loss_cb])
    print("============== Training Success ==============")

    # mox.file.copy_parallel(src_url='./preprocess', dst_url=args.train_url)
    # mox.file.copy_parallel(src_url='lstm-20_390.ckpt', dst_url=args.ckpt_path)

    ds_eval = lstm_create_dataset(args.preprocess_path, cfg.batch_size, training=False)

    print("============== Starting Testing ==============")
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(network, param_dict)
    if args.device_target == "CPU":
        acc = model.eval(ds_eval, dataset_sink_mode=False)
    else:
        acc = model.eval(ds_eval)
    print("============== {} ==============".format(acc))


