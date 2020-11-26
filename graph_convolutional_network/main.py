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
GCN training script.
"""

import os
# os.environ['DEVICE_ID']='7'

import time
import argparse
import numpy as np

from mindspore import context
from easydict import EasyDict as edict

from src.gcn import GCN, LossAccuracyWrapper, TrainNetWrapper
from src.config import ConfigGCN
from src.dataset import get_adj_features_labels, get_mask
from graph_to_mindrecord.writer import run

context.set_context(mode=context.GRAPH_MODE,device_target="Ascend", save_graphs=False)

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
    parser.add_argument('--data_url', required=True, help='Location of data.')
    parser.add_argument('--train_url', required=True, default=None, help='Location of training outputs.')
    args_opt = parser.parse_args()

    import moxing as mox
    mox.file.copy_parallel(src_url=args_opt.data_url, dst_url='./data')  # 将OBS桶中数据拷贝到容器中

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

    # src_url本地   将容器输出放入OBS桶中
    #mox.file.copy_parallel(src_url='data_mr', dst_url=cfg.MINDRECORD_PATH)  
