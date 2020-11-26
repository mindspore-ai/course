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
GAT training script.
"""

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


if __name__ == '__main__':
    #------------------------定义变量------------------------------
    parser = argparse.ArgumentParser(description='GAT')
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
    # mox.file.copy_parallel(src_url='data_mr', dst_url=cfg.MINDRECORD_PATH)  