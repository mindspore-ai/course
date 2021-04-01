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
#################eval gru ######################
"""
import argparse
import os
import numpy as np
from src.dataset import create_dataset
from src.seq2seq import Seq2Seq, InferCell
from src.config import cfg
from mindspore import Tensor, nn, Model, context, DatasetHelper
from mindspore.train.serialization import load_param_into_net, load_checkpoint
from mindspore.communication.management import init, get_rank
from mindspore.context import ParallelMode

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MindSpore GRU Example')
    parser.add_argument('--dataset_path', type=str, default='./preprocess', help='dataset path.')
    parser.add_argument('--checkpoint_path', type=str, default='', help='checkpoint path.')
    args = parser.parse_args()

    context.set_context(
        mode=context.GRAPH_MODE,#PYNATIVE_MODE,#GRAPH_MODE,
        save_graphs=False,
        device_target='Ascend')

    rank = 0
    device_num = 1
    ds_eval= create_dataset(args.dataset_path, cfg.eval_batch_size, is_training=False)

    network = Seq2Seq(cfg,is_train=False)
    network = InferCell(network, cfg)
    network.set_train(False)
    parameter_dict = load_checkpoint(args.checkpoint_path)
    load_param_into_net(network, parameter_dict)
    model = Model(network)

    with open(os.path.join(args.dataset_path,"en_vocab.txt"), 'r', encoding='utf-8') as f:
        data = f.read()
    en_vocab = list(data.split('\n'))

    with open(os.path.join(args.dataset_path,"ch_vocab.txt"), 'r', encoding='utf-8') as f:
        data = f.read()
    ch_vocab = list(data.split('\n'))

    for data in ds_eval.create_dict_iterator():
        en_data=''
        ch_data=''
        for x in data['encoder_data'][0]:
            if x == 0:
                break
            en_data += en_vocab[x]
            en_data += ' '
        for x in data['decoder_data'][0]:
            if x == 0:
                break
            if x == 1:
                continue
            ch_data += ch_vocab[x]
        output = network(data['encoder_data'],data['decoder_data'])
        print('English:', en_data)
        print('expect Chinese:', ch_data)
        out =''
        for x in output[0]:
            if x == 0:
                break
            out += ch_vocab[x]
        print('predict Chinese:', out)
        print(' ')
