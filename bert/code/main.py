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

'''
Bert finetune script.
'''

import os
import argparse
import numpy as np
import json

import mindspore.common.dtype as mstype
from mindspore import context
from mindspore import log as logger
from mindspore.common.tensor import Tensor
import mindspore.dataset as de
import mindspore.dataset.transforms.c_transforms as C
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn.optim import AdamWeightDecay, Lamb, Momentum
from mindspore.train.model import Model
from mindspore.train.callback import Callback
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.cluener_evaluation import submit
from src.utils import BertLearningRate, LossCallBack, Accuracy, F1, MCC, Spearman_Correlation
from src.bert_for_finetune import BertFinetuneCell, BertCLS, BertNER
from src.config import cfg, bert_net_cfg, optimizer_cfg


def get_dataset(data_file, batch_size):
    '''
    get dataset
    '''
    ds = de.TFRecordDataset([data_file], cfg.schema_file, columns_list=["input_ids", "input_mask","segment_ids", "label_ids"])
    type_cast_op = C.TypeCast(mstype.int32)
    ds = ds.map(input_columns="segment_ids", operations=type_cast_op)
    ds = ds.map(input_columns="input_mask", operations=type_cast_op)
    ds = ds.map(input_columns="input_ids", operations=type_cast_op)
    ds = ds.map(input_columns="label_ids", operations=type_cast_op)
    
    # apply shuffle operation
    buffer_size = 960
    ds = ds.shuffle(buffer_size=buffer_size)

    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds
        
def train():
    '''
    finetune function
    '''
    # BertCLS train for classification
    # BertNER train for sequence labeling


    if cfg.task == 'NER':
        tag_to_index =None
        if cfg.use_crf:
            tag_to_index = json.loads(open(cfg.label2id_file).read())
            print(tag_to_index)
            max_val = len(tag_to_index)
            tag_to_index["<START>"] = max_val
            tag_to_index["<STOP>"] = max_val + 1
            number_labels = len(tag_to_index)
        else:
            number_labels = cfg.num_labels

        netwithloss = BertNER(bert_net_cfg, cfg.batch_size, True, num_labels=number_labels,
                              use_crf=cfg.use_crf,
                              tag_to_index=tag_to_index, dropout_prob=0.1)
    elif cfg.task == 'Classification':
        netwithloss = BertCLS(bert_net_cfg, True, num_labels=cfg.num_labels, dropout_prob=0.1,
                              assessment_method=cfg.assessment_method)
    else:
        raise Exception("task error, NER or Classification is supported.")

    dataset = get_dataset(data_file=cfg.data_file, batch_size=cfg.batch_size)
    steps_per_epoch = dataset.get_dataset_size()
    print('steps_per_epoch:',steps_per_epoch)

    # optimizer
    steps_per_epoch = dataset.get_dataset_size()
    if cfg.optimizer == 'AdamWeightDecay':
        lr_schedule = BertLearningRate(learning_rate=optimizer_cfg.AdamWeightDecay.learning_rate,
                                       end_learning_rate=optimizer_cfg.AdamWeightDecay.end_learning_rate,
                                       warmup_steps=int(steps_per_epoch * cfg.epoch_num * 0.1),
                                       decay_steps=steps_per_epoch * cfg.epoch_num,
                                       power=optimizer_cfg.AdamWeightDecay.power)
        params = netwithloss.trainable_params()
        decay_params = list(filter(optimizer_cfg.AdamWeightDecay.decay_filter, params))
        other_params = list(filter(lambda x: not optimizer_cfg.AdamWeightDecay.decay_filter(x), params))
        group_params = [{'params': decay_params, 'weight_decay': optimizer_cfg.AdamWeightDecay.weight_decay},
                        {'params': other_params, 'weight_decay': 0.0}]
        optimizer = AdamWeightDecay(group_params, lr_schedule, eps=optimizer_cfg.AdamWeightDecay.eps)
    elif cfg.optimizer == 'Lamb':
        lr_schedule = BertLearningRate(learning_rate=optimizer_cfg.Lamb.learning_rate,
                                       end_learning_rate=optimizer_cfg.Lamb.end_learning_rate,
                                       warmup_steps=int(steps_per_epoch * cfg.epoch_num * 0.1),
                                       decay_steps=steps_per_epoch * cfg.epoch_num,
                                       power=optimizer_cfg.Lamb.power)
        optimizer = Lamb(netwithloss.trainable_params(), learning_rate=lr_schedule)
    elif cfg.optimizer == 'Momentum':
        optimizer = Momentum(netwithloss.trainable_params(), learning_rate=optimizer_cfg.Momentum.learning_rate,
                             momentum=optimizer_cfg.Momentum.momentum)
    else:
        raise Exception("Optimizer not supported.")
        
    # load checkpoint into network
    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=1)
    ckpoint_cb = ModelCheckpoint(prefix=cfg.ckpt_prefix, directory=cfg.ckpt_dir, config=ckpt_config)
    param_dict = load_checkpoint(cfg.pre_training_ckpt)
    load_param_into_net(netwithloss, param_dict)

    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2**32, scale_factor=2, scale_window=1000)
    netwithgrads = BertFinetuneCell(netwithloss, optimizer=optimizer, scale_update_cell=update_cell)
    model = Model(netwithgrads)
    callbacks = [TimeMonitor(dataset.get_dataset_size()), LossCallBack(dataset.get_dataset_size()), ckpoint_cb]
    model.train(cfg.epoch_num, dataset, callbacks=callbacks, dataset_sink_mode=True)


def eval():
    '''
    evaluation function
    '''
    if cfg.data_file[-4:] == 'json':
        dataset = None
    else:
        dataset = get_dataset(cfg.data_file, 1)

    if cfg.task == "NER":
        if cfg.use_crf:
            tag_to_index = json.loads(open(cfg.label2id_file).read())
            max_val = len(tag_to_index)
            tag_to_index["<START>"] = max_val
            tag_to_index["<STOP>"] = max_val + 1
            number_labels = len(tag_to_index)
        else:
            number_labels = cfg.num_labels
            tag_to_index =None
        netwithloss = BertNER(bert_net_cfg, 1, False, num_labels=number_labels,
                                 use_crf=cfg.use_crf,
                                 tag_to_index=tag_to_index)
    elif cfg.task == 'Classification':
        netwithloss = BertCLS(bert_net_cfg, False, cfg.num_labels, assessment_method=cfg.assessment_method)
    else:
        raise Exception("task error, NER or Classification is supported.")

    netwithloss.set_train(False)
    param_dict = load_checkpoint(cfg.finetune_ckpt)
    load_param_into_net(netwithloss, param_dict)
    model = Model(netwithloss)

    if cfg.data_file[-4:]=='json':
        submit(model, cfg.data_file, bert_net_cfg.seq_length)
        # import moxing as mox
        # mox.file.copy_parallel(src_url=cfg.eval_out_file, dst_url=os.path.join(args_opt.train_url, cfg.eval_out_file))
    else:
        callback = F1() if cfg.task == "NER" else Accuracy()
        columns_list = ["input_ids", "input_mask", "segment_ids", "label_ids"]
        for data in dataset.create_dict_iterator():
            input_data = []
            for i in columns_list:
                input_data.append(Tensor(data[i]))
            input_ids, input_mask, token_type_id, label_ids = input_data
            logits = model.predict(input_ids, input_mask, token_type_id, label_ids)
            callback.update(logits, label_ids)
        print("==============================================================")
        if cfg.task == "NER":
            print("Precision {:.6f} ".format(callback.TP / (callback.TP + callback.FP)))
            print("Recall {:.6f} ".format(callback.TP / (callback.TP + callback.FN)))
            print("F1 {:.6f} ".format(2*callback.TP / (2*callback.TP + callback.FP + callback.FN)))
        else:
            print("acc_num {} , total_num {}, accuracy {:.6f}".format(callback.acc_num, callback.total_num,
                                                                      callback.acc_num / callback.total_num))
        print("==============================================================")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bert finetune')
    parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')   
    parser.add_argument('--ckpt_url', required=True, default=None, help='Location of data.') 
    parser.add_argument('--train_url', required=True, default=None, help='Location of training outputs.')
    args_opt = parser.parse_args()

    target = args_opt.device_target

    if target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    elif target == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
        if bert_net_cfg.compute_type != mstype.float32:
            logger.warning('GPU only support fp32 temporarily, run with fp32.')
            bert_net_cfg.compute_type = mstype.float32
    else:
        raise Exception("Target error, GPU or Ascend is supported.")

    # import moxing as mox
    # mox.file.copy_parallel(src_url=args_opt.data_url, dst_url='./data/')
    # mox.file.copy_parallel(src_url=args_opt.ckpt_url, dst_url='./ckpt/')
    if cfg.is_train:
        train()
        # mox.file.copy_parallel(src_url=cfg.ckpt_dir, dst_url=args_opt.train_url)
    else:
        eval()
        
    
