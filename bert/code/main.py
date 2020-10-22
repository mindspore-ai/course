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

from src.utils import BertFinetuneCell, BertCLS, BertNER
from src.config import cfg, bert_net_cfg, tag_to_index, bert_optimizer_cfg

import mindspore.common.dtype as mstype
from mindspore import context
from mindspore import log as logger
from mindspore.common.tensor import Tensor
import mindspore.dataset as de
import mindspore.dataset.transforms.c_transforms as C
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn.optim import AdamWeightDecayDynamicLR, Lamb, Momentum
from mindspore.train.model import Model
from mindspore.train.callback import Callback
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.CRF import postprocess
from src.cluener_evaluation import submit


def get_dataset(batch_size=1, repeat_count=1, distribute_file=''):
    '''
    get dataset
    '''
    ds = de.TFRecordDataset([cfg.data_file], cfg.schema_file, columns_list=["input_ids", "input_mask","segment_ids", "label_ids"])
    type_cast_op = C.TypeCast(mstype.int32)
    ds = ds.map(input_columns="segment_ids", operations=type_cast_op)
    ds = ds.map(input_columns="input_mask", operations=type_cast_op)
    ds = ds.map(input_columns="input_ids", operations=type_cast_op)
    ds = ds.map(input_columns="label_ids", operations=type_cast_op)
    ds = ds.repeat(repeat_count)
    
    # apply shuffle operation
    buffer_size = 960
    ds = ds.shuffle(buffer_size=buffer_size)

    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds

class Accuracy():
    '''
    calculate accuracy
    '''
    def __init__(self):
        self.acc_num = 0
        self.total_num = 0
    def update(self, logits, labels):
        labels = labels.asnumpy()
        labels = np.reshape(labels, -1)
        logits = logits.asnumpy()
        logit_id = np.argmax(logits, axis=-1)
        self.acc_num += np.sum(labels == logit_id)
        self.total_num += len(labels)
        #print("=========================accuracy is ", self.acc_num / self.total_num)

class F1():
    '''
    calculate F1 score
    '''
    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0
    def update(self, logits, labels):
        '''
        update F1 score
        '''
        labels = labels.asnumpy()
        labels = np.reshape(labels, -1)
        if cfg.use_crf:
            backpointers, best_tag_id = logits
            best_path = postprocess(backpointers, best_tag_id)
            logit_id = []
            for ele in best_path:
                logit_id.extend(ele)
        else:
            logits = logits.asnumpy()
            logit_id = np.argmax(logits, axis=-1)
            logit_id = np.reshape(logit_id, -1)
        pos_eva = np.isin(logit_id, [i for i in range(1, cfg.num_labels)])
        pos_label = np.isin(labels, [i for i in range(1, cfg.num_labels)])
        self.TP += np.sum(pos_eva&pos_label)
        self.FP += np.sum(pos_eva&(~pos_label))
        self.FN += np.sum((~pos_eva)&pos_label)
        
def test_train():
    '''
    finetune function
    '''
    target = args_opt.device_target
    if target == "Ascend":
        devid = int(os.getenv('DEVICE_ID'))
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=devid)
    elif target == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
        if bert_net_cfg.compute_type != mstype.float32:
            logger.warning('GPU only support fp32 temporarily, run with fp32.')
            bert_net_cfg.compute_type = mstype.float32
    else:
        raise Exception("Target error, GPU or Ascend is supported.")
        
    #BertCLSTrain for classification
    #BertNERTrain for sequence labeling
    if cfg.task == 'NER':
        if cfg.use_crf:
            netwithloss = BertNER(bert_net_cfg, True, num_labels=len(tag_to_index), use_crf=True,
                                  tag_to_index=tag_to_index, dropout_prob=0.1)
        else:
            netwithloss = BertNER(bert_net_cfg, True, num_labels=cfg.num_labels, dropout_prob=0.1)
    elif cfg.task == 'Classification':
        netwithloss = BertCLS(bert_net_cfg, True, num_labels=cfg.num_labels, dropout_prob=0.1)
    else:
        raise Exception("task error, NER or Classification is supported.")
        
    
    dataset = get_dataset(bert_net_cfg.batch_size, cfg.epoch_num)
        
    # optimizer
    steps_per_epoch = dataset.get_dataset_size()
    if cfg.optimizer == 'AdamWeightDecayDynamicLR':
        optimizer = AdamWeightDecayDynamicLR(netwithloss.trainable_params(),
                                             decay_steps=steps_per_epoch * cfg.epoch_num,
                                             learning_rate=bert_optimizer_cfg.AdamWeightDecayDynamicLR.learning_rate,
                                             end_learning_rate=bert_optimizer_cfg.AdamWeightDecayDynamicLR.end_learning_rate,
                                             power=bert_optimizer_cfg.AdamWeightDecayDynamicLR.power,
                                             warmup_steps=int(steps_per_epoch * cfg.epoch_num * 0.1),
                                             weight_decay=bert_optimizer_cfg.AdamWeightDecayDynamicLR.weight_decay,
                                             eps=bert_optimizer_cfg.AdamWeightDecayDynamicLR.eps)
    elif cfg.optimizer == 'Lamb':
        optimizer = Lamb(netwithloss.trainable_params(), decay_steps=steps_per_epoch * cfg.epoch_num,
                         start_learning_rate=bert_optimizer_cfg.Lamb.start_learning_rate,
                         end_learning_rate=bert_optimizer_cfg.Lamb.end_learning_rate,
                         power=bert_optimizer_cfg.Lamb.power, weight_decay=bert_optimizer_cfg.Lamb.weight_decay,
                         warmup_steps=int(steps_per_epoch * cfg.epoch_num * 0.1), decay_filter=bert_optimizer_cfg.Lamb.decay_filter)
    elif cfg.optimizer == 'Momentum':
        optimizer = Momentum(netwithloss.trainable_params(), learning_rate=bert_optimizer_cfg.Momentum.learning_rate,
                             momentum=bert_optimizer_cfg.Momentum.momentum)
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
    model.train(cfg.epoch_num, dataset, callbacks=[LossMonitor(), ckpoint_cb], dataset_sink_mode=True)

def bert_predict(Evaluation):
    '''
    prediction function
    '''
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
    if cfg.data_file[-4:]=='json':
        dataset=None
    else:
        dataset = get_dataset(bert_net_cfg.batch_size, 1)
    if cfg.use_crf and cfg.task == "NER":
        net_for_pretraining = Evaluation(bert_net_cfg, False, num_labels=len(tag_to_index), use_crf=True,
                                         tag_to_index=tag_to_index, dropout_prob=0.0)
    else:
        net_for_pretraining = Evaluation(bert_net_cfg, False, cfg.num_labels)
    net_for_pretraining.set_train(False)
    param_dict = load_checkpoint(cfg.finetune_ckpt)
    load_param_into_net(net_for_pretraining, param_dict)
    model = Model(net_for_pretraining)
    return model, dataset

def test_eval():
    '''
    evaluation function
    '''
    task_type = BertNER if cfg.task == "NER" else BertCLS
    model, dataset = bert_predict(task_type)
    if cfg.data_file[-4:]=='json':
        submit(model, cfg.data_file, bert_net_cfg.seq_length)
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
    
    import moxing as mox
    mox.file.copy_parallel(src_url=args_opt.data_url, dst_url='./data/')
    mox.file.copy_parallel(src_url=args_opt.ckpt_url, dst_url='./ckpt/')
    if cfg.is_train:
        test_train()
        mox.file.copy_parallel(src_url=cfg.ckpt_dir, dst_url=args_opt.train_url)
    else:
        test_eval()
        mox.file.copy_parallel(src_url='./log.txt', dst_url=os.path.join(args_opt.train_url, 'log.txt'))
    