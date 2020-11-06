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
"""Transformer training script."""

import time
import argparse
import random
import numpy as np

import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.nn.optim import Adam
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.train.callback import Callback, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.dataset.engine.datasets as de
import mindspore.dataset.transforms.c_transforms as deC
import mindspore.communication.management as D
from mindspore.context import ParallelMode
from mindspore import context

from src.transformer_for_train import TransformerTrainOneStepCell, TransformerNetworkWithLoss, TransformerTrainOneStepWithLossScaleCell
from src.train_config import cfg, transformer_net_cfg
from src.lr_schedule import create_dynamic_lr

def get_ms_timestamp():
    t = time.time()
    return int(round(t * 1000))
time_stamp_init = False
time_stamp_first = 0


def create_transformer_dataset(batch_size,epoch_count=1, rank_size=1, rank_id=0, do_shuffle=True, dataset_path=None):
    """create dataset"""
    repeat_count = epoch_count
    print(dataset_path)
    ds = de.MindDataset(dataset_file=dataset_path,
                        columns_list=["source_eos_ids","source_eos_mask",
                                  "target_sos_ids", "target_sos_mask","target_eos_ids", "target_eos_mask"])
                        #shuffle=do_shuffle, num_shards=rank_size, shard_id=rank_id)

    type_cast_op = deC.TypeCast(mstype.int32)
    ds = ds.map(input_columns="source_eos_ids", operations=type_cast_op)
    ds = ds.map(input_columns="source_eos_mask", operations=type_cast_op)
    ds = ds.map(input_columns="target_sos_ids", operations=type_cast_op)
    ds = ds.map(input_columns="target_sos_mask", operations=type_cast_op)
    ds = ds.map(input_columns="target_eos_ids", operations=type_cast_op)
    ds = ds.map(input_columns="target_eos_mask", operations=type_cast_op)
    ds.channel_name = 'transformer'
    
    #data = ds.create_dict_iterator().get_next()
    #print(data['source_eos_ids'].shape)
    #print(data['source_eos_mask'].shape)
    #print(data['target_sos_ids'].shape)
    #print(data['target_sos_mask'].shape)
    #print(data['target_eos_ids'].shape)
    #print(data['target_eos_mask'].shape)

    print(ds.get_dataset_size())
    
    # apply batch operations
    train_data = ds.batch(batch_size, drop_remainder=True)
    #train_data = train_data.repeat(repeat_count)
    print(train_data.get_dataset_size())
    return train_data

class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss is NAN or INF terminating training.
    Note:
        If per_print_times is 0 do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """
    def __init__(self, per_print_times=1):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        global time_stamp_init, time_stamp_first
        if not time_stamp_init:
            time_stamp_first = get_ms_timestamp()
            time_stamp_init = True

    def step_end(self, run_context):
        global time_stamp_first
        time_stamp_current = get_ms_timestamp()
        cb_params = run_context.original_args()
        print("time: {}, epoch: {}, step: {}, outputs are {}".format(time_stamp_current - time_stamp_first,
                                                                     cb_params.cur_epoch_num, cb_params.cur_step_num,
                                                                     str(cb_params.net_outputs)))
        with open("./loss.log", "a+") as f:
            f.write("time: {}, epoch: {}, step: {}, outputs are {}".format(time_stamp_current - time_stamp_first,
                                                                           cb_params.cur_epoch_num,
                                                                           cb_params.cur_step_num,
                                                                           str(cb_params.net_outputs)))
            f.write('\n')

def run_transformer_train():
    """
    Transformer training.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(reserve_class_name_in_scope=False, enable_auto_mixed_precision=False)

    if cfg.distribute:
        device_num = cfg.device_num
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, mirror_mean=True,
                                          parameter_broadcast=True, device_num=device_num)
        D.init()
        rank_id = args.device_id % device_num
    else:
        device_num = 1
        rank_id = 0
    
    train_dataset = create_transformer_dataset(cfg.batch_size,epoch_count=cfg.epoch_size, rank_size=device_num,
                                                       rank_id=rank_id, do_shuffle=cfg.do_shuffle,
                                                       dataset_path=cfg.data_path)

    netwithloss = TransformerNetworkWithLoss(transformer_net_cfg, True)

    if cfg.checkpoint_path:
        parameter_dict = load_checkpoint(cfg.checkpoint_path)
        load_param_into_net(netwithloss, parameter_dict)

    lr = Tensor(create_dynamic_lr(schedule="constant*rsqrt_hidden*linear_warmup*rsqrt_decay",
                                  training_steps=train_dataset.get_dataset_size()*cfg.epoch_size,
                                  learning_rate=cfg.lr_schedule.learning_rate,
                                  warmup_steps=cfg.lr_schedule.warmup_steps,
                                  hidden_size=transformer_net_cfg.hidden_size,
                                  start_decay_step=cfg.lr_schedule.start_decay_step,
                                  min_lr=cfg.lr_schedule.min_lr), mstype.float32)
    optimizer = Adam(netwithloss.trainable_params(), lr)

    callbacks = [TimeMonitor(train_dataset.get_dataset_size()), LossCallBack()]
    if cfg.enable_save_ckpt:
        ckpt_config = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                       keep_checkpoint_max=cfg.save_checkpoint_num)
        ckpoint_cb = ModelCheckpoint(prefix=cfg.save_checkpoint_name, directory=cfg.save_checkpoint_path, config=ckpt_config)
        callbacks.append(ckpoint_cb)

    if cfg.enable_lossscale:
        scale_manager = DynamicLossScaleManager(init_loss_scale=cfg.init_loss_scale_value,
                                                scale_factor=cfg.scale_factor,
                                                scale_window=cfg.scale_window)
        update_cell = scale_manager.get_update_cell()
        netwithgrads = TransformerTrainOneStepWithLossScaleCell(netwithloss, optimizer=optimizer,scale_update_cell=update_cell)
    else:
        netwithgrads = TransformerTrainOneStepCell(netwithloss, optimizer=optimizer)

    netwithgrads.set_train(True)
    model = Model(netwithgrads)
    model.train(cfg.epoch_size, train_dataset, callbacks=callbacks, dataset_sink_mode=cfg.enable_data_sink)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer training')
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')   
    parser.add_argument('--train_url', required=True, default=None, help='Location of training outputs.')
    args_opt = parser.parse_args()
    
    import moxing as mox
    mox.file.copy_parallel(src_url=args_opt.data_url, dst_url='./data/')
    run_transformer_train()
    mox.file.copy_parallel(src_url=cfg.save_checkpoint_path, dst_url=args_opt.train_url)
