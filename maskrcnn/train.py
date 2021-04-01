# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""train MaskRcnn and get checkpoint files."""

import os
import time
import ast

from easydict import EasyDict as edict
import mindspore.common.dtype as mstype
from mindspore import context, Tensor
from mindspore.communication.management import init
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn import Momentum
from mindspore.common import set_seed
from mindspore.communication.management import get_rank, get_group_size

from src.maskrcnn.mask_rcnn_r50 import Mask_Rcnn_Resnet50
from src.network_define import LossCallBack, WithLossCell, TrainOneStepCell, LossNet
from src.config import config
from src.dataset import data_to_mindrecord_byte_image, create_maskrcnn_dataset
from src.lr_schedule import dynamic_lr

set_seed(1)

args_opt = edict({
    "only_create_dataset": False,
    "run_distribute": False,
    "do_train": True,
    "do_eval": False,
    "dataset": "coco",
    "pre_trained": "./resnet50_backbone.ckpt",
    "device_id": 0,
    "device_num": 1,
    "rank_id": 0,
})

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args_opt.device_id)

if __name__ == '__main__':
    import time
    print("Start train for maskrcnn!")
    if not args_opt.do_eval and args_opt.run_distribute:
        init()
        rank = get_rank()
        device_num = get_group_size()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
    else:
        rank = 0
        device_num = 1

    print("Start create dataset!")

    # It will generate mindrecord file in args_opt.mindrecord_dir,
    # and the file name is MaskRcnn.mindrecord0, 1, ... file_num.
    prefix = "MaskRcnn.mindrecord"
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix + "0")
    if rank == 0 and not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if args_opt.dataset == "coco":
            if os.path.isdir(config.coco_root):
                print("Create Mindrecord.")
                data_to_mindrecord_byte_image("coco", True, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                raise Exception("coco_root not exits.")
        else:
            if os.path.isdir(config.IMAGE_DIR) and os.path.exists(config.ANNO_PATH):
                print("Create Mindrecord.")
                data_to_mindrecord_byte_image("other", True, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                raise Exception("IMAGE_DIR or ANNO_PATH not exits.")
    while not os.path.exists(mindrecord_file+".db"):
        time.sleep(5)

    if not args_opt.only_create_dataset:
        loss_scale = float(config.loss_scale)

        # When create MindDataset, using the fitst mindrecord file, such as MaskRcnn.mindrecord0.
        dataset = create_maskrcnn_dataset(mindrecord_file, batch_size=config.batch_size,
                                          device_num=device_num, rank_id=rank)

        dataset_size = dataset.get_dataset_size()
        print("total images num: ", dataset_size)
        print("Create dataset done!")

        net = Mask_Rcnn_Resnet50(config=config)
        net = net.set_train()

        load_path = args_opt.pre_trained
        if load_path != "":
            param_dict = load_checkpoint(load_path)
            if config.pretrain_epoch_size == 0:
                for item in list(param_dict.keys()):
                    if not (item.startswith('backbone') or item.startswith('rcnn_mask')):
                        param_dict.pop(item)
            load_param_into_net(net, param_dict)

        loss = LossNet()
        lr = Tensor(dynamic_lr(config, rank_size=device_num, start_steps=config.pretrain_epoch_size * dataset_size),
                    mstype.float32)
        opt = Momentum(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum,
                       weight_decay=config.weight_decay, loss_scale=config.loss_scale)

        net_with_loss = WithLossCell(net, loss)
        if args_opt.run_distribute:
            net = TrainOneStepCell(net_with_loss, net, opt, sens=config.loss_scale, reduce_flag=True,
                                   mean=True, degree=device_num)
        else:
            net = TrainOneStepCell(net_with_loss, net, opt, sens=config.loss_scale)

        time_cb = TimeMonitor(data_size=dataset_size)
        loss_cb = LossCallBack(rank_id=rank)
        cb = [time_cb, loss_cb]
        if config.save_checkpoint:
            ckptconfig = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * dataset_size,
                                          keep_checkpoint_max=config.keep_checkpoint_max)
            save_checkpoint_path = os.path.join(config.save_checkpoint_path, 'ckpt_' + str(rank) + '/')
            ckpoint_cb = ModelCheckpoint(prefix='mask_rcnn', directory=save_checkpoint_path, config=ckptconfig)
            cb += [ckpoint_cb]

        model = Model(net)
        start_time = time.time()
        model.train(config.epoch_size, dataset, callbacks=cb)
        end_time = time.time()
        print("total cost time :",end_time-start_time)
