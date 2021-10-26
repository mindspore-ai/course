'''
Date: 2021-08-15 16:09:48
LastEditors: xgy
LastEditTime: 2021-09-14 14:34:03
FilePath: \code\ctpn\train.py
'''

"""train CTPN and get checkpoint files."""
import os
import time
import argparse
import ast
import mindspore.common.dtype as mstype
from mindspore import context, Tensor
from mindspore.communication.management import init
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn import Momentum
from mindspore.common import set_seed
from src.ctpn import CTPN
from src.config import config, pretrain_config, finetune_config
from src.dataset import create_ctpn_dataset
from src.lr_schedule import dynamic_lr
from src.network_define import LossCallBack, LossNet, WithLossCell, TrainOneStepCell

set_seed(1)

parser = argparse.ArgumentParser(description="CTPN training")
parser.add_argument("--pre_trained", type=str, default="", help="Pretrained file path.")
parser.add_argument("--task_type", type=str, default="Pretraining",\
    choices=['Pretraining', 'Finetune'], help="task type, default:Pretraining")
args_opt = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args_opt.device_id, save_graphs=True)

if __name__ == '__main__':
    
    rank = 0
    device_num = 1
    args_opt.run_distribute = False
    
    if args_opt.task_type == "Pretraining":
        print("Start to do pretraining")
        mindrecord_file = config.pretraining_dataset_file
        training_cfg = pretrain_config
    else:
        print("Start to do finetune")
        mindrecord_file = config.finetune_dataset_file
        training_cfg = finetune_config

    print("CHECKING MINDRECORD FILES ...")
    while not os.path.exists(mindrecord_file + ".db"):
        time.sleep(5)

    print("CHECKING MINDRECORD FILES DONE!")

    loss_scale = float(config.loss_scale)

    # When create MindDataset, using the fitst mindrecord file, such as ctpn_pretrain.mindrecord0.
    dataset = create_ctpn_dataset(mindrecord_file, repeat_num=1,\
        batch_size=config.batch_size, device_num=device_num, rank_id=rank)
    dataset_size = dataset.get_dataset_size()
    net = CTPN(config=config, is_training=True)
    net = net.set_train()

    load_path = args_opt.pre_trained
    if args_opt.task_type == "Pretraining":
        pass
        # You can choose whether to use pretrained model file
        # print("load backbone vgg16 ckpt {}".format(args_opt.pre_trained))
        # param_dict = load_checkpoint(load_path)
        # for item in list(param_dict.keys()):
        #     if not item.startswith('vgg16_feature_extractor'):
        #         param_dict.pop(item)
        # load_param_into_net(net, param_dict)
    else:
        if load_path != "":
            print("load pretrain ckpt {}".format(args_opt.pre_trained))
            param_dict = load_checkpoint(load_path)
            load_param_into_net(net, param_dict)
    loss = LossNet()
    lr = Tensor(dynamic_lr(training_cfg, dataset_size), mstype.float32)
    opt = Momentum(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum,\
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
        ckptconfig = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs*dataset_size,
                                      keep_checkpoint_max=config.keep_checkpoint_max)
        save_checkpoint_path = os.path.join(config.save_checkpoint_path, "ckpt_" + str(rank) + "/")
        ckpoint_cb = ModelCheckpoint(prefix='ctpn', directory=save_checkpoint_path, config=ckptconfig)
        cb += [ckpoint_cb]

    model = Model(net)
    # model.train(training_cfg.total_epoch, dataset, callbacks=cb, dataset_sink_mode=True)
    model.train(training_cfg.total_epoch, dataset, callbacks=cb, dataset_sink_mode=False)
