# # Copyright 2020 Huawei Technologies Co., Ltd
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# # http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ============================================================================
"""train."""
import argparse
from mindspore import context
from mindspore.communication.management import init
from mindspore.nn.optim.momentum import Momentum
from mindspore import Model, ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import Callback, CheckpointConfig, ModelCheckpoint, TimeMonitor
from src.md_dataset import create_dataset
from src.losses import OhemLoss
from src.deeplabv3 import deeplabv3_resnet50
from src.config import config
from src.miou_precision import MiouPrecision

parser = argparse.ArgumentParser(description="Deeplabv3 training")
parser.add_argument("--distribute", type=str, default="false", help="Run distribute, default is false.")
parser.add_argument('--data_url', required=True, default=None, help='Train data url')
parser.add_argument('--train_url', required=True, default=None, help='Train data output url')
parser.add_argument('--checkpoint_url', default=None, help='Checkpoint path')
args_opt = parser.parse_args()
print(args_opt)
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend") #无需指定DEVICE_ID

data_path = "./voc2012"
train_checkpoint_path = "./checkpoint/deeplabv3_train_14-1_1.ckpt" #预训练的ckpt
eval_checkpoint_path = "./checkpoint_deeplabv3-%s_732.ckpt" % config.epoch_size #训练结束存的ckpt

class LossCallBack(Callback):
    """
    Monitor the loss in training.
    Note:
        if per_print_times is 0 do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """
    def __init__(self, per_print_times=1):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0")
        self._per_print_times = per_print_times
    def step_end(self, run_context):
        cb_params = run_context.original_args()
        print("epoch: {}, step: {}, outputs are {}".format(cb_params.cur_epoch_num, cb_params.cur_step_num,
                                                           str(cb_params.net_outputs)))
def model_fine_tune(flags, train_net, fix_weight_layer):
    path = flags.checkpoint_url
    if path is None:
        return
    path = train_checkpoint_path
    param_dict = load_checkpoint(path)
    load_param_into_net(train_net, param_dict)
    for para in train_net.trainable_params():
        if fix_weight_layer in para.name:
            para.requires_grad = False

if __name__ == "__main__":
    if args_opt.distribute == "true":
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, mirror_mean=True)
        init()
    args_opt.base_size = config.crop_size
    args_opt.crop_size = config.crop_size

    import moxing as mox
    mox.file.copy_parallel(src_url=args_opt.data_url, dst_url='voc2012/')
    mox.file.copy_parallel(src_url=args_opt.checkpoint_url, dst_url='checkpoint/')

    # train
    train_dataset = create_dataset(args_opt, data_path, config.epoch_size, config.batch_size, usage="train")
    dataset_size = train_dataset.get_dataset_size()
    time_cb = TimeMonitor(data_size=dataset_size)
    callback = [time_cb, LossCallBack()]
    if config.enable_save_ckpt:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                                     keep_checkpoint_max=config.save_checkpoint_num)
        ckpoint_cb = ModelCheckpoint(prefix='checkpoint_deeplabv3', config=config_ck)
        callback.append(ckpoint_cb)
    net = deeplabv3_resnet50(config.seg_num_classes, [config.batch_size, 3, args_opt.crop_size, args_opt.crop_size],
                             infer_scale_sizes=config.eval_scales, atrous_rates=config.atrous_rates,
                             decoder_output_stride=config.decoder_output_stride, output_stride=config.output_stride,
                             fine_tune_batch_norm=config.fine_tune_batch_norm, image_pyramid=config.image_pyramid)
    net.set_train()
    model_fine_tune(args_opt, net, 'layer')
    loss = OhemLoss(config.seg_num_classes, config.ignore_label)
    opt = Momentum(filter(lambda x: 'beta' not in x.name and 'gamma' not in x.name and 'depth' not in x.name and 'bias' not in x.name, net.trainable_params()), learning_rate=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    model = Model(net, loss, opt)
    model.train(config.epoch_size, train_dataset, callback)

    # eval
    eval_dataset = create_dataset(args_opt, data_path, config.epoch_size, config.batch_size, usage="eval")
    net = deeplabv3_resnet50(config.seg_num_classes, [config.batch_size, 3, args_opt.crop_size, args_opt.crop_size],
                             infer_scale_sizes=config.eval_scales, atrous_rates=config.atrous_rates,
                             decoder_output_stride=config.decoder_output_stride, output_stride=config.output_stride,
                             fine_tune_batch_norm=config.fine_tune_batch_norm, image_pyramid=config.image_pyramid)

    param_dict = load_checkpoint(eval_checkpoint_path)
    load_param_into_net(net, param_dict)
    mIou = MiouPrecision(config.seg_num_classes)
    metrics = {'mIou': mIou}
    loss = OhemLoss(config.seg_num_classes, config.ignore_label)
    model = Model(net, loss, metrics=metrics)
    model.eval(eval_dataset)