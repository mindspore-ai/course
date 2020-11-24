import os
import argparse
import ast
import numpy as np

import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as F
from mindspore import Model, context
from mindspore.nn.loss.loss import _Loss
from mindspore.communication.management import init, get_group_size
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.unet import UNet
from src.data_loader import create_dataset
from src.loss import CrossEntropyWithLogits
from src.utils import StepLossTimeMonitor
from src.config import cfg_unet
from scipy.special import softmax

device_id = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False, device_id=device_id)

mindspore.set_seed(1)


class CrossEntropyWithLogits(_Loss):
    def __init__(self):
        super(CrossEntropyWithLogits, self).__init__()
        self.transpose_fn = F.Transpose()
        self.reshape_fn = F.Reshape()
        self.softmax_cross_entropy_loss = nn.SoftmaxCrossEntropyWithLogits()
        self.cast = F.Cast()

    def construct(self, logits, label):
        # NCHW->NHWC
        logits = self.transpose_fn(logits, (0, 2, 3, 1))
        logits = self.cast(logits, mindspore.float32)
        label = self.transpose_fn(label, (0, 2, 3, 1))

        loss = self.reduce_mean(self.softmax_cross_entropy_loss(self.reshape_fn(logits, (-1, 2)),
                                                                self.reshape_fn(label, (-1, 2))))
        return self.get_loss(loss)


class dice_coeff(nn.Metric):
    def __init__(self):
        super(dice_coeff, self).__init__()
        self.clear()

    def clear(self):
        self._dice_coeff_sum = 0
        self._samples_num = 0

    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError('Mean dice coeffcient need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))

        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        self._samples_num += y.shape[0]
        y_pred = y_pred.transpose(0, 2, 3, 1)
        y = y.transpose(0, 2, 3, 1)
        y_pred = softmax(y_pred, axis=3)

        inter = np.dot(y_pred.flatten(), y.flatten())
        union = np.dot(y_pred.flatten(), y_pred.flatten()) + np.dot(y.flatten(), y.flatten())

        single_dice_coeff = 2 * float(inter) / float(union + 1e-6)
        print("single dice coeff is:", single_dice_coeff)
        self._dice_coeff_sum += single_dice_coeff

    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return self._dice_coeff_sum / float(self._samples_num)


class dice_coeff(nn.Metric):
    def __init__(self):
        super(dice_coeff, self).__init__()
        self.clear()

    def clear(self):
        self._dice_coeff_sum = 0
        self._samples_num = 0

    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError('Mean dice coeffcient need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))

        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        self._samples_num += y.shape[0]
        y_pred = y_pred.transpose(0, 2, 3, 1)
        y = y.transpose(0, 2, 3, 1)
        y_pred = softmax(y_pred, axis=3)

        inter = np.dot(y_pred.flatten(), y.flatten())
        union = np.dot(y_pred.flatten(), y_pred.flatten()) + np.dot(y.flatten(), y.flatten())

        single_dice_coeff = 2 * float(inter) / float(union + 1e-6)
        print("single dice coeff is:", single_dice_coeff)
        self._dice_coeff_sum += single_dice_coeff

    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return self._dice_coeff_sum / float(self._samples_num)


def train_net(data_dir, cross_valid_ind=1, epochs=400, batch_size=16, lr=0.0001, run_distribute=False, cfg=None):
    if run_distribute:
        init()
        group_size = get_group_size()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode,
                                          device_num=group_size,
                                          gradients_mean=False)
    net = UNet(n_channels=cfg['num_channels'], n_classes=cfg['num_classes'])

    if cfg['resume']:
        param_dict = load_checkpoint(cfg['resume_ckpt'])
        load_param_into_net(net, param_dict)

    criterion = CrossEntropyWithLogits()
    train_dataset, _ = create_dataset(data_dir, epochs, batch_size, True, cross_valid_ind, run_distribute)
    train_data_size = train_dataset.get_dataset_size()
    print("dataset length is:", train_data_size)
    ckpt_config = CheckpointConfig(save_checkpoint_steps=train_data_size,
                                   keep_checkpoint_max=cfg['keep_checkpoint_max'])
    ckpoint_cb = ModelCheckpoint(prefix='ckpt_unet_medical_adam',
                                 directory='./ckpt_{}/'.format(device_id),
                                 config=ckpt_config)

    optimizer = nn.Adam(params=net.trainable_params(), learning_rate=lr, weight_decay=cfg['weight_decay'],
                        loss_scale=cfg['loss_scale'])

    loss_scale_manager = mindspore.train.loss_scale_manager.FixedLossScaleManager(cfg['FixedLossScaleManager'], False)

    model = Model(net, loss_fn=criterion, loss_scale_manager=loss_scale_manager, optimizer=optimizer, amp_level="O3")

    print("============== Starting Training ==============")
    model.train(1, train_dataset, callbacks=[StepLossTimeMonitor(batch_size=batch_size), ckpoint_cb],
                dataset_sink_mode=False)
    print("============== End Training ==============")


def test_net(data_dir, ckpt_path, cross_valid_ind=1, cfg=None):
    net = UNet(n_channels=cfg['num_channels'], n_classes=cfg['num_classes'])
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)

    criterion = CrossEntropyWithLogits()
    _, valid_dataset = create_dataset(data_dir, 1, 1, False, cross_valid_ind, False)
    model = Model(net, loss_fn=criterion, metrics={"dice_coeff": dice_coeff()})

    print("============== Starting Evaluating ============")
    dice_score = model.eval(valid_dataset, dataset_sink_mode=False)
    print("Cross valid dice coeff is:", dice_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UNET')
    parser.add_argument('--data_url', required=True, help='Location of data.')
    parser.add_argument('--train_url', required=True, default=None, help='Location of training outputs.')
    args_opt = parser.parse_args()

    import moxing as mox
    # Copy dataset from OBS bucket to container/cache.
    mox.file.copy_parallel(src_url=args_opt.data_url, dst_url='./data')

    data_url = './data'
    ckpt_path = './ckpt_0/ckpt_unet_medical_adam-1_600.ckpt'
    run_distribute = False
    epoch_size = cfg_unet['epochs'] if not run_distribute else cfg_unet['distribute_epochs']

    train_net(data_dir=data_url,
              cross_valid_ind=cfg_unet['cross_valid_ind'],
              epochs=epoch_size,
              batch_size=cfg_unet['batchsize'],
              lr=cfg_unet['lr'],
              run_distribute=run_distribute,
              cfg=cfg_unet)

    print('*' * 60)

    test_net(data_dir=data_url,
             ckpt_path=ckpt_path,
             cross_valid_ind=cfg_unet['cross_valid_ind'],
             cfg=cfg_unet)