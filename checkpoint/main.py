# Save and load model

import os
# os.environ['DEVICE_ID'] = '0'
# Log level includes 3(ERROR), 2(WARNING), 1(INFO), 0(DEBUG).
os.environ['GLOG_v'] = '2'

import matplotlib.pyplot as plt
import numpy as np

import mindspore as ms
import mindspore.context as context
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV

from mindspore import nn, Tensor
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

import logging; logging.getLogger('matplotlib.font_manager').disabled = True

context.set_context(mode=context.GRAPH_MODE, device_target='Ascend') # Ascend, CPU, GPU


def create_dataset(data_dir, training=True, batch_size=32, resize=(32, 32),
                   rescale=1/(255*0.3081), shift=-0.1307/0.3081, buffer_size=64):
    data_train = os.path.join(data_dir, 'train') # train set
    data_test = os.path.join(data_dir, 'test') # test set
    ds = ms.dataset.MnistDataset(data_train if training else data_test)

    ds = ds.map(input_columns=["image"], operations=[CV.Resize(resize), CV.Rescale(rescale, shift), CV.HWC2CHW()])
    ds = ds.map(input_columns=["label"], operations=C.TypeCast(ms.int32))
    ds = ds.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)

    return ds


class LeNet5(nn.Cell):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, pad_mode='valid')
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(400, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 10)

    def construct(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


def train(data_dir, lr=0.01, momentum=0.9, num_epochs=2, ckpt_name="lenet"):
    dataset_sink = context.get_context('device_target') == 'Ascend'
    repeat = num_epochs if dataset_sink else 1
    ds_train = create_dataset(data_dir)
    ds_eval = create_dataset(data_dir, training=False)
    steps_per_epoch = ds_train.get_dataset_size()

    net = LeNet5()
    loss = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    opt = nn.Momentum(net.trainable_params(), lr, momentum)

    ckpt_cfg = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=5)
    ckpt_cb = ModelCheckpoint(prefix=ckpt_name, directory='ckpt', config=ckpt_cfg)
    loss_cb = LossMonitor(steps_per_epoch)

    model = Model(net, loss, opt, metrics={'acc', 'loss'})
    model.train(num_epochs, ds_train, callbacks=[ckpt_cb, loss_cb], dataset_sink_mode=dataset_sink)
    metrics = model.eval(ds_eval, dataset_sink_mode=dataset_sink)
    print('Metrics:', metrics)


CKPT_1 = 'ckpt/lenet-2_1875.ckpt'

def resume_train(data_dir, lr=0.001, momentum=0.9, num_epochs=2, ckpt_name="lenet"):
    dataset_sink = context.get_context('device_target') == 'Ascend'
    ds_train = create_dataset(data_dir)
    ds_eval = create_dataset(data_dir, training=False)
    steps_per_epoch = ds_train.get_dataset_size()

    net = LeNet5()
    loss = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    opt = nn.Momentum(net.trainable_params(), lr, momentum)

    param_dict = load_checkpoint(CKPT_1)
    load_param_into_net(net, param_dict)
    load_param_into_net(opt, param_dict)

    ckpt_cfg = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=5)
    ckpt_cb = ModelCheckpoint(prefix=ckpt_name, directory='ckpt', config=ckpt_cfg)
    loss_cb = LossMonitor(steps_per_epoch)

    model = Model(net, loss, opt, metrics={'acc', 'loss'})
    model.train(num_epochs, ds_train, callbacks=[ckpt_cb, loss_cb], dataset_sink_mode=dataset_sink)

    metrics = model.eval(ds_eval, dataset_sink_mode=dataset_sink)
    print('Metrics:', metrics)


CKPT_2 = 'ckpt/lenet_1-2_1875.ckpt'

def infer(data_dir):
    ds = create_dataset(data_dir, training=False).create_dict_iterator(output_numpy=True)
    data = ds.get_next()
    images = data['image']
    labels = data['label']
    net = LeNet5()
    load_checkpoint(CKPT_2, net=net)
    model = Model(net)
    output = model.predict(Tensor(data['image']))
    preds = np.argmax(output.asnumpy(), axis=1)

    for i in range(1, 5):
        plt.subplot(2, 2, i)
        plt.imshow(np.squeeze(images[i]))
        color = 'blue' if preds[i] == labels[i] else 'red'
        plt.title("prediction: {}, truth: {}".format(preds[i], labels[i]), color=color)
        plt.xticks([])
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_url', required=False, default='MNIST/', help='Location of data.')
    parser.add_argument('--train_url', required=False, default=None, help='Location of training outputs.')
    args, unknown = parser.parse_known_args()

    COPY_OTHER = False
    if args.data_url.startswith('s3'):
        import moxing

        # WAY1: copy dataset from your own OBS bucket to container/cache.
        # moxing.file.copy_parallel(src_url=args.data_url, dst_url='MNIST/')

        # WAY2: copy dataset from other's OBS bucket, which has been set public read or public read&write.
        moxing.file.copy_parallel(src_url="s3://share-course/dataset/MNIST/", dst_url='MNIST/')

        data_path = 'MNIST/'
    else:
        data_path = os.path.abspath(args.data_url)

    # Please remove stale checkpoint folder `ckpt`
    train(data_path)
    print('Checkpoints after first training:')
    print('\n'.join(sorted([x for x in os.listdir('ckpt') if x.startswith('lenet')])))

    resume_train(data_path)
    print('Checkpoints after resuming training:')
    print('\n'.join(sorted([x for x in os.listdir('ckpt') if x.startswith('lenet')])))

    infer(data_path)
    if args.data_url.startswith('s3'):
        import moxing
        # Copy the ckpt directory to OBS, then there will be a ckpt directory in the ʻargs.train_url` directory in OBS.
        moxing.file.copy_parallel(src_url='ckpt', dst_url=os.path.join(args.train_url, 'ckpt'))
        print('Copied checkpoints from ./ckpt to %s' % os.path.join(args.train_url, 'ckpt'))
