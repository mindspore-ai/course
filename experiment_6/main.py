# Save and load model

import matplotlib.pyplot as plt
import numpy as np

import mindspore as ms
import mindspore.context as context
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.transforms.vision.c_transforms as CV

from mindspore import nn, Tensor
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

DATA_DIR_TRAIN = "MNIST/train"  # 训练集信息
DATA_DIR_TEST = "MNIST/test"  # 测试集信息


def create_dataset(training=True, num_epoch=1, batch_size=32, resize=(32, 32),
                   rescale=1 / (255 * 0.3081), shift=-0.1307 / 0.3081, buffer_size=64):
    ds = ms.dataset.MnistDataset(DATA_DIR_TRAIN if training else DATA_DIR_TEST)

    # define map operations
    resize_op = CV.Resize(resize)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()

    # apply map operations on images
    ds = ds.map(input_columns="image", operations=[resize_op, rescale_op, hwc2chw_op])
    ds = ds.map(input_columns="label", operations=C.TypeCast(ms.int32))

    ds = ds.shuffle(buffer_size=buffer_size)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.repeat(num_epoch)

    return ds


class LeNet5(nn.Cell):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, pad_mode='valid')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(400, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 10)

    def construct(self, input_x):
        output = self.conv1(input_x)
        output = self.relu(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.pool(output)
        output = self.flatten(output)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output


def test_train(lr=0.01, momentum=0.9, num_epoch=2, check_point_name="b_lenet"):
    ds_train = create_dataset(num_epoch=num_epoch)
    ds_eval = create_dataset(training=False)
    steps_per_epoch = ds_train.get_dataset_size()

    net = LeNet5()
    loss = nn.loss.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction='mean')
    opt = nn.Momentum(net.trainable_params(), lr, momentum)

    ckpt_cfg = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=5)
    ckpt_cb = ModelCheckpoint(prefix=check_point_name, config=ckpt_cfg)
    loss_cb = LossMonitor(steps_per_epoch)

    model = Model(net, loss, opt, metrics={'acc', 'loss'})
    model.train(num_epoch, ds_train, callbacks=[ckpt_cb, loss_cb], dataset_sink_mode=False)
    metrics = model.eval(ds_eval, dataset_sink_mode=False)
    print('Metrics:', metrics)


CKPT = 'b_lenet-2_1875.ckpt'


def resume_train(lr=0.001, momentum=0.9, num_epoch=2, ckpt_name="b_lenet"):
    ds_train = create_dataset(num_epoch=num_epoch)
    ds_eval = create_dataset(training=False)
    steps_per_epoch = ds_train.get_dataset_size()

    net = LeNet5()
    loss = nn.loss.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction='mean')
    opt = nn.Momentum(net.trainable_params(), lr, momentum)

    param_dict = load_checkpoint(CKPT)
    load_param_into_net(net, param_dict)
    load_param_into_net(opt, param_dict)

    ckpt_cfg = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=5)
    ckpt_cb = ModelCheckpoint(prefix=ckpt_name, config=ckpt_cfg)
    loss_cb = LossMonitor(steps_per_epoch)

    model = Model(net, loss, opt, metrics={'acc', 'loss'})
    model.train(num_epoch, ds_train, callbacks=[ckpt_cb, loss_cb], dataset_sink_mode=False)
    metrics = model.eval(ds_eval, dataset_sink_mode=False)
    print('Metrics:', metrics)

def plot_images(pred_fn, ds, net):
    for i in range(1, 5):
        pred, image, label = pred_fn(ds, net)
        plt.subplot(2, 2, i)
        plt.imshow(np.squeeze(image))
        color = 'blue' if pred == label else 'red'
        plt.title("prediction: {}, truth: {}".format(pred, label), color=color)
        plt.xticks([])
    plt.show()

CKPT = 'b_lenet_1-2_1875.ckpt'

def infer(ds, model):
    data = ds.get_next()
    images = data['image']
    labels = data['label']
    output = model.predict(Tensor(data['image']))
    pred = np.argmax(output.asnumpy(), axis=1)
    return pred[0], images[0], labels[0]

def test_infer():
    ds = create_dataset(training=False, batch_size=1).create_dict_iterator()
    net = LeNet5()
    param_dict = load_checkpoint(CKPT, net)
    model = Model(net)
    plot_images(infer, ds, model)

if __name__ == "__main__":

    test_train()

    resume_train()

    test_infer()