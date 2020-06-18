# LeNet5 mnist
import matplotlib.pyplot as plt

import mindspore as ms
import mindspore.context as context
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.transforms.vision.c_transforms as CV

from mindspore import nn
from mindspore.model_zoo.lenet import LeNet5
from mindspore.train import Model
from mindspore.train.callback import LossMonitor

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

DATA_DIR_TRAIN = "MNIST/train"  # 训练集信息
DATA_DIR_TEST = "MNIST/test"  # 测试集信息


def create_dataset(training=True, num_epoch=1, batch_size=32, resize=(32, 32),
                   rescale=1 / (255 * 0.3081), shift=-0.1307 / 0.3081, buffer_size=64):
    ds = ms.dataset.MnistDataset(DATA_DIR_TRAIN if training else DATA_DIR_TEST)

    ds = ds.map(input_columns="image", operations=[CV.Resize(resize), CV.Rescale(rescale, shift), CV.HWC2CHW()])
    ds = ds.map(input_columns="label", operations=C.TypeCast(ms.int32))
    ds = ds.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True).repeat(num_epoch)

    return ds


def test_train(lr=0.01, momentum=0.9, num_epoch=3, ckpt_name="a_lenet"):
    ds_train = create_dataset(num_epoch=num_epoch)
    ds_eval = create_dataset(training=False)

    net = LeNet5()
    loss = nn.loss.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction='mean')
    opt = nn.Momentum(net.trainable_params(), lr, momentum)

    loss_cb = LossMonitor(per_print_times=1)

    model = Model(net, loss, opt, metrics={'acc', 'loss'})
    model.train(num_epoch, ds_train, callbacks=[loss_cb], dataset_sink_mode=False)
    metrics = model.eval(ds_eval, dataset_sink_mode=False)
    print('Metrics:', metrics)

def show_dataset():
    ds = create_dataset(training=False)
    data = ds.create_dict_iterator().get_next()
    images = data['image']
    labels = data['label']

    for i in range(1, 5):
        plt.subplot(2, 2, i)
        plt.imshow(images[i][0])
        plt.title('Number: %s' % labels[i])
        plt.xticks([])
    plt.show()

if __name__ == "__main__":
    show_dataset()

    test_train()