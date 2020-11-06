import os
# os.environ['DEVICE_ID'] = '0'
import csv
import numpy as np

import mindspore as ms
from mindspore import nn
from mindspore import context
from mindspore import dataset
from mindspore.train.callback import LossMonitor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


def create_dataset(data_path):
    with open(data_path) as csv_file:
        data = list(csv.reader(csv_file, delimiter=','))

    label_map = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
    }
    # 取前两类的样本数据
    X = np.array([[float(x) for x in s[:-1]] for s in data[:100]], np.float32)
    Y = np.array([[label_map[s[-1]]] for s in data[:100]], np.float32)

    # 将数据集按8:2划分为训练集和测试集
    train_idx = np.random.choice(100, 80, replace=False)
    test_idx = np.array(list(set(range(100)) - set(train_idx)))
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    # 对数据进行处理通常是有益的，但对于这个简单的任务是不必要的
    # def normalize(data):
    #     v_max = np.max(data, axis=0)
    #     v_min = np.min(data, axis=0)
    #     return np.divide(data - v_min, v_max - v_min) * 2.0 - 1.0
    # train_data = list(zip(normalize(X_train), Y_train))
    XY_train = list(zip(X_train, Y_train))
    ds_train = dataset.GeneratorDataset(XY_train, ['x', 'y'])
    ds_train = ds_train.shuffle(buffer_size=80).batch(32, drop_remainder=True)

    return ds_train, X_test, Y_test


# 自定义Loss
class Loss(nn.Cell):
    def __init__(self):
        super(Loss, self).__init__()
        self.sigmoid_cross_entropy_with_logits = P.SigmoidCrossEntropyWithLogits()
        self.reduce_mean = P.ReduceMean(keep_dims=False)
    def construct(self, z, y):
        loss = self.sigmoid_cross_entropy_with_logits(z, y)
        return self.reduce_mean(loss, -1)


def logistic_regression(ds_train, X_test, Y_test):
    net = nn.Dense(4, 1)
    loss = Loss()
    opt = nn.optim.SGD(net.trainable_params(), learning_rate=0.003)

    model = ms.train.Model(net, loss, opt)
    model.train(5, ds_train, callbacks=[LossMonitor(per_print_times=ds_train.get_dataset_size())], dataset_sink_mode=False)

    # 计算测试集上的精度
    x = model.predict(ms.Tensor(X_test)).asnumpy()
    pred = np.round(1 / (1 + np.exp(-x)))
    correct = np.equal(pred, Y_test)
    acc = np.mean(correct)
    print('Test accuracy is', acc)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    args, unknown = parser.parse_known_args()

    if args.data_url.startswith('s3'):
        data_path = 'iris.data'
        import moxing
        moxing.file.copy_parallel(src_url=os.path.join(args.data_url, 'iris.data'), dst_url=data_path)
    else:
        data_path = os.path.abspath(args.data_url)

    logistic_regression(*create_dataset(data_path))
