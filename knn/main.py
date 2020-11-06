# KNN

import os
# os.environ['DEVICE_ID'] = '4'
import csv
import numpy as np

import mindspore as ms
from mindspore import context
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


def create_dataset(data_path):
    with open(data_path) as csv_file:
        data = list(csv.reader(csv_file, delimiter=','))
    print(data[56:62]+data[130:133]) # print some samples

    X = np.array([[float(x) for x in s[1:]] for s in data[:178]], np.float32)
    Y = np.array([s[0] for s in data[:178]], np.int32)

    train_idx = np.random.choice(178, 128, replace=False)
    test_idx = np.array(list(set(range(178)) - set(train_idx)))
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    return X_train, Y_train, X_test, Y_test


class KnnNet(nn.Cell):
    def __init__(self, k):
        super(KnnNet, self).__init__()
        self.tile = P.Tile()
        self.sum = P.ReduceSum()
        self.topk = P.TopK()
        self.k = k

    def construct(self, x, X_train):
        # Tile input x to match the number of samples in X_train
        x_tile = self.tile(x, (128, 1))
        square_diff = F.square(x_tile - X_train)
        square_dist = self.sum(square_diff, 1)
        dist = F.sqrt(square_dist)
        # -dist mean the bigger the value is, the nearer the samples are
        values, indices = self.topk(-dist, self.k)
        return indices


def knn(knn_net, x, X_train, Y_train):
    x, X_train = ms.Tensor(x), ms.Tensor(X_train)
    indices = knn_net(x, X_train)
    topk_cls = [0]*len(indices.asnumpy())
    for idx in indices.asnumpy():
        topk_cls[Y_train[idx]] += 1
    cls = np.argmax(topk_cls)
    return cls


def test_knn(X_train, Y_train, X_test, Y_test):
    acc = 0
    knn_net = KnnNet(5)
    for x, y in zip(X_test, Y_test):
        pred = knn(knn_net, x, X_train, Y_train)
        acc += (pred == y)
        print('label: %d, prediction: %s' % (y, pred))
    print('Validation accuracy is %f' % (acc/len(Y_test)))


"""
# Code for PyNative mode, P.TopK is not supported
def knn(x, X_train, Y_train, k):
    x, X_train = ms.Tensor(x), ms.Tensor(X_train)
    # Tile input x to match the number of samples in X_train
    x_tile = P.Tile()(x, (X_train.shape[0], 1))
    square_diff = F.square(x_tile - X_train)
    square_dist = P.ReduceSum()(square_diff, axis=1)
    dist = F.sqrt(square_dist)
    # -dist mean the bigger the value is, the nearer the samples are
    values, indices = P.TopK()(-dist, k)
    topk_cls = [0]*len(indices.asnumpy())
    for idx in indices.asnumpy():
        topk_cls[Y_train[idx]] += 1
    cls = np.argmax(topk_cls)
    return cls
def test_knn(X_train, Y_train, X_test, Y_test):
    acc = 0
    for x, y in zip(X_test, Y_test):
        pred = knn(x, X_train, Y_train, 5)
        acc += (pred == y)
        print('sample: %s, label: %d, prediction: %s' % (x, y, pred))
    print('Validation accuracy is %f' % (acc/len(Y_test)))
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    args, unknown = parser.parse_known_args()

    if args.data_url.startswith('s3'):
        data_path = 'wine.data'
        import moxing
        moxing.file.copy_parallel(src_url=os.path.join(args.data_url, 'wine.data'), dst_url=data_path)
    else:
        data_path = os.path.abspath(args.data_url)

    test_knn(*create_dataset(data_path))
