import os
import csv
import numpy as np
import seaborn as sns
import mindspore as ms
import matplotlib.pyplot as plt

from mindspore import nn, context
from mindspore.ops import operations as ops
context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

def create_dataset(data_path):
    with open(data_path) as csv_file:
        data = list(csv.reader(csv_file, delimiter=','))

    label_map = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }
    X = np.array([[float(x) for x in s[:-1]] for s in data[:150]], np.float32)
    Y = np.array([label_map[s[-1]] for s in data[:150]], np.int32)
    return X,Y


class Pca(nn.Cell):

    def __init__(self):
        super(Pca, self).__init__()
        self.reduce_mean = ops.ReduceMean(keep_dims=True)
        self.reshape = ops.Reshape()
        self.matmul_a = ops.MatMul(transpose_a=True)
        self.matmul_b = ops.MatMul(transpose_b=True)
        self.top_k = ops.TopK(sorted=True)
        self.gather = ops.GatherV2()

    def construct(self, x, dim=2):
        '''
        x:输入矩阵
        dim:降维之后的维度数
        '''
        X,Y = create_dataset(data_path)
        m = X.shape[0]
        # 计算张量的各个维度上的元素的平均值
        mean = self.reduce_mean(x, axis=1)
        # 去中心化
        x_new = x - self.reshape(mean, (-1, 1))
        # 无偏差的协方差矩阵
        cov = self.matmul_a(x_new, x_new) / (m - 1)
        # 计算特征分解
        cov = cov.asnumpy()
        e, v = np.linalg.eigh(cov)
        # 将特征值从大到小排序，选出前dim个的index
        e_index_sort = self.top_k(ms.Tensor(e), dim)[1]
        # 提取前排序后dim个特征向量
        v_new = self.gather(ms.Tensor(v), e_index_sort, 0)
        # 降维操作
        pca = self.matmul_b(x_new, v_new)
        return pca

def pca(X,Y):
    net = Pca()
    pca_data_tensor = ms.Tensor(np.reshape(X,(X.shape[0],-1)),ms.float32)
    pca_data = net(pca_data_tensor,dim=2)

    # plt
    color_mapping = {0: sns.xkcd_rgb['bright purple'],1: sns.xkcd_rgb['pale red'], 2: sns.xkcd_rgb['ochre']}
    colors = list(map(lambda x: color_mapping[x], Y))
    plt.scatter(pca_data[:, 0].asnumpy(), pca_data[:, 1].asnumpy(), c=colors)
    plt.show()

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

    pca(*create_dataset(data_path))
