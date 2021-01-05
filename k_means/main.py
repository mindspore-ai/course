import os
import csv
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import mindspore as ms
from mindspore import context, Tensor, nn
from mindspore.ops import operations as ops

context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
args, unknown = parser.parse_known_args()

import moxing
# src_url形如's3://OBS/PATH'，为OBS桶中数据集的路径，dst_url为执行容器中的路径，两者皆为目录/皆为文件
moxing.file.copy_parallel(src_url=os.path.join(args.data_url, 'iris.data'), dst_url='iris.data')

def create_dataset(data_path):
    with open(data_path) as csv_file:
        data = list(csv.reader(csv_file, delimiter=','))

    label_map = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }
    X = np.array([[float(x) for x in s[2:-1]] for s in data[:150]], np.float32)
    Y = np.array([label_map[s[-1]] for s in data[:150]], np.int32)
    return X,Y

# 设置K值为3。iris数据集有三类花
# 实际上是分类任务，因为已经给了堆大小了
# 迭代次数25
k=3
generations = 100

x,y = create_dataset('./iris.data')
num_pts = len(x)
num_feats = len(x[0])
data_points = Tensor(x)
cluster_labels = np.zeros(num_pts)
# 先随机选择iris数据集中的三个数据点作为每个堆的中心点
rand_starts = np.array([x[np.random.choice(len(x))] for _ in range(k)])
centroids = Tensor(rand_starts)


def calculate():
    # 计算每个数据点到每个中心点的欧氏距离
    # 这里是将数据点都放入矩阵，直接按矩阵进行运算
    reshape = ops.Reshape()
    tile = ops.Tile()
    reduce_sum = ops.ReduceSum(keep_dims=False)
    square = ops.Square()
    argmin = ops.Argmin()

    centroid_matrix = reshape(tile(centroids, (num_pts, 1)), (num_pts, k, num_feats))
    point_matrix = reshape(tile(data_points, (1, k)), (num_pts, k, num_feats))
    distances = reduce_sum(square(point_matrix - centroid_matrix), 2)
    centroid_group = argmin(distances)

    return centroid_group

# 计算三个堆的平均距离更新堆中新的中心点
unsorted_segment_sum = ops.UnsortedSegmentSum()
ones_like = ops.OnesLike()

def data_group_avg(group_ids, data):
    # 分组求和
    sum_total = unsorted_segment_sum(data, group_ids, 3)
    #计算堆大小
    num_total = unsorted_segment_sum(ones_like(data), group_ids, 3)
    #求距离均值
    avg_by_group = sum_total/num_total
    return avg_by_group

assign = ops.Assign()
# 遍历循环训练，更新每组分类的中心点
for i in range(generations):
    print('Calculating gen {}'.format(i))
    centroid_group = calculate()
    means = data_group_avg(centroid_group, data_points)
    centroids = assign(ms.Parameter(centroids, name='w'), means)
    cluster_labels = assign(ms.Parameter(Tensor(cluster_labels,ms.int64),name='w'), centroid_group)
    centroid_group_count = assign(ms.Parameter(Tensor(cluster_labels,ms.int64),name='w'), centroid_group).asnumpy()
    group_count = []
#     print(centroid_group_count)
    for ix in range(k):
        group_count.append(np.sum(centroid_group_count==ix))
    print('Group counts: {}'.format(group_count))

# 输出准确率。
# 聚类结果和iris数据集中的标签进行对比
centers, assignments = centroids, cluster_labels.asnumpy()

def most_common(my_list):
    return(max(set(my_list), key=my_list.count))

label0 = most_common(list(assignments[0:50]))
label1 = most_common(list(assignments[50:100]))
label2 = most_common(list(assignments[100:150]))
group0_count = np.sum(assignments[0:50]==label0)
group1_count = np.sum(assignments[50:100]==label1)
group2_count = np.sum(assignments[100:150]==label2)
accuracy = (group0_count + group1_count + group2_count)/150.
print('Accuracy: {:.2}'.format(accuracy))

reduced_data = x
reduced_centers = means

# 设置图例
symbols = ['o']
label_name = ['Setosa', 'Versicolour', 'Virginica']
for i in range(3):
    temp_group = reduced_data[(i*50):(50)*(i+1)]
    plt.plot(temp_group[:, 0], temp_group[:, 1], symbols[0], markersize=8, label=label_name[i])
# 绘图
plt.scatter(reduced_centers[:, 0].asnumpy(), reduced_centers[:, 1].asnumpy(),
            marker='x', s=169, linewidths=3,color='red', zorder=10)
plt.title('K-means clustering on Iris Dataset\n'
          'Centroids are marked with white cross')
plt.legend(loc='lower right')
plt.show()
print('Successful !!!')