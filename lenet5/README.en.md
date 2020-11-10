# Handwritten Digit Recognition

## Introduction

LeNet5 + MNIST is known as the "Hello World" in the deep learning domain. This experiment describes how to use MindSpore to develop and train a LeNet5 model based on the MNIST handwritten digits dataset and validate the model accuracy.

## Objectives

Learn how to use MindSpore to develop a simple convolutional neural network.
Learn how to use MindSpore to train simple image classification tasks.
Learn how to use MindSpore to validate a simple image classification task.

## Prerequisites

Be proficient in Python and understand the basic knowledge of Shell and Linux operating systems.
Have certain theoretical knowledge of deep learning, such as convolutional neural networks, loss functions, optimizers, and training strategies.

## Environment

MindSpore 1.0.0 CPU and third-party auxiliary modules:

- MindSpore: https://www.mindspore.cn/install/en
- Jupyter Notebook/JupyterLab: https://jupyter.org/install
- MatplotLib: https://matplotlib.org/users/installing.html

MindSpore supports running on local CPU/GPU/Ascend environments, such as Windows/Ubuntu x64 notebooks, 
NVIDIA GPU servers, and Atlas Ascend servers. Before running the experiment in the local environment, 
you need to refer to [Installation Tutorial](https://www.mindspore.cn/install/) to install and configure the environment.

## Preparation

### Dataset Preparation

MNIST is a handwritten digits dataset. The training set contains 60,000 handwritten digits, and the test set contains 10,000 handwritten digits. The dataset contains 10 categories in total. MNIST official website:
http://yann.lecun.com/exdb/mnist/、

Download the following files from the MNIST official website to the local PC and decompress them:

    train-images-idx3-ubyte.gz:  training set images (9912422 bytes)
    train-labels-idx1-ubyte.gz:  training set labels (28881 bytes)
    t10k-images-idx3-ubyte.gz:  test set images (1648877 bytes)
    t10k-labels-idx1-ubyte.gz:  test set labels (4542 bytes)

### Script Preparation

Create a Jupyter Notebook and copy the code in the subsequent experiment steps to the Notebook for execution. 
Alternatively, download corresponding scripts from [mindspore/course](https://gitee.com/mindspore/course) and run the script in a Terminal. Organize the scripts and dataset as follows:

    lenet5
    ├── MNIST
    │   ├── test
    │   │   ├── t10k-images-idx3-ubyte
    │   │   └── t10k-labels-idx1-ubyte
    │   └── train
    │       ├── train-images-idx3-ubyte
    │       └── train-labels-idx1-ubyte
    └── main.ipynb # Alternatively, main.py

## Procedures(Notebook)

You can launch a Jupyter Notebook on local environments or try 
[ModelArts Notebook](https://support.huaweicloud.com/en-us/engineers-modelarts/modelarts_23_0032.html) on 
[HuaweiCloud ModelArts](https://www.huaweicloud.com/en-us/product/modelarts.html).

### Import modules

Import the MindSpore module and auxiliary modules. Set MindSpore context, such as execution mode and device platform.

```python
import os
# os.environ['DEVICE_ID'] = '0'

import mindspore as ms
import mindspore.context as context
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV

from mindspore import nn
from mindspore.train import Model
from mindspore.train.callback import LossMonitor

context.set_context(mode=context.GRAPH_MODE, device_target='CPU') # Ascend, CPU, GPU
```

### Data Processing

Before using a dataset to train a network, process the data as follows:

```python
def create_dataset(data_dir, training=True, batch_size=32, resize=(32, 32),
                   rescale=1/(255*0.3081), shift=-0.1307/0.3081, buffer_size=64):
    data_train = os.path.join(data_dir, 'train') # train set
    data_test = os.path.join(data_dir, 'test') # test set
    ds = ms.dataset.MnistDataset(data_train if training else data_test)

    ds = ds.map(input_columns=["image"], operations=[CV.Resize(resize), CV.Rescale(rescale, shift), CV.HWC2CHW()])
    ds = ds.map(input_columns=["label"], operations=C.TypeCast(ms.int32))
    # When `dataset_sink_mode=True` on Ascend, append `ds = ds.repeat(num_epochs) to the end
    ds = ds.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)

    return ds
```

Visualize some of these images to see handwritten digits in the dataset. The image size is 32 x 32.

```python
import matplotlib.pyplot as plt
ds = create_dataset('MNIST', training=False)
data = ds.create_dict_iterator().get_next()
images = data['image']
labels = data['label']

for i in range(1, 5):
    plt.subplot(2, 2, i)
    plt.imshow(images[i][0])
    plt.title('Number: %s' % labels[i])
    plt.xticks([])
plt.show()
```

![png](images/mnist.png)

### Network Definition

Define LeNet5 model. The structure of the LeNet5 model is shown below:

![](images/lenet5.jpg)

[1] Picture is from http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

> **Tips**: MindSpore model_zoo provides multiple common models that can be directly used.

```python
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
```

### Network Training

Use the MNIST dataset to train the LeNet5 model. The training strategy is shown as follows. Adjust the training strategy and view the training effect. The validation accuracy should be greater than 95%.

| batch size | number of epochs | learning rate | optimizer |
| -- | -- | -- | -- |
| 32 | 3 | 0.01 | Momentum 0.9 |

```python
def train(data_dir, lr=0.01, momentum=0.9, num_epochs=3):
    ds_train = create_dataset(data_dir)
    ds_eval = create_dataset(data_dir, training=False)

    net = LeNet5()
    loss = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    opt = nn.Momentum(net.trainable_params(), lr, momentum)
    loss_cb = LossMonitor(per_print_times=ds_train.get_dataset_size())

    model = Model(net, loss, opt, metrics={'acc', 'loss'})
    # dataset_sink_mode can be True when using Ascend
    model.train(num_epochs, ds_train, callbacks=[loss_cb], dataset_sink_mode=False)
    metrics = model.eval(ds_eval, dataset_sink_mode=False)
    print('Metrics:', metrics)

train('MNIST/')
```

    epoch: 1 step 1875, loss is 0.23394052684307098
    epoch: 2 step 1875, loss is 0.4737345278263092
    epoch: 3 step 1875, loss is 0.07734094560146332
    Metrics: {'loss': 0.10531254443608654, 'acc': 0.9701522435897436}

## Procedure(Terminal)

Experiment on a Windows/Ubuntu x64 PC/Laptop:

```sh
# Edit main.py, and set context to `device_target='CPU'` in line 15.
python main.py --data_url=D:\dataset\MNIST
```

Experiment on an Ascend Server：

```shell script
vim main.py # Set context to `device_target='CPU'` in line 15.
python main.py --data_url=/PATH/TO/MNIST
```

## Experiment Summary

This experiment demonstrates how to use MindSpore to recognize handwritten digits and how to develop and train a LeNet5 model. 
Train the LeNet5 model for several generations and use it to recognize handwritten digits. 
The recognition accuracy is greater than 95%. That is, LeNet5 has learned how to recognize handwritten digits.
