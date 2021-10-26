import numpy as np
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype

def _weight_variable(shape, factor=0.01):
    ''''weight initialize'''
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)

def _BatchNorm2dInit(out_chls, momentum=0.1, affine=True, use_batch_statistics=False):
    """Batchnorm2D wrapper."""
    gamma_init = Tensor(np.array(np.ones(out_chls)).astype(np.float32))
    beta_init = Tensor(np.array(np.ones(out_chls) * 0).astype(np.float32))
    moving_mean_init = Tensor(np.array(np.ones(out_chls) * 0).astype(np.float32))
    moving_var_init = Tensor(np.array(np.ones(out_chls)).astype(np.float32))

    return nn.BatchNorm2d(out_chls, momentum=momentum, affine=affine, gamma_init=gamma_init,
                          beta_init=beta_init, moving_mean_init=moving_mean_init,
                          moving_var_init=moving_var_init, use_batch_statistics=use_batch_statistics)

def _conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0, pad_mode='pad', weights_update=True):
    """Conv2D wrapper."""
    layers = []
    conv = nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     pad_mode=pad_mode, has_bias=False)
    if not weights_update:
        conv.weight.requires_grad = False
    layers += [conv]
    layers += [_BatchNorm2dInit(out_channels)]
    return nn.SequentialCell(layers)


def _fc(in_channels, out_channels):
    '''full connection layer'''
    return nn.Dense(in_channels, out_channels)


class VGG16FeatureExtraction(nn.Cell):
    def __init__(self, weights_update=False):
        """
        VGG16 feature extraction

        Args:
            weights_updata(bool): whether update weights for top two layers, default is False.
        """
        super(VGG16FeatureExtraction, self).__init__()
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv1_1 = _conv(in_channels=3, out_channels=64, kernel_size=3,\
            padding=1, weights_update=weights_update)
        self.conv1_2 = _conv(in_channels=64, out_channels=64, kernel_size=3,\
            padding=1, weights_update=weights_update)

        self.conv2_1 = _conv(in_channels=64, out_channels=128, kernel_size=3,\
            padding=1, weights_update=weights_update)
        self.conv2_2 = _conv(in_channels=128, out_channels=128, kernel_size=3,\
            padding=1, weights_update=weights_update)

        self.conv3_1 = _conv(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = _conv(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = _conv(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv4_1 = _conv(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2 = _conv(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_3 = _conv(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv5_1 = _conv(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2 = _conv(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_3 = _conv(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.cast = P.Cast()

    def construct(self, x):
        """
        :param x: shape=(B, 3, 224, 224)
        :return:
        """
        x = self.cast(x, mstype.float32)
        x = self.conv1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv3_1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.relu(x)
        x = self.conv3_3(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.relu(x)
        x = self.conv4_3(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv5_1(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.relu(x)
        x = self.conv5_3(x)
        x = self.relu(x)
        return x

class VGG16Classfier(nn.Cell):
    def __init__(self):
        """VGG16 classfier structure"""
        super(VGG16Classfier, self).__init__()
        self.flatten = P.Flatten()
        self.relu = nn.ReLU()
        self.fc1 = _fc(in_channels=7*7*512, out_channels=4096)
        self.fc2 = _fc(in_channels=4096, out_channels=4096)
        self.reshape = P.Reshape()
        self.dropout = nn.Dropout(0.5)

    def construct(self, x):
        """
        :param x: shape=(B, 512, 7, 7)
        :return:
        """
        x = self.reshape(x, (-1, 7*7*512))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class VGG16(nn.Cell):
    def __init__(self, num_classes):
        """VGG16 construct for training backbone"""
        super(VGG16, self).__init__()
        self.vgg16_feature_extractor = VGG16FeatureExtraction(weights_update=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = VGG16Classfier()
        self.fc3 = _fc(in_channels=4096, out_channels=num_classes)

    def construct(self, x):
        """
        :param x: shape=(B, 3, 224, 224)
        :return: logits, shape=(B, 1000)
        """
        feature_maps = self.vgg16_feature_extractor(x)
        x = self.max_pool(feature_maps)
        x = self.classifier(x)
        x = self.fc3(x)
        return x
