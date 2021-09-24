"""
DFCNN model
"""
from mindspore import nn, Tensor, ops
import numpy as np


class DFCNN(nn.Cell):
    """DFCNN model
    """

    def __init__(self, num_classes, input_nc=1, padding=1, pad_mode='pad', has_bias=False, use_dropout=False):
        super(DFCNN, self).__init__()

        if pad_mode == 'pad':
            assert padding >= 0, "when the pad_mode is 'pad', the padding must be greater than or equal to 0!"

        if pad_mode == 'same' or pad_mode == 'valid':
            assert padding == 0, "when the pad_mode is 'same' or 'valid', the padding must be equal to 0!"

        self.use_dropout = use_dropout

        # structure

        # seq 1
        self.conv11 = nn.Conv2d(
            in_channels=input_nc, out_channels=64,
            kernel_size=3, stride=1, padding=padding, has_bias=has_bias, pad_mode=pad_mode
        )
        self.bn11 = nn.BatchNorm2d(64)
        self.relu11 = nn.ReLU()
        self.conv12 = nn.Conv2d(in_channels=64, out_channels=64,
                                kernel_size=3, stride=1, padding=padding, has_bias=has_bias, pad_mode=pad_mode
                                )
        self.bn12 = nn.BatchNorm2d(64)
        self.relu12 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        # seq 2
        self.conv21 = nn.Conv2d(
            in_channels=64, out_channels=128,
            kernel_size=3, stride=1, padding=padding, has_bias=has_bias, pad_mode=pad_mode
        )
        self.bn21 = nn.BatchNorm2d(128)
        self.relu21 = nn.ReLU()
        self.conv22 = nn.Conv2d(in_channels=128, out_channels=128,
                                kernel_size=3, stride=1, padding=padding, has_bias=has_bias, pad_mode=pad_mode
                                )
        self.bn22 = nn.BatchNorm2d(128)
        self.relu22 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        # seq 3
        self.conv31 = nn.Conv2d(
            in_channels=128, out_channels=256,
            kernel_size=3, stride=1, padding=padding, has_bias=has_bias, pad_mode=pad_mode
        )
        self.bn31 = nn.BatchNorm2d(256)
        self.relu31 = nn.ReLU()
        self.conv32 = nn.Conv2d(in_channels=256, out_channels=256,
                                kernel_size=3, stride=1, padding=padding, has_bias=has_bias, pad_mode=pad_mode
                                )
        self.bn32 = nn.BatchNorm2d(256)
        self.relu32 = nn.ReLU()
        self.conv33 = nn.Conv2d(in_channels=256, out_channels=256,
                                kernel_size=3, stride=1, padding=padding, has_bias=has_bias, pad_mode=pad_mode
                                )
        self.bn33 = nn.BatchNorm2d(256)
        self.relu33 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        # seq 4
        self.conv41 = nn.Conv2d(
            in_channels=256, out_channels=512,
            kernel_size=3, stride=1, padding=padding, has_bias=has_bias, pad_mode=pad_mode
        )
        self.bn41 = nn.BatchNorm2d(512)
        self.relu41 = nn.ReLU()
        self.conv42 = nn.Conv2d(in_channels=512, out_channels=512,
                                kernel_size=3, stride=1, padding=padding, has_bias=has_bias, pad_mode=pad_mode
                                )
        self.bn42 = nn.BatchNorm2d(512)
        self.relu42 = nn.ReLU()
        self.conv43 = nn.Conv2d(in_channels=512, out_channels=512,
                                kernel_size=3, stride=1, padding=padding, has_bias=has_bias, pad_mode=pad_mode
                                )
        self.bn43 = nn.BatchNorm2d(512)
        self.relu43 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=1, stride=1, pad_mode='valid')

        # seq 5
        self.conv51 = nn.Conv2d(
            in_channels=512, out_channels=512,
            kernel_size=3, stride=1, padding=padding, has_bias=has_bias, pad_mode=pad_mode
        )
        self.bn51 = nn.BatchNorm2d(512)
        self.relu51 = nn.ReLU()
        self.conv52 = nn.Conv2d(in_channels=512, out_channels=512,
                                kernel_size=3, stride=1, padding=padding, has_bias=has_bias, pad_mode=pad_mode
                                )
        self.bn52 = nn.BatchNorm2d(512)
        self.relu52 = nn.ReLU()
        self.conv53 = nn.Conv2d(in_channels=512, out_channels=512,
                                kernel_size=3, stride=1, padding=padding, has_bias=has_bias, pad_mode=pad_mode
                                )
        self.bn53 = nn.BatchNorm2d(512)
        self.relu53 = nn.ReLU()
        self.maxpool5 = nn.MaxPool2d(kernel_size=1, stride=1, pad_mode='valid')

        self.bn = nn.BatchNorm2d(512)
        if self.use_dropout:
            self.drop1 = nn.Dropout(0.8)
            self.drop2 = nn.Dropout(0.8)
            self.drop3 = nn.Dropout(0.8)
            self.drop4 = nn.Dropout(0.8)
            self.drop5 = nn.Dropout(0.8)
            self.drop_fc1 = nn.Dropout(0.5)
            self.drop_fc2 = nn.Dropout(0.5)
        self.fc1 = nn.Dense(25 * 512, 4096, activation='relu')
        self.fc2 = nn.Dense(4096, 4096, activation='relu')
        self.fc3 = nn.Dense(4096, num_classes, activation='relu')

        # operation
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()

    def construct(self, x):
        x = self.feature(x)  # [batch, 256, 200, 25] -> [batch, channels, max_time, fq]
        x = self.bn(x)
        x = self.transpose(x, (0, 2, 1, 3))  # [batch, channels, max_time, fq] -> [batch, max_time, channels, fq]
        x = self.reshape(x, (-1, x.shape[1], x.shape[2] * x.shape[3]))  # [batch, max_time, channels*fq]

        x = self.fc1(x)
        if self.use_dropout:
            x = self.drop_fc1(x)
        x = self.fc2(x)
        if self.use_dropout:
            x = self.drop_fc2(x)
        x = self.fc3(x)

        return x

    def feature(self, x):
        """
        get feature


        Args:
            x: input

        Returns:
            x: output
        """
        # seq 1
        x = self.conv11(x)
        x = self.bn11(x)
        x = self.relu11(x)
        x = self.conv12(x)
        x = self.bn12(x)
        x = self.relu12(x)
        x = self.maxpool1(x)
        if self.use_dropout:
            x = self.drop1(x)

        # seq 2
        x = self.conv21(x)
        x = self.bn21(x)
        x = self.relu21(x)
        x = self.conv22(x)
        x = self.bn22(x)
        x = self.relu22(x)
        x = self.maxpool2(x)
        if self.use_dropout:
            x = self.drop2(x)

        # seq 3
        x = self.conv31(x)
        x = self.bn31(x)
        x = self.relu31(x)
        x = self.conv32(x)
        x = self.bn32(x)
        x = self.relu32(x)
        x = self.conv33(x)
        x = self.bn33(x)
        x = self.relu33(x)
        x = self.maxpool3(x)  # [batch, 256, 200, 25]
        if self.use_dropout:
            x = self.drop3(x)

        # seq 4
        x = self.conv41(x)
        x = self.bn41(x)
        x = self.relu41(x)
        x = self.conv42(x)
        x = self.bn42(x)
        x = self.relu42(x)
        x = self.conv43(x)
        x = self.bn43(x)
        x = self.relu43(x)
        x = self.maxpool4(x)  # [batch, 256, 200, 12]
        if self.use_dropout:
            x = self.drop4(x)

        #  seq 5
        x = self.conv51(x)
        x = self.bn51(x)
        x = self.relu51(x)
        x = self.conv52(x)
        x = self.bn52(x)
        x = self.relu52(x)
        x = self.conv53(x)
        x = self.bn53(x)
        x = self.relu53(x)
        x = self.maxpool5(x)  # [batch, 512, 200, 6]
        if self.use_dropout:
            x = self.drop5(x)

        return x  # [batch, 512, 200, 6]


if __name__ == '__main__':
    from mindspore import context

    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU', device_id=0)
    net = DFCNN(num_classes=1209, input_nc=1)
    input = Tensor(np.full(shape=(16, 1, 1600, 200), fill_value=1., dtype=np.float32))
    output = net(input)
