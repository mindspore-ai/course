'''
Author: jojo
Date: 2021-08-03 10:20:33
LastEditors: jojo
LastEditTime: 2021-08-03 10:20:57
FilePath: /210610338/model/CTC_v2.py
'''
from mindspore import nn
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
class ctc_loss(nn.Cell):

    def __init__(self):
        super(ctc_loss, self).__init__()

        self.loss = P.CTCLoss(preprocess_collapse_repeated=False,
                              ctc_merge_repeated=True,
                              ignore_longer_outputs_than_inputs=False)

        self.mean = P.ReduceMean()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()

    def construct(self, inputs, labels_indices, labels_values, sequence_length):
        inputs = self.transpose(inputs, (1, 0, 2))

        loss, _ = self.loss(inputs, labels_indices, labels_values, sequence_length)

        loss = self.mean(loss)
        return loss
