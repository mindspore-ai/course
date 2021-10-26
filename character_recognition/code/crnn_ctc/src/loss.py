'''
Date: 2021-09-05 14:53:34
LastEditors: xgy
LastEditTime: 2021-09-25 22:45:07
FilePath: \code\crnn_ctc\src\loss.py
'''

"""CTC Loss."""
import numpy as np
from mindspore.nn.loss.loss import _Loss
from mindspore import Tensor, Parameter
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P


# class CTCLoss(_Loss):
#     """
#      CTCLoss definition

#      Args:
#         max_sequence_length(int): max number of sequence length. For text images, the value is equal to image width
#         max_label_length(int): max number of label length for each input.
#         batch_size(int): batch size of input logits
#      """

#     def __init__(self, max_sequence_length, max_label_length, batch_size):
#         super(CTCLoss, self).__init__()
#         self.sequence_length = Parameter(Tensor(np.array([max_sequence_length] * batch_size), mstype.int32),
#                                          name="sequence_length")
#         labels_indices = []
#         for i in range(batch_size):
#             for j in range(max_label_length):
#                 labels_indices.append([i, j])
#         self.labels_indices = Parameter(Tensor(np.array(labels_indices), mstype.int64), name="labels_indices")
#         self.reshape = P.Reshape()
#         self.ctc_loss = P.CTCLoss(ctc_merge_repeated=True)

#     def construct(self, logit, label):
#         labels_values = self.reshape(label, (-1,))
#         loss, _ = self.ctc_loss(logit, self.labels_indices, labels_values, self.sequence_length)
#         return loss

class CTCLoss(_Loss):
    """
     CTCLoss definition

     Args:
        max_sequence_length(int): max number of sequence length. For captcha images, the value is equal to image
        width
        max_label_length(int): max number of label length for each input.
        batch_size(int): batch size of input logits
     """

    def __init__(self, max_sequence_length, max_label_length, batch_size):
        super(CTCLoss, self).__init__()
        self.sequence_length = Parameter(Tensor(np.array([max_sequence_length] * batch_size), mstype.int32))
        labels_indices = []
        for i in range(batch_size):
            for j in range(max_label_length):
                labels_indices.append([i, j])
        self.labels_indices = Parameter(Tensor(np.array(labels_indices), mstype.int64))
        self.reshape = P.Reshape()
        self.ctc_loss = P.CTCLoss(ctc_merge_repeated=True)

    def construct(self, logit, label):
        labels_values = self.reshape(label, (-1,))
        loss, _ = self.ctc_loss(logit, self.labels_indices, labels_values, self.sequence_length)
        return loss