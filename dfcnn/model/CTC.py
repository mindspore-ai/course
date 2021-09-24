"""
CTCLoss
"""
from mindspore import nn
from mindspore.ops import operations as P


class CTCLoss(nn.Cell):
    """
    CTCLoss
    """

    def __init__(self):
        super(CTCLoss, self).__init__()

        self.loss = P.CTCLoss(preprocess_collapse_repeated=False,
                              ctc_merge_repeated=True,
                              ignore_longer_outputs_than_inputs=False)

        self.mean = P.ReduceMean()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()

    def construct(self, inputs, labels_indices, labels_values, sequence_length):
        """
        CTCLoss forward


        Args:
            inputs: the output of the DFCNN
            labels_indices: get from the data generator
            labels_values: get from the data generator
            sequence_length: get from the data generator

        Returns:
            loss: the loss value
        """
        inputs = self.transpose(inputs, (1, 0, 2))

        loss, _ = self.loss(inputs, labels_indices, labels_values, sequence_length)

        loss = self.mean(loss)
        return loss
