'''
Date: 2021-08-02 22:38:28
LastEditors: xgy
LastEditTime: 2021-08-15 16:12:14
FilePath: \code\ctpn\src\CTPN\BoundingBoxDecode.py
'''
import mindspore.nn as nn
from mindspore.ops import operations as P

class BoundingBoxDecode(nn.Cell):
    """
    BoundintBox Decoder.

    Returns:
        pred_box(Tensor): decoder bounding boxes.
    """
    def __init__(self):
        super(BoundingBoxDecode, self).__init__()
        self.split = P.Split(axis=1, output_num=4)
        self.ones = 1.0
        self.half = 0.5
        self.log = P.Log()
        self.exp = P.Exp()
        self.concat = P.Concat(axis=1)

    def construct(self, bboxes, deltas):
        """
        boxes(Tensor): boundingbox.
        deltas(Tensor): delta between boundingboxs and anchors.
        """
        x1, y1, x2, y2 = self.split(bboxes)
        width = x2 - x1 + self.ones
        height = y2 - y1 + self.ones
        ctr_x = x1 + self.half * width
        ctr_y = y1 + self.half * height
        _, dy, _, dh = self.split(deltas)
        pred_ctr_x = ctr_x
        pred_ctr_y = dy * height + ctr_y
        pred_w = width
        pred_h = self.exp(dh) * height

        x1 = pred_ctr_x - self.half * pred_w
        y1 = pred_ctr_y - self.half * pred_h
        x2 = pred_ctr_x + self.half * pred_w
        y2 = pred_ctr_y + self.half * pred_h
        pred_box = self.concat((x1, y1, x2, y2))
        return pred_box
