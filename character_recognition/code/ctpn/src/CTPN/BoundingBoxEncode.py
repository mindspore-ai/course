'''
Date: 2021-08-02 22:38:28
LastEditors: xgy
LastEditTime: 2021-08-15 16:12:21
FilePath: \code\ctpn\src\CTPN\BoundingBoxEncode.py
'''
import mindspore.nn as nn
from mindspore.ops import operations as P

class BoundingBoxEncode(nn.Cell):
    """
    BoundintBox Decoder.

    Returns:
        pred_box(Tensor): decoder bounding boxes.
    """
    def __init__(self):
        super(BoundingBoxEncode, self).__init__()
        self.split = P.Split(axis=1, output_num=4)
        self.ones = 1.0
        self.half = 0.5
        self.log = P.Log()
        self.concat = P.Concat(axis=1)
    def construct(self, anchor_box, gt_box):
        """
        boxes(Tensor): boundingbox.
        deltas(Tensor): delta between boundingboxs and anchors.
        """
        x1, y1, x2, y2 = self.split(anchor_box)
        width = x2 - x1 + self.ones
        height = y2 - y1 + self.ones
        ctr_x = x1 + self.half * width
        ctr_y = y1 + self.half * height
        gt_x1, gt_y1, gt_x2, gt_y2 = self.split(gt_box)
        gt_width = gt_x2 - gt_x1 + self.ones
        gt_height = gt_y2 - gt_y1 + self.ones
        ctr_gt_x = gt_x1 + self.half * gt_width
        ctr_gt_y = gt_y1 + self.half * gt_height

        target_dx = (ctr_gt_x - ctr_x) / width
        target_dy = (ctr_gt_y - ctr_y) / height
        dw = gt_width / width
        dh = gt_height / height
        target_dw = self.log(dw)
        target_dh = self.log(dh)
        deltas = self.concat((target_dx, target_dy, target_dw, target_dh))
        return deltas
