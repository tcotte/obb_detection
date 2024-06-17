from torch import nn

import box_ops


def oriented_giou_loss(src_bboxes, target_bboxes):
    return 1 - box_ops.generalized_box_iou(target_bboxes, src_bboxes)


class OrientedGIoULoss(nn.Module):
    def __init__(self):
        super(OrientedGIoULoss, self).__init__()

    def forward(self, predictions, target):
        return oriented_giou_loss(src_bboxes=predictions, target_bboxes=target)
