#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import math
import torch
from torch.nn.functional import smooth_l1_loss


class l1loss:
    """ Calculate IoU loss.
    """

    def __init__(self):
        """ Setting of the class.
        Args:
            box_format: (string), must be one of 'xywh' or 'xyxy'.
            iou_type: (string), can be one of 'ciou', 'diou', 'giou' or 'siou'
            reduction: (string), specifies the reduction to apply to the output, must be one of 'none', 'mean','sum'.
            eps: (float), a value to avoid divide by zero error.
        """

    def __call__(self, predbox, gtbox):
        pred_xy_list=list(map(lambda x:x,predbox)) # len = 8 (every positive anchor)
        gt_xy_list=list(map(lambda x:x,gtbox))
        point_loss_list=list(map(lambda x,y:smooth_l1_loss(x,y,reduction="sum"),pred_xy_list,gt_xy_list))
        point_loss_tensor=torch.tensor(point_loss_list).cuda()
        point_loss_tensor=torch.unsqueeze(point_loss_tensor,-1)
        # total_l1_loss=sum(point_loss_list)

        return point_loss_tensor