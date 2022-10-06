#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import math
import torch
from torch.nn.functional import smooth_l1_loss
from shapely.geometry import Polygon



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
        point_loss_list=list(map(lambda x,y:smooth_l1_loss(x,y,reduction="mean"),pred_xy_list,gt_xy_list)) #l1loss point

        pk1=abs((predbox[:,1]-predbox[:,5])/(predbox[:,0]-predbox[:,4]))
        pk2=abs((predbox[:,7]-predbox[:,3])/(predbox[:,6]-predbox[:,2]))

        gk1=abs((gtbox[:,1]-gtbox[:,5])/(gtbox[:,0]-gtbox[:,4]))
        gk2=abs((gtbox[:,7]-gtbox[:,3])/(gtbox[:,6]-gtbox[:,2]))

        pk1_list = list(map(lambda x: x, pk1))
        pk2_list = list(map(lambda x: x, pk2))
        gk1_list = list(map(lambda x: x, gk1))
        gk2_list = list(map(lambda x: x, gk2))

        k1_loss_list=list(map(lambda x,y:smooth_l1_loss(x,y,reduction="mean"),pk1_list,gk1_list)) #l1loss k
        k2_loss_list=list(map(lambda x,y:smooth_l1_loss(x,y,reduction="mean"),pk2_list,gk2_list)) #l1loss k

        point_loss_tensor=torch.tensor(point_loss_list).cuda()
        point_loss_tensor=torch.unsqueeze(point_loss_tensor,-1)

        k1_loss_tensor=torch.tensor(k1_loss_list).cuda()
        k1_loss_tensor=torch.unsqueeze(k1_loss_tensor,-1)

        k2_loss_tensor = torch.tensor(k2_loss_list).cuda()
        k2_loss_tensor = torch.unsqueeze(k2_loss_tensor, -1)
        k1_weight=10
        k2_weight=10
        k_point_loss=point_loss_tensor+k1_loss_tensor*k1_weight+k2_loss_tensor*k2_weight

        return k_point_loss