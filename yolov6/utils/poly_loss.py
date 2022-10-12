#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import torch
from torch.nn.functional import smooth_l1_loss


class Poly_loss:
    def __init__(self,centroid_loss_weight=0.2,angle_loss_weight=5.0,length_loss_weight=1.0):
        self.centroid_loss_weight=centroid_loss_weight
        self.angle_loss_weight=angle_loss_weight
        self.length_loss_weight=length_loss_weight

    def __call__(self, predbox, gtbox):
        return self.smoothLnloss(predbox,gtbox)

    def smoothLnloss(self,predbox,gtbox):
        wchr=torch.max(predbox[:,2],predbox[:,4])-torch.min(predbox[:,0],predbox[:,6])
        hchr=torch.max(predbox[:,5],predbox[:,7])-torch.min(predbox[:,1],predbox[:,3])
        _, predbox_x, predbox_y, gtbox_x, gtbox_y = self.get_centroid(predbox, gtbox)

        d_centroid_x=(gtbox_x-predbox_x)/wchr
        d_centroid_y=(gtbox_y-predbox_y)/hchr

        d_p1_x=(gtbox[:,0]-predbox[:,0])/wchr
        d_p1_y=(gtbox[:,1]-predbox[:,1])/hchr

        d_p2_x=(gtbox[:,2]-predbox[:,2])/wchr
        d_p2_y=(gtbox[:,3]-predbox[:,3])/hchr

        d_p3_x=(gtbox[:,4]-predbox[:,4])/wchr
        d_p3_y=(gtbox[:,5]-predbox[:,5])/hchr

        d_p4_x=(gtbox[:,6]-predbox[:,6])/wchr
        d_p4_y=(gtbox[:,7]-predbox[:,7])/hchr

        lnloss=lambda x:(torch.abs(x)+1)*torch.log(torch.abs(x)+1)-abs(x)

        d_points=torch.cat([d_centroid_x,d_centroid_y,d_p1_x,d_p1_y,d_p2_x,d_p2_y,d_p3_x,d_p3_y,d_p4_x,d_p4_y],dim=-1)

        d_centroid_x=lnloss(d_points)

        result=torch.sum(d_centroid_x)

        result=torch.unsqueeze(result,-1)

        return result

    def centroid_loss(self,predbox,gtbox):
        bias = abs(predbox - gtbox)
        max = torch.max(predbox, gtbox)
        log = torch.log(1 - torch.clip(input=(bias / max), min=1))
        result = -torch.mean(log)
        if torch.isnan(result):
            result = smooth_l1_loss(predbox, gtbox)
            print("Now the loss is from smoothL1")
        length_loss = torch.unsqueeze(result, -1)

        centroid_loss, predbox_x, predbox_y, gtbox_x, gtbox_y = self.get_centroid(predbox, gtbox)

        a_p1_angle = torch.arctan((predbox[:, 1] - predbox_y) / (predbox[:, 0] - predbox_x))
        a_p2_angle = torch.arctan((predbox[:, 3] - predbox_y) / (predbox[:, 2] - predbox_x))
        a_p3_angle = torch.arctan((predbox[:, 5] - predbox_y) / (predbox[:, 4] - predbox_x))
        a_p4_angle = torch.arctan((predbox[:, 7] - predbox_y) / (predbox[:, 6] - predbox_x))

        b_p1_angle = torch.arctan((gtbox[:, 1] - gtbox_y) / (gtbox[:, 0] - gtbox_x))
        b_p2_angle = torch.arctan((gtbox[:, 3] - gtbox_y) / (gtbox[:, 2] - gtbox_x))
        b_p3_angle = torch.arctan((gtbox[:, 5] - gtbox_y) / (gtbox[:, 4] - gtbox_x))
        b_p4_angle = torch.arctan((gtbox[:, 7] - gtbox_y) / (gtbox[:, 6] - gtbox_x))

        p1_result = torch.pow(b_p1_angle - a_p1_angle, 2)
        p2_result = torch.pow(b_p2_angle - a_p2_angle, 2)
        p3_result = torch.pow(b_p3_angle - a_p3_angle, 2)
        p4_result = torch.pow(b_p4_angle - a_p4_angle, 2)

        result = torch.mean(p1_result + p2_result + p3_result + p4_result)
        angle_loss = torch.unsqueeze(result, -1)

        loss = self.centroid_loss_weight * centroid_loss + self.angle_loss_weight * angle_loss + self.length_loss_weight * length_loss
        return loss

    def get_centroid(self,predbox, gtbox):
        c1ax = (predbox[:, 0] + predbox[:, 2] + predbox[:, 4]) / 3
        c1ay = (predbox[:, 1] + predbox[:, 3] + predbox[:, 5]) / 3

        c1_a_s=(predbox[:,0]*predbox[:,5]+predbox[:,4]*predbox[:,3]+predbox[:,2]*predbox[:,1]-predbox[:,0]*predbox[:,3]-predbox[:,4]*predbox[:,1]-predbox[:,2]*predbox[:,5])/2

        c2ax = (predbox[:, 0] + predbox[:, 4] + predbox[:, 6]) / 3
        c2ay = (predbox[:, 1] + predbox[:, 5] + predbox[:, 7]) / 3

        c2_a_s=(predbox[:,0]*predbox[:,7]+predbox[:,6]*predbox[:,5]+predbox[:,4]*predbox[:,1]-predbox[:,0]*predbox[:,5]-predbox[:,6]*predbox[:,1]-predbox[:,4]*predbox[:,7])/2

        a_cx = (c1_a_s * c1ax + c2_a_s * c2ax) / (c1_a_s + c2_a_s)
        a_cy = (c1_a_s * c1ay + c2_a_s * c2ay) / (c1_a_s + c2_a_s)

        c1bx = (gtbox[:, 0] + gtbox[:, 2] + gtbox[:, 4]) / 3
        c1by = (gtbox[:, 1] + gtbox[:, 3] + gtbox[:, 5]) / 3

        c1_b_s=(gtbox[:,0]*gtbox[:,5]+gtbox[:,4]*gtbox[:,3]+gtbox[:,2]*gtbox[:,1]-gtbox[:,0]*gtbox[:,3]-gtbox[:,4]*gtbox[:,1]-gtbox[:,2]*gtbox[:,5])/2

        c2bx = (gtbox[:, 0] + gtbox[:, 4] + gtbox[:, 6]) / 3
        c2by = (gtbox[:, 1] + gtbox[:, 5] + gtbox[:, 7]) / 3

        c2_b_s=(gtbox[:,0]*gtbox[:,7]+gtbox[:,6]*gtbox[:,5]+gtbox[:,4]*gtbox[:,1]-gtbox[:,0]*gtbox[:,5]-gtbox[:,6]*gtbox[:,1]-gtbox[:,4]*gtbox[:,7])/2

        b_cx = (c1_b_s * c1bx + c2_b_s * c2bx) / (c1_b_s + c2_b_s)
        b_cy = (c1_b_s * c1by + c2_b_s * c2by) / (c1_b_s + c2_b_s)

        cx_bias = abs(a_cx - b_cx)
        cy_bias = abs(a_cy - b_cy)

        bias = cx_bias + cy_bias

        result = torch.mean(bias)
        centroid_loss = torch.unsqueeze(result, -1)
        return centroid_loss, a_cx, a_cy, b_cx, b_cy