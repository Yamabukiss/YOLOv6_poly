import torch
import torch.nn.functional as F

def dist_calculator(gt_bboxes, anchor_bboxes):
    """compute center distance between all bbox and gt

    Args:
        gt_bboxes (Tensor): shape(bs*n_max_boxes, 4)
        anchor_bboxes (Tensor): shape(num_total_anchors, 4)
    Return:
        distances (Tensor): shape(bs*n_max_boxes, num_total_anchors)
        ac_points (Tensor): shape(num_total_anchors, 2)
    """  
    gt_crx = (gt_bboxes[:, 2] + gt_bboxes[:, 4]) / 2.0 # top edge center point
    gt_clx = (gt_bboxes[:, 0] + gt_bboxes[:, 6]) / 2.0 # top edge center point
    gt_lrx=(gt_crx+gt_clx)/2.0

    gt_cty = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0 # top edge center point
    gt_cby = (gt_bboxes[:, 5] + gt_bboxes[:, 7]) / 2.0 # top edge center point
    gt_tby=(gt_cty+gt_cby)/2.0

    gt_points = torch.stack([gt_lrx, gt_tby], dim=1)
    ac_cx = (anchor_bboxes[:, 0] + anchor_bboxes[:, 2]) / 2.0
    ac_cy = (anchor_bboxes[:, 1] + anchor_bboxes[:, 3]) / 2.0
    ac_points = torch.stack([ac_cx, ac_cy], dim=1)

    distances = (gt_points[:, None, :] - ac_points[None, :, :]).pow(2).sum(-1).sqrt()
    
    return distances, ac_points

def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
    """select the positive anchors's center in gt

    Args:
        xy_centers (Tensor): shape(bs*n_max_boxes, num_total_anchors, 4)
        gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
    Return:
        (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    """  
    n_anchors = xy_centers.size(0)
    bs, n_max_boxes, _ = gt_bboxes.size()
    _gt_bboxes = gt_bboxes.reshape([-1, 8])
    xy_centers = xy_centers.unsqueeze(0).repeat(bs * n_max_boxes, 1, 1) 

    gt_bboxes_p1 = _gt_bboxes[:, 0:2].unsqueeze(1).repeat(1, n_anchors, 1)
    gt_bboxes_p2 = _gt_bboxes[:, 2:4].unsqueeze(1).repeat(1, n_anchors, 1)
    gt_bboxes_p3 = _gt_bboxes[:, 4:6].unsqueeze(1).repeat(1, n_anchors, 1)
    gt_bboxes_p4 = _gt_bboxes[:, 6:8].unsqueeze(1).repeat(1, n_anchors, 1)
    b_p2=torch.zeros(xy_centers.shape).cuda()
    b_p4=torch.zeros(xy_centers.shape).cuda()
    b_p1=xy_centers-gt_bboxes_p1
    b_p2[:,:,0]=gt_bboxes_p2[:,:,0]-xy_centers[:,:,0]
    b_p2[:,:,1]=xy_centers[:,:,1]-gt_bboxes_p2[:,:,1]
    b_p3=gt_bboxes_p3-xy_centers
    b_p4[:, :, 0] = xy_centers[:, :, 0] - gt_bboxes_p4[:, :,0]
    b_p4[:, :, 1] = gt_bboxes_p4[:,:,1] - xy_centers[:, :, 1]

    # the point should be included by the polygon

    # b_lt = xy_centers - gt_bboxes_lt
    # b_rb = gt_bboxes_rb - xy_centers
    bbox_deltas = torch.cat([b_p1,b_p2,b_p3,b_p4], dim=-1)
    bbox_deltas = bbox_deltas.reshape([bs, n_max_boxes, n_anchors, -1])
    return (bbox_deltas.min(axis=-1)[0] > eps).to(gt_bboxes.dtype)

def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """if an anchor box is assigned to multiple gts,
        the one with the highest iou will be selected.

    Args:
        mask_pos (Tensor): shape(bs, n_max_boxes, num_total_anchors)
        overlaps (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    Return:
        target_gt_idx (Tensor): shape(bs, num_total_anchors)
        fg_mask (Tensor): shape(bs, num_total_anchors)
        mask_pos (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    """  
    fg_mask = mask_pos.sum(axis=-2)
    if fg_mask.max() > 1:
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])
        max_overlaps_idx = overlaps.argmax(axis=1)
        is_max_overlaps = F.one_hot(max_overlaps_idx, n_max_boxes)
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
        fg_mask = mask_pos.sum(axis=-2)
    target_gt_idx = mask_pos.argmax(axis=-2)
    return target_gt_idx, fg_mask , mask_pos

def iou_calculator(box1, box2, eps=1e-9): # for poly now
    """Calculate iou for batch

    Args:
        box1 (Tensor): shape(bs, n_max_boxes, 1, 4)
        box2 (Tensor): shape(bs, 1, num_total_anchors, 4)
    Return:
        (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    """
    box1 = box1.unsqueeze(2)  # [N, M1, 4] -> [N, M1, 1, 4]
    box2 = box2.unsqueeze(1)  # [N, M2, 4] -> [N, 1, M2, 4]
    px1y1, px2y2,px3y3,px4y4 = box1[:, :, :, 0:2], box1[:, :, :, 2:4],box1[:, :, :, 4:6],box1[:, :, :, 6:8]
    gx1y1, gx2y2,gx3y3,gx4y4 = box2[:, :, :, 0:2], box2[:, :, :, 2:4],box2[:, :, :, 4:6],box2[:, :, :, 6:8]

    x1y1 = torch.maximum(px1y1, gx1y1) # 输出二者中较大的那一个数 （在该维度上）
    # x2y2 = torch.minimum(px2y2, gx2y2)
    x3y3 = torch.minimum(px3y3, gx3y3)
    # x4y4 = torch.minimum(px4y4, gx4y4)

    overlap = (x3y3 - x1y1).clip(0).prod(-1) # clip limit to [] prod 将该维度的所有数相乘  此处仍需优化 目前沿用正方形的方法近似 clip 0 可以规避问题 令不相交的交集为0
    area1 = (px3y3 - px1y1).clip(0).prod(-1)
    area2 = (gx3y3 - gx1y1).clip(0).prod(-1)
    union = area1 + area2 - overlap + eps

    return overlap / union

from shapely.geometry import Polygon

# def poly_iou_calculator(box1, box2, eps=1e-9): # for poly now
#     """Calculate iou for batch
#
#     Args:
#         box1 (Tensor): shape(bs, n_max_boxes, 1, 4)
#         box2 (Tensor): shape(bs, 1, num_total_anchors, 4)
#     Return:
#         (Tensor): shape(bs, n_max_boxes, num_total_anchors)
#     """
#     box1 = box1.unsqueeze(2)  # [N, M1, 4] -> [N, M1, 1, 4]
#     box2 = box2.unsqueeze(1)  # [N, M2, 4] -> [N, 1, M2, 4]
#     px1y1, px2y2,px3y3,px4y4 = box1[:, :, :, 0:2], box1[:, :, :, 2:4],box1[:, :, :, 4:6],box1[:, :, :, 6:8]
#     gx1y1, gx2y2,gx3y3,gx4y4 = box2[:, :, :, 0:2], box2[:, :, :, 2:4],box2[:, :, :, 4:6],box2[:, :, :, 6:8]
#     poly_iou=torch.zeros(gx2y2.shape[0:3]).cuda()
#     for batch in range(px1y1.shape[0]):
#         for i in range(px1y1.shape[-2]):# gt num
#             poly1=Polygon([[px1y1[batch][0][i][0],px1y1[batch][0][i][1]],[px2y2[batch][0][i][0],px2y2[batch][0][i][1]],[px3y3[batch][0][i][0],px3y3[batch][0][i][1]],[px4y4[batch][0][i][0],px4y4[batch][0][i][1]]])
#             for j in range(gx1y1.shape[-2]): #pred num
#                 poly2 = Polygon([[gx1y1[batch][0][j][0], gx1y1[batch][0][j][1]], [gx2y2[batch][0][j][0], gx2y2[batch][0][j][1]],[gx3y3[batch][0][j][0], gx3y3[batch][0][j][1]], [gx4y4[batch][0][j][0], gx4y4[batch][0][j][1]]])
#                 iou = poly1.intersection(poly2).area / poly1.union(poly2).area+eps
#                 poly_iou[:,:,j]=iou
#
#     return poly_iou