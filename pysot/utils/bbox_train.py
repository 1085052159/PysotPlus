from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
import numpy as np
import torch


def gen_anchor(batch_size):
    """
    :param batch_size:
    :return:
    anchor_center_: N * k * h * w, 4
    anchor_box_: N * k * h * w, 4
    """
    anchors = Anchors(cfg.ANCHOR.STRIDE,
                      cfg.ANCHOR.RATIOS,
                      cfg.ANCHOR.SCALES)
    anchors.generate_all_anchors(im_c=cfg.TRAIN.SEARCH_SIZE // 2,
                                 size=cfg.TRAIN.OUTPUT_SIZE)
    anchor_box = anchors.all_anchors[0]
    anchor_center = anchors.all_anchors[1]
    shape = np.ones(len(anchor_center.shape) + 1, dtype=np.int32)
    shape[0] = batch_size
    anchor_center_ = np.tile(anchor_center, shape)
    anchor_box_ = np.tile(anchor_box, shape)
    anchor_center_ = torch.from_numpy(anchor_center_).permute(0, 2, 3, 4, 1).reshape(-1, 4)
    anchor_box_ = torch.from_numpy(anchor_box_).permute(0, 2, 3, 4, 1).reshape(-1, 4)

    return anchor_center_, anchor_box_


def convert_bbox(delta, anchor_center):
    """
    :param delta: N * k * h * w, 4
    :param anchor_center: N * k * h * w, 4
    :return: N * k * h * w, 4
    """
    anchor = torch.zeros_like(delta)
    bbox = torch.zeros_like(delta)
    anchor[:] = anchor_center

    bbox[:, 0] = delta[:, 0] * anchor[:, 2] + anchor[:, 0]
    bbox[:, 1] = delta[:, 1] * anchor[:, 3] + anchor[:, 1]
    bbox[:, 2] = torch.exp(delta[:, 2]) * anchor[:, 2]
    bbox[:, 3] = torch.exp(delta[:, 3]) * anchor[:, 3]
    # # limit min cx cy, w, h and max cx, cy, w, h
    bbox_ = torch.zeros_like(bbox)
    # bbox_[:, 0] = torch.clamp(bbox[:, 0], min=0, max=cfg.TRAIN.SEARCH_SIZE)
    # bbox_[:, 1] = torch.clamp(bbox[:, 1], min=0, max=cfg.TRAIN.SEARCH_SIZE)
    bbox_[:, 2] = torch.clamp(bbox[:, 2], min=10, max=cfg.TRAIN.SEARCH_SIZE)
    bbox_[:, 3] = torch.clamp(bbox[:, 3], min=10, max=cfg.TRAIN.SEARCH_SIZE)
    return bbox_


def center2bbox(center):
    """
    :param center: N * k * h * w, 4
    :return: corner: n * k * h * w, 4
    """
    corner = torch.zeros_like(center)
    corner[:, 0] = center[:, 0] - center[:, 2] * 0.5
    corner[:, 1] = center[:, 1] - center[:, 3] * 0.5
    corner[:, 2] = center[:, 0] + center[:, 2] * 0.5
    corner[:, 3] = center[:, 1] + center[:, 3] * 0.5
    return corner


def compute_IoU(pred_corner, target_corner):
    """
    :param pred_corner: N * k * h * w, 4
    :param target_corner: N * k * h * w, 4
    :return: iou: N * k * h * w
    """
    x1_p, y1_p, x2_p, y2_p = pred_corner[:, 0], pred_corner[:, 1], pred_corner[:, 2], pred_corner[:, 3]
    x1_t, y1_t, x2_t, y2_t = target_corner[:, 0], target_corner[:, 1], target_corner[:, 2], target_corner[:, 3]

    xx1 = torch.maximum(x1_p, x1_t)
    yy1 = torch.maximum(y1_p, y1_t)
    xx2 = torch.minimum(x2_p, x2_t)
    yy2 = torch.minimum(y2_p, y2_t)
    area_intersect = torch.zeros_like(xx1)
    mask = (yy2 > yy1) * (xx2 > xx1)
    area_intersect[mask] = (xx2[mask] - xx1[mask]) * (yy2[mask] - yy1[mask])

    w_t = (x2_t - x1_t)
    h_t = (y2_t - y1_t)
    w_p = (x2_p - x1_p)
    h_p = (y2_p - y1_p)
    area_union = w_t * h_t + w_p * h_p - area_intersect + 1e-7
    iou = area_intersect / area_union
    # print("iou: ", torch.max(iou), torch.min(iou))
    return iou


def pred2center(loc):
    """
    :param loc: N * k * h * w, 4, predicted delta by rpn head
    :return: pred_center: N * k * h * w, 4
    """
    if (loc == torch.zeros_like(loc)).all():
        return loc
    anchor_center, _ = gen_anchor(cfg.TRAIN.BATCH_SIZE)
    # N * k * h * w, 4
    pred_center = convert_bbox(loc, anchor_center)
    return pred_center


def pred2corner(loc):
    """
    :param loc: N * k * h * w, 4, predicted delta by rpn head
    :return: pred_corner: N * k * h * w, 4
    """
    if (loc == torch.zeros_like(loc)).all():
        return loc
    # N * k * h * w, 4
    pred_center = pred2center(loc)
    pred_corner = center2bbox(pred_center)
    return pred_corner
