# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
import math

def bbox_transform(ex_rois, gt_rois, im_info):
    ex_x1 = ex_rois[:, 0]
    ex_y1 = ex_rois[:, 1]
    ex_x2 = ex_rois[:, 2]
    ex_y2 = ex_rois[:, 3]
    ex_x3 = ex_rois[:, 4]
    ex_y3 = ex_rois[:, 5]
    ex_x4 = ex_rois[:, 6]
    ex_y4 = ex_rois[:, 7]
    ex_w1 = np.sqrt((np.fabs(ex_x1-ex_x2)+1.0)**2 + (np.fabs(ex_y1-ex_y2)+1.0)**2)
    ex_w2 = np.sqrt((np.fabs(ex_x3-ex_x4)+1.0)**2 + (np.fabs(ex_y3-ex_y4)+1.0)**2)
    ex_h1 = np.sqrt((np.fabs(ex_x1-ex_x4)+1.0)**2 + (np.fabs(ex_y1-ex_y4)+1.0)**2)
    ex_h2 = np.sqrt((np.fabs(ex_x3-ex_x2)+1.0)**2 + (np.fabs(ex_y3-ex_y2)+1.0)**2)
    # ex_w1 = im_info[1]
    # ex_w2 = im_info[1]
    # ex_h1 = im_info[0]
    # ex_h2 = im_info[0]
    # ex_ctr_x = (ex_x1 + ex_x2 + ex_x3 + ex_x4) / 4
    # ex_ctr_y = (ex_y1 + ex_y2 + ex_y3 + ex_y4) / 4

    gt_x1 = gt_rois[:, 0]
    gt_y1 = gt_rois[:, 1]
    gt_x2 = gt_rois[:, 2]
    gt_y2 = gt_rois[:, 3]
    gt_x3 = gt_rois[:, 4]
    gt_y3 = gt_rois[:, 5]
    gt_x4 = gt_rois[:, 6]
    gt_y4 = gt_rois[:, 7]


    targets_dx1 = (gt_x1 - ex_x1) / ex_w1
    targets_dy1 = (gt_y1 - ex_y1) / ex_h1
    targets_dx2 = (gt_x2 - ex_x2) / ex_w1
    targets_dy2 = (gt_y2 - ex_y2) / ex_h2
    targets_dx3 = (gt_x3 - ex_x3) / ex_w2
    targets_dy3 = (gt_y3 - ex_y3) / ex_h2
    targets_dx4 = (gt_x4 - ex_x4) / ex_w2
    targets_dy4 = (gt_y4 - ex_y4) / ex_h1

    targets = np.vstack(
        (targets_dx1, targets_dy1, targets_dx2, targets_dy2, targets_dx3, targets_dy3, targets_dx4, targets_dy4)).transpose()
    return targets

def bbox_transform_inv(boxes, deltas, im_info):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)
    boxes = boxes.astype(deltas.dtype, copy=False)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    x3 = boxes[:, 4]
    y3 = boxes[:, 5]
    x4 = boxes[:, 6]
    y4 = boxes[:, 7]

    # w1 = im_info[1]
    # w2 = im_info[1]
    # h1 = im_info[0]
    # h2 = im_info[0]

    w1 = np.sqrt((np.fabs(x1-x2) + 1)**2 + (np.fabs(y1-y2) + 1)**2)
    w2 = np.sqrt((np.fabs(x3-x4) + 1)**2 + (np.fabs(y3-y4) + 1)**2)
    h1 = np.sqrt((np.fabs(x1-x4) + 1)**2 + (np.fabs(y1-y4) + 1)**2)
    h2 = np.sqrt((np.fabs(x3-x2) + 1)**2 + (np.fabs(y3-y2) + 1)**2)

    dx1 = deltas[:, 0::8]
    dy1 = deltas[:, 1::8]
    dx2 = deltas[:, 2::8]
    dy2 = deltas[:, 3::8]
    dx3 = deltas[:, 4::8]
    dy3 = deltas[:, 5::8]
    dx4 = deltas[:, 6::8]
    dy4 = deltas[:, 7::8]

    pred_x1 = dx1 * w1[:, np.newaxis] + x1[:, np.newaxis]
    pred_y1 = dy1 * h1[:, np.newaxis] + y1[:, np.newaxis]
    pred_x2 = dx2 * w1[:, np.newaxis] + x2[:, np.newaxis]
    pred_y2 = dy2 * h2[:, np.newaxis] + y2[:, np.newaxis]
    pred_x3 = dx3 * w2[:, np.newaxis] + x3[:, np.newaxis]
    pred_y3 = dy3 * h2[:, np.newaxis] + y3[:, np.newaxis]
    pred_x4 = dx4 * w2[:, np.newaxis] + x4[:, np.newaxis]
    pred_y4 = dy4 * h1[:, np.newaxis] + y4[:, np.newaxis]


    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::8] = pred_x1
    # y1
    pred_boxes[:, 1::8] = pred_y1
    # x2
    pred_boxes[:, 2::8] = pred_x2
    # y2
    pred_boxes[:, 3::8] = pred_y2
    # x3
    pred_boxes[:, 4::8] = pred_x3
    # y3
    pred_boxes[:, 5::8] = pred_y3
    # x4
    pred_boxes[:, 6::8] = pred_x4
    # y4
    pred_boxes[:, 7::8] = pred_y4

    return pred_boxes

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """

    # 0 <= x <= im_shape[1]
    boxes[:, 0::8] = np.maximum(np.minimum(boxes[:, 0::8], im_shape[1] - 1), 0)
    boxes[:, 2::8] = np.maximum(np.minimum(boxes[:, 2::8], im_shape[1] - 1), 0)
    boxes[:, 4::8] = np.maximum(np.minimum(boxes[:, 4::8], im_shape[1] - 1), 0)
    boxes[:, 6::8] = np.maximum(np.minimum(boxes[:, 6::8], im_shape[1] - 1), 0)

    # 0 <= y < im_shape[0]
    boxes[:, 1::8] = np.maximum(np.minimum(boxes[:, 1::8], im_shape[0] - 1), 0)
    boxes[:, 3::8] = np.maximum(np.minimum(boxes[:, 3::8], im_shape[0] - 1), 0)
    boxes[:, 5::8] = np.maximum(np.minimum(boxes[:, 5::8], im_shape[0] - 1), 0)
    boxes[:, 7::8] = np.maximum(np.minimum(boxes[:, 7::8], im_shape[0] - 1), 0)

    return boxes



