# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
import sys
import utils.bbox_overlap as bbox_olp

def py_soft_nms(dets, thresh, sigma = 0.5, drop_thres = 0.01):
    boxes = dets[:,:8]
    scores = dets[:, 8]
    olp = bbox_olp.plg_nms_area(boxes, sample_range=100)
    keep = []
    N = boxes.shape[0]
    remained = [i for i in range(N)]

    while len(remained) > 0:
        remained_max = scores[remained].argmax()
        maxpos = remained[remained_max]
        keep.append(maxpos)
        remained.remove(maxpos)
        # update scores
        remained_list = remained[:]
        for pos in remained_list:
            iou = olp[maxpos, pos]
            if iou>0:
                weight = 1.0 - iou #if iou > thresh else 1.0
                # weight = np.exp(-(iou * iou) / sigma)
                scores[pos] = weight * scores[pos]
                if scores[pos] <= drop_thres:
                    remained.remove(pos)
    return boxes[keep, :], scores[keep, np.newaxis]

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    boxes = dets[:,:8]
    scores = dets[:, 8]
    order = scores.argsort()[::-1]
    olp = bbox_olp.plg_nms_overlaps(boxes, sample_range=100)

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = olp[i, order[1:]]
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return boxes[keep, :], scores[keep, np.newaxis]

if __name__ == '__main__':
    boxes = np.array([[7,20,22,27,17,40,9,33,0.3], [9,11,25,28,20,37,7,30,0.7], [3,10,33,16,24,17,10,18,0.1] ,[2,11,30,15,27,18,7,18,0.5]])
    thresh = 0.4
    a = py_cpu_nms(boxes, thresh)
    print a
