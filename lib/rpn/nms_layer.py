# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import numpy as np
import yaml
import math
import time
from utils.timer import Timer
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from fast_rcnn.nms_wrapper import nms
from nms.py_cpu_nms import py_cpu_nms
from nms.py_cpu_nms import py_soft_nms
DEBUG = True

class NMSLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[0].reshape(1, 9)

        # scores blob: holds scores for R regions of interest
        if len(top) > 1:
            top[1].reshape(1, 1, 1, 1)

    def forward(self, bottom, top):
        cfg_key = str(self.phase)
        if len(cfg_key) == 1:
            cfg_key = 'TEST' if cfg_key == '1' else 'TRAIN'
        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH

        #bottom
        proposals = bottom[0].data[:,1:] #(N, 8)
        scores = bottom[1].data #(N, 1)
        #keep scores>=0.5
        keep_inds = np.where(scores >= 0.5)[0]
        proposals = proposals[keep_inds, :]
        scores = scores[keep_inds, :]
#########################NMS######################################
        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 100(300))
        # 8. return the top proposals (-> RoIs top)
        nms_time = Timer()
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        if DEBUG:
            print 'pre', len(order)
        proposals = proposals[order, :]
        scores = scores[order]

        nms_time.tic()
        proposals, scores = py_soft_nms(np.hstack((proposals, scores)), nms_thresh)

        if DEBUG:
            print 'num keep proposals', proposals.shape
            nms_time.toc()
            print 'nms done:', proposals.shape[0]
            print 'nms time: %.5f'%(nms_time.average_time)
#########################END of NMS######################################

        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        top[0].reshape(*(blob.shape))
        top[0].data[...] = blob

        # [Optional] output scores blob
        if len(top) > 1:
            top[1].reshape(*(scores.shape))
            top[1].data[...] = scores


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
