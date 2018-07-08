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
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from fast_rcnn.nms_wrapper import nms
from nms.py_cpu_nms import py_cpu_nms

DEBUG = True

class ProposalLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        if not hasattr(self, 'param_str_'):
            self.param_str_= self.param_str #adapt to new caffe version
        layer_params = yaml.load(self.param_str_)
        self._feat_stride = layer_params['feat_stride']
        self._anchors = np.array([1, 1, self._feat_stride, 1, self._feat_stride, self._feat_stride, 1, self._feat_stride]) - 1.0
        self._num_anchors = 1
        self._kernel_size = 1
        if layer_params.has_key('kernel_size'):
            self._kernel_size = layer_params['kernel_size']
        # if DEBUG:
        #     print 'feat_stride: {}'.format(self._feat_stride)
        #     print 'anchors:'
        #     print self._anchors

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[0].reshape(1, 9)

        # scores blob: holds scores for R regions of interest
        if len(top) > 1:
            top[1].reshape(1, 1, 1, 1)

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'

        cfg_key = str(self.phase)
        if len(cfg_key) == 1:
            cfg_key = 'TEST' if cfg_key == '1' else 'TRAIN'
        print cfg_key # either 'TRAIN' or 'TEST'

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        scores = bottom[0].data[:, self._num_anchors:, :, :]
        # print 'scores.shape', scores.shape
        bbox_deltas = bottom[1].data
        im_info = bottom[2].data[0, :]
        # print bottom[3].data.shape



        if DEBUG:
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])

        # 1. Generate proposals from bbox deltas and shifted anchors
        height, width = scores.shape[-2:]

        if DEBUG:
            print 'score map size: {}'.format(scores.shape)

        # Enumerate all shifts
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride

        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        # Enumerate all shifted anchors:
        # 4->5
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        anchors = self._anchors.reshape((1, A, 8)) + \
                  shifts.reshape((1, K, 8)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 8))
        total_anchors = int(K * A)

        if DEBUG:
            print 'anchors.shape', anchors.shape
        
        # anchor transform
        #################################################################
        origin_anchors = anchors.copy()
        coord_scale = self._feat_stride / 1.0
        base_box = np.array([-0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5]) * (self._kernel_size - 1.0)
        if len(bottom)>4:
            proj = bottom[4].data.copy()
            new_proj = proj.transpose(0, 2, 3, 1)
            if proj.shape[1] == 8:
                proj_mat = np.hstack( (new_proj.reshape((K, 8)), np.zeros((K, 1))) )
                proj_mat += np.tile(np.identity(3).reshape((1,-1)), (K, 1))
                for idx in range(total_anchors):
                    x_ctr = origin_anchors[idx, 0] / coord_scale
                    y_ctr = origin_anchors[idx, 1] / coord_scale
                    for p in range(4):
                        point = np.dot(proj_mat[idx, :].reshape((3, 3)), np.hstack((base_box[2*p:2*p+2], 1)).transpose()).transpose()
                        anchors[idx, 2*p] = point[0] / point[2]*1.0 + x_ctr
                        anchors[idx, 2*p+1] = point[1] / point[2]*1.0 + y_ctr
            elif proj.shape[1] == 3:
                pass
                # proj_mat = new_proj.reshape((K, 3))

                # for idx in range(total_anchors):
                #     x_ctr = (anchors[idx, 0] + anchors[idx, 4]) / 2.0
                #     y_ctr = (anchors[idx, 1] + anchors[idx, 5]) / 2.0
                #     T = proj_mat[idx, :]
                #     for p in range(4):
                #         x = anchors[idx, 2*p] - x_ctr
                #         y = anchors[idx, 2*p+1] - y_ctr
                #         sx = (T[1] + 1.0) * x;
                #         sy = (T[2] + 1.0) * y;
                #         x_new = np.cos(T[0]) * sx - np.sin(T[0]) * sy;
                #         y_new = np.sin(T[0]) * sx + np.cos(T[0]) * sy;
                #         anchors[idx, 2*p] = x_new + x_ctr
                #         anchors[idx, 2*p+1] = y_new + y_ctr
            else:
                print 'projection matrix has wrong channel number (only allow 3 or 8)!'

            anchors = anchors * coord_scale 
            anchors = clip_boxes(anchors, im_info[:2])
        #################################################################


        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 8))

        # Same story for the scores:
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas, im_info)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info[:2])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        # keep = _filter_boxes(proposals, min_size * im_info[2])
        # proposals = proposals[keep, :]
        # scores = scores[keep]
        # print 'filter done:', proposals.shape[0]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 1000(6000))

#########################NMS######################################
#        # 6. apply nms (e.g. threshold = 0.7)
#        # 7. take after_nms_topN (e.g. 100(300))
#        # 8. return the top proposals (-> RoIs top)
#        start = time.time()
#
#        keep = py_cpu_nms(np.hstack((proposals, scores)), nms_thresh)
#        proposals = proposals[keep, :]
#        scores = scores[keep]
#
#       if DEBUG:
#            print 'post', post_nms_topN
#            print 'num keep proposals', proposals.shape
#            nms_time = time.time()
#            print 'nms done:', proposals.shape[0]
#            print 'nms time: %.5f'%(nms_time - start)
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
