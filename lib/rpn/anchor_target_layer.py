# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import os
import caffe
import yaml
from fast_rcnn.config import cfg
import numpy as np
import numpy.random as npr
import utils.bbox_overlap as bbox_olp
# from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import bbox_transform, clip_boxes
import math
from utils.timer import Timer
import cv2

DEBUG = False
OVERLAP_TEST = False

class AnchorTargetLayer(caffe.Layer):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        if not hasattr(self, 'param_str_'):
            self.param_str_= self.param_str #adapt to new caffe version
        layer_params = yaml.load(self.param_str_)
        self._feat_stride = layer_params['feat_stride']
        self._anchors = np.array([1, 1, self._feat_stride, 1, self._feat_stride, self._feat_stride, 1, self._feat_stride]) - 1.0
        self._num_anchors = 1
        self._kernel_size = 1
        if layer_params.has_key('kernel_size'):
            self._kernel_size = layer_params['kernel_size']

        if DEBUG:
            self._counts = cfg.EPS
            self._sums = np.zeros((1, 8))
            self._squared_sums = np.zeros((1, 8))
            self._fg_sum = 0
            self._bg_sum = 0
            self._count = 0

        height, width = bottom[0].data.shape[-2:]

        if DEBUG:
            print 'AnchorTargetLayer: height', height, 'width', width

        A = self._num_anchors
        # labels
        top[0].reshape(1, 1, A * height, width)
        # bbox_targets
        top[1].reshape(1, A * 8, height, width)
        # bbox_inside_weights
        top[2].reshape(1, A * 8, height, width)
        # bbox_outside_weights
        top[3].reshape(1, A * 8, height, width)
        # homography
        if len(top)>4:
            top[4].reshape(1, A * 8, height, width)

        if len(top)>5:
            top[5].reshape(1, A * 8, height, width)
    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors
        # measure GT overlap

        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'

        # map of shape (..., H, W)
        height, width = bottom[0].data.shape[-2:]
        # GT boxes (x1, y1, x2, y2, label)
        gt_boxes = bottom[1].data
        # print 'GTBOXES', gt_boxes
        # im_info
        im_info = bottom[2].data[0, :]
        # print 'im_info', im_info



        # if DEBUG:
            # print ''
            # print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            # print 'scale: {}'.format(im_info[2])
            # print 'height, width: ({}, {})'.format(height, width)
            # print 'rpn: gt_boxes.shape', gt_boxes.shape
            #print 'rpn: gt_boxes', gt_boxes

        # 1. Generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride

        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        # add A anchors (1, A, 4) tost
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        all_anchors = (self._anchors.reshape((1, A, 8)) +
                       shifts.reshape((1, K, 8)).transpose((1, 0, 2)))
        anchors = all_anchors.reshape((K * A, 8))
        total_anchors = int(K * A)

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


        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((total_anchors, ), dtype=np.float32)
        labels.fill(-1)

        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)
        if OVERLAP_TEST:
            timer = Timer()
            timer.tic()

        ###########Use origin 16x16 anchors to detect positive or negative######################
        overlaps = bbox_olp.bbox_intersection(
            np.ascontiguousarray(origin_anchors, dtype=np.float),
            np.ascontiguousarray(gt_boxes[:, :8], dtype=np.float))

        if OVERLAP_TEST:
            timer.toc()
            print 'speed: {:.3f}s / intersect_total'.format(timer.average_time)

        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(total_anchors), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]


        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps <= cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps <= cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # computing homograph for each positive anchor
        # regression using Euclidean loss between H(predicted bottom[4] matrix) and G(computed homograph matrix)
        # non-positive anchors do not contribute to regression, so let G = H to produce 0 loss.
        #################################################################
        if len(top)>4:
            ####projective affine transformation##################
            height, width = bottom[0].data.shape[-2:]
            proj_G_L = np.zeros((K,8))
            for idx in range(total_anchors):
                x_ctr = (origin_anchors[idx, 0] + origin_anchors[idx, 4]) / 2.0 / coord_scale
                y_ctr = (origin_anchors[idx, 1] + origin_anchors[idx, 5]) / 2.0 / coord_scale
                gt_idx_current = argmax_overlaps[idx]
                if labels[idx] == 1:
                    current_anchor = base_box + np.tile([x_ctr, y_ctr],(4,))
                    current_gt = (gt_boxes[gt_idx_current, :8]) / coord_scale
                    G = _find_affine(current_anchor, current_gt)
                    G_L = G.copy()
                    G_L[[0,1,3,4]] *= cfg.TRAIN.MATRIX_SCALE
                    proj_G_L[idx, :6] = G_L[:6] - np.array([1.0, 0, 0, 0, 1.0, 0])
            proj_targets_large = proj_G_L.reshape((1,height,width,8))
            top_proj_targets_large = proj_targets_large.transpose(0, 3, 1, 2)
            top[4].reshape(*top_proj_targets_large.shape)
            top[4].data[...] = top_proj_targets_large
            #################################################################



        # subsample negative labels if we have too many
        num_fg = np.sum(labels == 1)
        num_bg = np.int32(num_fg * cfg.TRAIN.BG_NUM_SCALE) #keep top fg*3 hard negatives
        bg_inds = np.where(labels == 0)[0]
        fg_inds = np.where(labels == 1)[0]
        if len(bg_inds) > num_bg:
            all_bg_scores = bottom[0].data[0,0,:,:].reshape((-1,))
            bg_scores = all_bg_scores[bg_inds]
            bg_score_order = bg_scores.argsort() #low score -> high, hard neg -> easy neg
            num_bg = num_bg if num_bg>0 else 1
            easy_neg_idx = bg_score_order[num_bg:] #keep top fg*3 hard negatives
            easy_neg_idx = np.random.choice(easy_neg_idx, len(easy_neg_idx) - num_bg, replace=False)
            disable_inds =  bg_inds[easy_neg_idx]
            labels[disable_inds] = -1
        if len(bg_inds) < num_fg:
            all_fg_scores = bottom[0].data[0,1,:,:].reshape((-1,))
            fg_scores = all_fg_scores[fg_inds]
            fg_score_order = fg_scores.argsort() #low score -> high, hard neg -> easy neg
            num_bg = len(bg_inds) if len(bg_inds)>0 else 1
            easy_pos_idx = fg_score_order[num_bg:] #keep top bg number of hard positives
            disable_inds =  fg_inds[easy_pos_idx]
            labels[disable_inds] = -1

        bbox_targets = np.zeros((total_anchors, 8), dtype=np.float32)
        bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :], im_info)

        bbox_inside_weights = np.zeros((total_anchors, 8), dtype=np.float32)
        bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)
        bbox_outside_weights = np.zeros((total_anchors, 8), dtype=np.float32)

        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(labels >= 0)
            positive_weights = np.ones((1, 8)) * 1.0 / num_examples
            negative_weights = np.ones((1, 8)) * 1.0 / num_examples
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
            positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                                np.sum(labels == 1))
            negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                                np.sum(labels == 0))
        bbox_outside_weights[labels == 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights

        if 0:
            # print 'labels:', labels[0], labels[23], labels[167], labels[500]
            # print 'according anchors:', anchors[0,:], anchors[23,:], anchors[167,:], anchors[500,:]
            self._sums += bbox_targets[labels == 1, :].sum(axis=0)
            self._squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
            self._counts += np.sum(labels == 1)
            means = self._sums / self._counts
            stds = np.sqrt(self._squared_sums / self._counts - means ** 2)
            print 'means:'
            print means
            print 'stdevs:'
            print stds

        # map up to original set of anchors
        # labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        # bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        # bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
        # bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

        if DEBUG:
            print 'rpn: max max_intersect', np.max(max_overlaps)
            print 'rpn: num_positive', np.sum(labels == 1)
            print 'rpn: num_negative', np.sum(labels == 0)
            print 'rpn: num_ignored',np.sum(labels == -1)
            self._fg_sum += np.sum(labels == 1)
            self._bg_sum += np.sum(labels == 0)
            self._count += 1
            print 'rpn: num_positive avg', self._fg_sum / self._count
            print 'rpn: num_negative avg', self._bg_sum / self._count

        # labels
        labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        labels = labels.reshape((1, 1, A * height, width))
        #print 'reshaped rpn labels:', labels.shape
        top[0].reshape(*labels.shape)
        top[0].data[...] = labels

        # bbox_targets
        bbox_targets = bbox_targets \
            .reshape((1, height, width, A * 8)).transpose(0, 3, 1, 2)
        #print 'rpn bbox_targets shape:', bbox_targets.shape
        top[1].reshape(*bbox_targets.shape)
        top[1].data[...] = bbox_targets

        # bbox_inside_weights
        bbox_inside_weights = bbox_inside_weights \
            .reshape((1, height, width, A * 8)).transpose(0, 3, 1, 2)
        assert bbox_inside_weights.shape[2] == height
        assert bbox_inside_weights.shape[3] == width
        #print 'rpn bbox_inside_weights shape:', bbox_inside_weights.shape
        top[2].reshape(*bbox_inside_weights.shape)
        top[2].data[...] = bbox_inside_weights

        # bbox_outside_weights
        bbox_outside_weights = bbox_outside_weights \
            .reshape((1, height, width, A * 8)).transpose(0, 3, 1, 2)
        assert bbox_outside_weights.shape[2] == height
        assert bbox_outside_weights.shape[3] == width
        #print 'rpn bbox_outside_weights shape:', bbox_outside_weights.shape
        top[3].reshape(*bbox_outside_weights.shape)
        top[3].data[...] = bbox_outside_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois, im_info):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 8
    assert gt_rois.shape[1] == 9

    return bbox_transform(ex_rois, gt_rois[:, :8], im_info).astype(np.float32, copy=False)

def _find_homo(anchor, gt_box):
    """Compute homography matrix, output is a 3x3 matrix, transform center is the center of this anchor,
    substract identity"""
    x_ctr = (anchor[0] + anchor[4]) / 2.0
    y_ctr = (anchor[1] + anchor[5]) / 2.0
    pts_src = np.float32([[anchor[0] - x_ctr, anchor[1] - y_ctr],
                [anchor[2] - x_ctr, anchor[3] - y_ctr],
                [anchor[4] - x_ctr, anchor[5] - y_ctr],
                [anchor[6] - x_ctr, anchor[7] - y_ctr]])

    pts_dst = np.float32([[gt_box[0] - x_ctr, gt_box[1] - y_ctr],
                [gt_box[2] - x_ctr, gt_box[3] - y_ctr],
                [gt_box[4] - x_ctr, gt_box[5] - y_ctr],
                [gt_box[6] - x_ctr, gt_box[7] - y_ctr]])

    H, mask = cv2.findHomography(pts_src, pts_dst)
    return H.reshape((-1,))

def _find_affine(anchor, gt_box):
    x_ctr = (anchor[0] + anchor[4]) / 2.0
    y_ctr = (anchor[1] + anchor[5]) / 2.0
    pts_src = np.float32([[anchor[0] - x_ctr, anchor[1] - y_ctr],
                [anchor[2] - x_ctr, anchor[3] - y_ctr],
                [anchor[4] - x_ctr, anchor[5] - y_ctr],
                [anchor[6] - x_ctr, anchor[7] - y_ctr]])

    pts_dst = np.float32([[gt_box[0] - x_ctr, gt_box[1] - y_ctr],
                [gt_box[2] - x_ctr, gt_box[3] - y_ctr],
                [gt_box[4] - x_ctr, gt_box[5] - y_ctr],
                [gt_box[6] - x_ctr, gt_box[7] - y_ctr]])

    pts_src = np.hstack((pts_src, np.ones((4,1))))
    H = np.linalg.lstsq(pts_src, pts_dst)[0].T
    return H.reshape((-1,))
