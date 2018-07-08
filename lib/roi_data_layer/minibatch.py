# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
import os
import sys
#sys.path.append('/home/wangfangfang/py-faster-rcnn/lib/')
import numpy as np
import numpy.random as npr
# import random
import cv2
from fast_rcnn.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob

def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

    # Get the input image blob, formatted for caffe
    im_blob, im_scales, gt_boxes = _get_image_blob(roidb, random_scale_inds)

    blobs = {'data': im_blob}

    assert len(roidb) == 1, "Single batch only"
    # print 'gt_boxes when getting minibatch:', gt_boxes
    blobs['gt_boxes'] = gt_boxes
    blobs['im_info'] = np.array(
        [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
        dtype=np.float32)

    return blobs


def _augment_data(img, gt_boxes):
    if cfg.TRAIN.DATA_AUGMENT == False:
        return img, gt_boxes
    H = img.shape[0]
    W = img.shape[1]
    # random data augmentation
    angle = 15 * (np.random.rand()*2.0-1.0) * np.pi / 180.0
    if cfg.TRAIN.DATA_AUGMENT_ROTATION == False:
        angle = 0
    scale = 1.0 + (np.random.rand()*2.0-1.0)*0.2
    if cfg.TRAIN.DATA_AUGMENT_SCALE == False:
        scale = 1.0
    shift_x = 0
    shift_y = 0
    #affine transformation matrix
    c_x = W / 2.0
    c_y = H / 2.0
    alpha = scale * np.cos(angle)
    beta = scale * np.sin(angle)
    center_x = (1-alpha)*c_x - beta*c_y
    center_y = beta*c_x + (1-alpha)*c_y
    M = np.array([[alpha, beta, center_x + shift_x],
                 [-beta, alpha, center_y + shift_y]])
    #apply affine transformation
    newImg = cv2.warpAffine(img, M, (W, H))
    new_gt_list = []
    for n in range(len(gt_boxes)):
        new_gt = np.zeros((8,))
        for i in range(4):
            new_gt[2*i:2*i+2]=np.dot(M,np.hstack([gt_boxes[n,2*i:2*i+2],1]))
        #new_gt validation, whole new_gt out of image, then exclude it
        if new_gt[[0,2,4,6]].max() < 0 or new_gt[[1,3,5,7]].max() < 0 \
           or new_gt[[0,2,4,6]].min() > W or new_gt[[1,3,5,7]].min() > H:
               continue
        #new_gt validation, part of new_gt box goes out of image, then no augmentation
        if new_gt.min() < 0 or new_gt[[0,2,4,6]].max() > W or new_gt[[1,3,5,7]].max() > H:
            return img, gt_boxes
        new_gt_list.append(new_gt)
    if len(new_gt_list) == 0:
        return img, gt_boxes
    new_gt = np.array(new_gt_list)

    return newImg, new_gt

def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    # gt boxes: (x1, y1, x2, y2, theta, cls) 5->6,4->5
    im = cv2.imread(roidb[0]['image'])
    if im is None:
        print "Read image failed:", roidb[0]['image']

    if roidb[0]['flipped']:
        im = im[:, ::-1, :]

    gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
    im, new_gt_boxes = _augment_data(im, roidb[0]['boxes'][gt_inds, 0:8])

    gt_boxes = np.empty((len(new_gt_boxes), 9), dtype=np.float32)
    gt_boxes[:, 0:8] = new_gt_boxes

    target_size = cfg.TRAIN.SCALES[scale_inds[0]]
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                    cfg.TRAIN.MAX_SIZE)
    im_scales.append(im_scale)
    processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
    gt_boxes[:, 0:8] *= im_scales
    gt_boxes[:, 8] = roidb[0]['gt_classes'][gt_inds[:len(new_gt_boxes)]]
    return blob, im_scales, gt_boxes

def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = np.zeros(im_rois.shape)
    rois[:,0:8] = im_rois[:,0:8] * im_scale_factor
    return rois

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 8 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = int(clss[ind])
        start = 8 * cls
        end = start + 8
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
        # print 'bbox_targets', bbox_targets.shape
    return bbox_targets, bbox_inside_weights

def _vis_minibatch(im_blob, rois_blob, labels_blob, overlaps):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    for i in xrange(rois_blob.shape[0]):
        rois = rois_blob[i, :]
        im_ind = rois[0]
        roi = rois[1:]
        im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        cls = labels_blob[i]
        plt.imshow(im)
        print 'class: ', cls, ' overlap: ', overlaps[i]
        plt.gca().add_patch(
            plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor='r', linewidth=3)
            )
        plt.show()
