#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Faster R-CNN network using alternating optimization.
This tool implements the alternating optimization algorithm described in our
NIPS 2015 paper ("Faster R-CNN: Towards Real-time Object Detection with Region
Proposal Networks." Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.)
"""

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from rpn.generate import imdb_proposals
import argparse
import pprint
import numpy as np
import sys, os
import multiprocessing as mp
import cPickle
import shutil

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a ROI_conv network')
    parser.add_argument('--max_iter', dest='max_iter',
                        help='max training iters',
                        default=200000, type=int)
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use',
                        default=0, type=int)
    parser.add_argument('--model', dest='model_name',
                        help='model directory name (e.g., "roi_dconv_regular")',
                        default='roi_dconv_regular', type=str)
    parser.add_argument('--solver', dest='solver',
                        help='solver_name',
                        default='./proto/solver.pt', type=str)
    parser.add_argument('--net_name', dest='net_name',
                        help='network name (e.g., "ZF")',
                        default='VGG16', type=str)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default="./data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel", type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args

def get_roidb(imdb_name, rpn_file=None):
    from fast_rcnn.train import get_training_roidb
    from datasets.factory import get_imdb
    imdb = get_imdb(imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
    if rpn_file is not None:
        imdb.config['rpn_file'] = rpn_file
    roidb = get_training_roidb(imdb)
    return roidb, imdb
# ------------------------------------------------------------------------------
# Pycaffe doesn't reliably free GPU memory when instantiated nets are discarded
# (e.g. "del net" in Python code). To work around this issue, each training
# stage is executed in a separate process using multiprocessing.Process.
# ------------------------------------------------------------------------------

def _init_caffe(cfg):
    """Initialize pycaffe in a training process.
    """

    import caffe
    # fix the random seeds (numpy and caffe) for reproducibility
    np.random.seed(cfg.RNG_SEED)
    caffe.set_random_seed(cfg.RNG_SEED)
    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)

def train_roi_conv(imdb_name=None, init_model=None, solver=None,
              max_iter=None, cfg=None):
    """Train a Region Proposal Network in a separate training process.
    """
    #imdb_name: train

    _init_caffe(cfg)

    import caffe
    roidb, imdb = get_roidb(imdb_name)

    from fast_rcnn.train import train_net
    model_paths = train_net(solver, roidb, './',
                            pretrained_model=init_model,
                            max_iters=max_iter)
    print 'Training are finished in train process'

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    cfg.GPU_ID = args.gpu_id

    # --------------------------------------------------------------------------
    # Pycaffe doesn't reliably free GPU memory when instantiated nets are
    # discarded (e.g. "del net" in Python code). To work around this issue, each
    # training stage is executed in a separate process using
    # multiprocessing.Process.
    # --------------------------------------------------------------------------

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Training Stage'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    # Not using any proposals, just ground-truth boxes
    print('Using config:')
    pprint.pprint(cfg)

    mp_kwargs = dict(
            imdb_name="train",
            init_model=args.pretrained_model,
            solver=args.solver,
            max_iter=cfg.TRAIN.MAX_TRAIN_ITERS,
            cfg=cfg)
    p = mp.Process(target=train_roi_conv, kwargs=mp_kwargs)
    p.start()
    p.join()
    print 'Training exit!'
