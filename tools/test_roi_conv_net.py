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
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use',
                        default=0, type=int)
    parser.add_argument('--test_proto', dest='test_prototxt',
                        help='model directory name ',
                        default='./proto/test.pt', type=str)
    parser.add_argument('--model', dest='model_name',
                        help='model directory name (e.g., "roi_dconv_regular")',
                        default='roi_dconv_regular', type=str)
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
    parser.add_argument('--thres', dest='threshold',
                        help='score threshold for evaluation',
                        default=0.98, type=float)
    parser.add_argument('--overwrite', dest='overwrite',
                        help='score threshold for evaluation',
                        default=0, type=int)

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

def roi_generate(queue=None, imdb_name=None, roi_conv_model_path=None, cfg=None,
                 test_prototxt=None, overwrite=None):
    """Use a trained RPN to generate proposals.
    """
    output_dir = './output/'
    net_name = os.path.splitext(os.path.basename(roi_conv_model_path))[0]
    output_path_name = os.path.join(output_dir,net_name)
    queue.put({'proposal_path': output_path_name})

    if not os.path.exists(output_path_name):
        os.makedirs(output_path_name)
    elif overwrite:
        shutil.rmtree(output_path_name)
        os.makedirs(output_path_name)
    else:
        return

    print 'RPN model: {}'.format(roi_conv_model_path)
    print('Using config:')
    pprint.pprint(cfg)

    # fix the random seeds (numpy and caffe) for reproducibility
    import caffe
    np.random.seed(cfg.RNG_SEED)
    caffe.set_random_seed(cfg.RNG_SEED)
    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)


    # NOTE: the matlab implementation computes proposals on flipped images, too.
    # We compute them on the image once and then flip the already computed
    # proposals. This might cause a minor loss in mAP (less proposal jittering).

    from datasets.factory import get_imdb
    imdb = get_imdb(imdb_name)
    print 'Loaded dataset `{:s}` for bbox generation'.format(imdb.name)
    # Load RPN and configure output directory
    roi_net = caffe.Net(test_prototxt, roi_conv_model_path, caffe.TEST)
    roi_net.name = net_name
    print 'Output will be saved to `{:s}`'.format(output_path_name)
    print 'roinet.name: ',roi_net.name
    # Generate proposals on the imdb
    roi_proposals = imdb_proposals(roi_net, imdb, output_path_name)



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

    # queue for communicated results between processes
    mp_queue = mp.Queue()
    # solves, iters, etc. for each training stage
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Testing Stage'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    mp_kwargs = dict(
            queue=mp_queue,
            imdb_name="test",
            roi_conv_model_path=args.pretrained_model,
            cfg=cfg,
            test_prototxt=args.test_prototxt,
            overwrite=args.overwrite)
    p = mp.Process(target=roi_generate, kwargs=mp_kwargs)
    p.start()
    proposal_path = mp_queue.get()['proposal_path']
    p.join()


    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Evaluation Stage'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    import tools.evaluation as eval
    eval.THRES=args.threshold
    eval.proposal_dir_name=proposal_path
    print '1. Proposal dir: ',eval.proposal_dir_name
    print '2, Threshold:  ',eval.THRES
    eval.main()
