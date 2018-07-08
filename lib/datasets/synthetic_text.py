# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
import sys
sys.path.append('./lib/')
from datasets.imdb import imdb
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import cPickle
import subprocess
import uuid
from fast_rcnn.config import cfg
import math

class synthetic_text(imdb):
    def __init__(self, image_set, data, devkit_path=None):
        imdb.__init__(self, 'SynthText')
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path)
        self._data = data
        self._classes = ('__background__', 'text')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_index = data['imnames'][0, :50000]
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # print index
        image_path = os.path.join(self._data_path, str(index)[3:-2])
        # print image_path
        image_path = str(image_path)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    # def _load_image_set_index(self):
    #     """
    #     Load the indexes listed in this dataset's image set file.
    #     """
    #     # Example path to image set file:
    #     # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
    #     image_set_file = os.path.join(self._data_path, self._image_set + '.txt')
    #     assert os.path.exists(image_set_file), \
    #             'Path does not exist: {}'.format(image_set_file)
    #     # data = sio.loadmat(image_set_file)
    #     # image_index = data.imnames
    #     # print 'image_index_1', image_index{1}
    #     with open(image_set_file) as f:
    #         image_index = [x.strip() for x in f.readlines()]
    #     return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join('/home/wff/data/dataset/SynthText/')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        # cache_file = os.path.join(self.cache_path, 'SynthText_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_syntext_annotation(index)
                    for index in range(self._image_index.shape[0])]

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        #if int(self._year) == 2007 or self._image_set != 'test':
        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def rpn_roidb(self):
        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_syntext_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        # print index
        objs = self._data['wordBB'][0, index]
        # print 'objs_shape:', objs.shape
        num_objs = objs.size/8
        # print 'num_objs', num_objs
        boxes = np.zeros((num_objs, 5), dtype=np.float64)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix in range(num_objs):
            if num_objs == 1:
                x1 = float(objs[0, 0])
                x2 = float(objs[0, 1])
                x3 = float(objs[0, 2])
                x4 = float(objs[0, 3])
                y1 = float(objs[1, 0])
                y2 = float(objs[1, 1])
                y3 = float(objs[1, 2])
                y4 = float(objs[1, 3])
            else:
                x1 = float(objs[0, 0, ix])
                x2 = float(objs[0, 1, ix])
                x3 = float(objs[0, 2, ix])
                x4 = float(objs[0, 3, ix])
                y1 = float(objs[1, 0, ix])
                y2 = float(objs[1, 1, ix])
                y3 = float(objs[1, 2, ix])
                y4 = float(objs[1, 3, ix])

            x_ctr = (x1 + x2 + x3 + x4) / 4
            y_ctr = (y1 + y2 + y3 + y4) / 4
            w_ = (np.sqrt((x1-x2)**2 + (y1-y2)**2) + np.sqrt((x3-x4)**2 + (y3-y4)**2)) / 2
            h_ = (np.sqrt((x1-x4)**2 + (y1-y4)**2) + np.sqrt((x3-x2)**2 + (y3-y2)**2)) / 2

            if w_ >= h_:
                w = w_
                h = h_
                x_t1 = (x1 + x4) / 2
                y_t1 = (y1 + y4) / 2
                x_t2 = (x2 + x3) / 2
                y_t2 = (y2 + y3) / 2
                
            else:
                w = h_
                h = w_
                x_t1 = (x1 + x2) / 2
                y_t1 = (y1 + y2) / 2
                x_t2 = (x4 + x3) / 2
                y_t2 = (y4 + y3) / 2

            if x_t2 - x_t1 == 0:
                the = math.pi/2 if y_t2 - y_t1 >= 0 else -math.pi/2
            else:
                the = np.maximum(np.minimum(np.arctan((y_t2 - y_t1) / (x_t2 - x_t1)), math.pi/2), -math.pi/2)

            x_1 = np.round(x_ctr - w/2)
            x_2 = np.round(x_ctr + w/2)
            y_1 = np.round(y_ctr - h/2)
            y_2 = np.round(y_ctr + h/2)
            
            cls = self._class_to_ind['text'] # set class to be 'text'
            boxes[ix, :] = [x_1, y_1, x_2, y_2, the]
            gt_classes[ix] = cls
            # hard_flg[ix] = int(objs[1])
            #_theta[ix,:] = [math.cos(the),math.sin(the)]
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x_2 - x_1 + 1) * (y_2 - y_1 + 1)
        overlaps = scipy.sparse.csr_matrix(overlaps)

        ######################################################################################

        det_file = os.path.join('/home/wff/data/dataset/SynthText/annotations/', str(index) + '.txt')
        # print 'det_file:', det_file
        np.savetxt(det_file, boxes, fmt = '%.6f')

        #####################################################################################

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                #'gt_theta': _theta,
                # 'hard_flg': hard_flg,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            'VOC' + self._year,
            'Main',
            filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))    


    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.synthetic_text import synthetic_text
    filename = os.path.join('/home/wff/data/dataset/SynthText/', 'gt.mat')
    data = sio.loadmat(filename)
    # print type(data)
    # imnames = data['imnames']
    # wordBB = data['wordBB']
    # print imnames.shape
    # print wordBB.shape
    # print wordBB[0, 3]
    # print wordBB[0, 3].shape
    d = synthetic_text('train', data)
    res = d.roidb
    from IPython import embed; embed()
