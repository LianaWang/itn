# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

# from datasets.synthetic_text import synthetic_text
from datasets.text import text
from datasets.text_eight import text_eight
# from datasets.coco import coco
import numpy as np
import os
import scipy.io as sio
# Set up voc_<year>_<split> using selective search "fast" mode
#for year in ['2007', '2012']:
    #for split in ['train', 'val', 'trainval', 'test']:
        #name = 'voc_{}_{}'.format(year, split)
        #__sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# name = 'train'
# name = 'test'

# filename = os.path.join('/home/wff/data/dataset/SynthText/', 'gt.mat')
# data = sio.loadmat(filename)
# __sets[name] = (lambda name=name, data=data, devkit=None: synthetic_text(name,data,devkit))

# Set up coco_2014_<split>
#for year in ['2014']:
    #for split in ['train', 'val', 'minival', 'valminusminival']:
        #name = 'coco_{}_{}'.format(year, split)
        #__sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
#for year in ['2015']:
    #for split in ['test', 'test-dev']:
        #name = 'coco_{}_{}'.format(year, split)
        #__sets[name] = (lambda split=split, year=year: coco(split, year))

def get_imdb(name):
    __sets[name] = (lambda name=name, devkit=None: text_eight(name,devkit))
            
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
