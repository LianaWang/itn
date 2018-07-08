import os
import sys
import numpy as np
sys.path.append('/home/wff/py-faster-rcnn/lib/')
import cPickle
# from fast_rcnn.config import cfg
from utils.bbox_overlap import bbox_overlaps
from utils.bbox_overlap import coordinates_transfer
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


# load ground-truth
def _load_text_annotation(index):

    filename = os.path.join('/mnt/disk/wangfangfang/synthetic/annotations/', index + '.txt')
    with open(filename) as f:
        texts = f.readlines()
    num_objs = len(texts)
    boxes = np.zeros((num_objs, 5), dtype=np.float64)

    # Load object bounding boxes into a data frame.
    for ix in range(num_objs):
        objs = texts[ix].split()
        x1 = float(objs[0])
        y1 = float(objs[1])
        x2 = float(objs[2])
        y2 = float(objs[3])
        the = float(objs[4]) 
        boxes[ix, :] = [x1, y1, x2, y2, the]
    
    return boxes, hard_flg

def vis_results(imname, gt_boxes):
    filename = os.path.join('/home/wff/py-faster-rcnn/data/Multi-orient/JPGImages', imname)
    from PIL import Image
    im=Image.open(filename)

    boxes,_,_ = coordinates_transfer(gt_boxes)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im)

    for bbox in boxes:
        ax.add_patch(Polygon(list(bbox.exterior.coords)[:-1],fill=False,color='green',linewidth=3))

    plt.axis('off')
    plt.tight_layout()
    plt.savefig('./tools/result/%s.png'%index)

    return


# recall
if __name__ == '__main__':

    filename = '/mnt/disk/wangfangfang/synthetic/SynthText/gt.mat'
    data = sio.loadmat(filename)
    for idx in range(100):
        gt_boxes = _load_text_annotation(idx)
        imname = data['imnames'][0, idx]
        print idx
        vis_results(str(imname)[3:-2], gt_boxes)

        


