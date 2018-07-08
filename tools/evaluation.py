import os
import sys
import numpy as np
sys.path.append('./lib/')
import cPickle
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import scipy.io as sio
from fast_rcnn.config import cfg
from pycuda import autoinit
import utils.bbox_overlap as bbox_olp

THRES=0.98
proposal_dir_name=""
VIS_TYPE = 3 #only eval (0), vis + eval (1), only vis (2), only save txt (3)
# load results
def _load_proposals(index):

    filename = os.path.join(proposal_dir_name, index + '.txt')
    if not os.path.exists(filename):
        return None
    with open(filename) as f:
        texts = f.readlines()
    num_objs = len(texts)

    if num_objs == 0:
        return None
    proposals = np.zeros((num_objs, 9), dtype=np.float64)

    # Load object bounding boxes into a data frame.
    for ix in range(num_objs):
        objs = texts[ix].split()
        proposals[ix, :] = np.array(objs)
    proposals = proposals[np.where(proposals[:, -1] > THRES)[0], :]
    return proposals


# load ground-truth
def _load_text_annotation(index):
    filename = os.path.join(cfg.DATA_DIR, 'Annotations', index + '.gt')
    if not os.path.exists(filename):
        return None,np.array([])
    with open(filename) as f:
        texts = f.readlines()
    num_objs = len(texts)

    boxes = np.zeros((num_objs, 8), dtype=np.float64)
    hard_flgs = np.zeros((num_objs), dtype=np.int32)

    # Load object bounding boxes into a data frame.
    for ix in range(num_objs):
        objs = texts[ix].split()
        hard_flgs[ix] = int(objs[0])
        boxes[ix,:] = np.array(objs[1:9])

    return boxes, hard_flgs

def save_submit(index, preds):
    import utils.bbox_overlap as bbox_olp
    submit_dir= os.path.join(proposal_dir_name,'submit_roi')
    if preds is None or preds.shape[0]<=0:
        return
    if not os.path.exists(submit_dir):
        os.makedirs(submit_dir)
    res_name = '%s'%(index)
    if 'ICDAR2015' in cfg.DATA_DIR: 
        res_name = 'res_img_%d.txt'%(int(index)-1000);
    res_filename = os.path.join(submit_dir, res_name)
    np.savetxt(res_filename, preds, fmt = '%.6f')

def vis_results(index, gt_boxes, proposals, predict_tags, gt_tags):
    filename = os.path.join(cfg.DATA_DIR, 'JPGImages', index + '.jpg')
    from PIL import Image
    im=Image.open(filename)
    import utils.bbox_overlap as bbox_olp
    boxes = bbox_olp.overlap_coord_transfer(gt_boxes)
    preds = bbox_olp.overlap_coord_transfer(proposals)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im)
    w = 3
    line_w = 5
    gt_color = 'lime'
    pred_color = 'red'
    for idx in range(len(boxes)):
        bbox = boxes[idx]
        a = list(np.float32(bbox.T))
        ax.add_patch(Polygon([[a[0]-w,a[1]-w], [a[2]+w,a[3]-w], [a[4]+w,a[5]+w], [a[6]-w,a[7]+w]],fill=False,color=gt_color,linewidth=w))
    for idx in range(len(preds)):
        pred = preds[idx]
        a = list(np.float32(pred.T))
        ax.add_patch(Polygon([[a[0]-w,a[1]-w], [a[2]+w,a[3]-w], [a[4]+w,a[5]+w], [a[6]-w,a[7]+w]],fill=False,color=pred_color,linewidth=line_w))
    plt.axis('off')
    plt.tight_layout()
    vis_dir= os.path.join(proposal_dir_name,'vis')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    plt.savefig('%s/%s.png'%(vis_dir,index))
    plt.close('all')
    return

def main():
    filename = os.path.join(cfg.DATA_DIR, 'ImageSets/Main/test.txt')
    with open(filename) as f:
        indexes = f.readlines()
    num_index = len(indexes)
    idx = []
    for ix in range(num_index):
        index = indexes[ix].split()
        idx.append(index[0])

    num_all_posi = 0
    num_all_gt = 0
    num_all_pred = 0

    import time
    for index in idx:
        all_start=time.time()
        num_posi = 0
        proposals = _load_proposals(index)
        if VIS_TYPE == 3:
            save_submit(index, proposals)
            continue

        gt_boxes, hard_flgs = _load_text_annotation(index)
        idx_easy = np.where(hard_flgs[:] == 0)[0]
        num_gt = gt_boxes.shape[0] if gt_boxes is not None else 0
        num_pred = proposals.shape[0] if proposals is not None else 0
        num_all_gt += num_gt
        num_all_pred += num_pred
        if VIS_TYPE == 2:
            vis_results(index, gt_boxes, proposals,None,None)
            print 'index: %s / %s'%(index, idx[-1])
            continue
        if num_pred==0:
            num_all_gt -= np.sum(hard_flgs)
            continue
        if num_gt==0:
            continue
        start=time.time()

        overlaps = bbox_olp.plg_overlaps(proposals, gt_boxes)

        argmax_overlaps = overlaps.argmax(axis = 0)

        max_overlaps = overlaps.max(axis = 0)

        arg_recall = argmax_overlaps[np.where(max_overlaps>0.5)[0]]

        hard_flg_missed = hard_flgs[np.where(max_overlaps<=0.5)[0]]
        num_ignore = np.sum(hard_flg_missed)

        hard_flg_recalled = hard_flgs[np.where(max_overlaps>0.5)[0]]
        num_hard_pos =  np.sum(hard_flg_recalled)

        num_all_gt -=  num_ignore

        num_posi = np.sum(max_overlaps > 0.5)

        predict_tags = np.ones((len(proposals),)) #right detections, yellow
        gt_tags = np.ones((len(idx_easy),))

        predict_tags[np.where(overlaps.max(axis = 1) <=0.5)[0]] = 0 #wrong detections, red
        gt_tags[max_overlaps[idx_easy]<0.5] = 0
        if VIS_TYPE > 0:
            vis_results(index, gt_boxes[idx_easy, :], proposals, predict_tags, gt_tags)
        num_all_posi += num_posi
        print 'index: %s / %s'%(index, idx[-1]), '  time: %.2f'%(time.time() - all_start), ' (num_pred, num_gt): (%2d, %2d)'%(num_pred, num_gt)

    recall = num_all_posi * 1.0 / num_all_gt if num_all_gt != 0 else 0
    precision = num_all_posi * 1.0 / num_all_pred if num_all_pred != 0 else 0
    f_measure = ( 2.0 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

    print 'exp_dir = ', os.path.basename(os.getcwd())
    print 'model = ', proposal_dir_name[proposal_dir_name[:-1].rfind('/')+1:]
    print 'threshold = ', THRES
    print 'recall = %.6f'%(recall)
    print 'precision = %.6f'%(precision)
    print 'f-measure = %.6f'%(f_measure)
    print 'num_gt = ', num_all_gt
    print 'num_pred = ', num_all_pred




if __name__ == '__main__':
    main()
