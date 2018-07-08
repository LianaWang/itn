######################GPU CUDA VERSION: zhaoliming@zju.edu.cn####################################
import Polygon as plg
import numpy as np
import time
######################CUDA IMPORTS################################
THREAD_NUM = 1024
from contextlib import contextmanager
import pycuda.driver as cuda
import pycuda.gpuarray
from pycuda.compiler import SourceModule
@contextmanager
def cuda_context():
    in_caffe = True
    try:
        ctx = cuda.Context.attach()
    except:
        import pycuda.autoinit
        in_caffe = False
#        print "pyCUDA not in CAFFE, use pycuda.autoinit in bbox_overlap.py"
    try:
        yield
    finally:
        if in_caffe:
            ctx.detach()
######################CUDA FUNCTION#############################
kernel_overlap="""
__global__ void cuda_transfer_one(const float *box, float *vertex, int n, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= K)
    {
        return;
    }

    float x1 = box[idx * n + 0];
    float y1 = box[idx * n + 1];
    float x3 = box[idx * n + 2];
    float y3 = box[idx * n + 3];
    float x2 = x3;
    float y2 = y1;
    float x4 = x1;
    float y4 = y3;
    float _theta = -box[idx * n + 4];
    float x_ctr = 0.5 * (x1 + x2);
    float y_ctr = 0.5 * (y1 + y4);

    int dim = 9; //one vertex has dim values
    vertex[idx * dim + 0] = round((x1 - x_ctr) * cos(_theta) + (y1 - y_ctr) * sin(_theta) + x_ctr);
    vertex[idx * dim + 1] = round((y1 - y_ctr) * cos(_theta) - (x1 - x_ctr) * sin(_theta) + y_ctr);
    vertex[idx * dim + 2] = round((x2 - x_ctr) * cos(_theta) + (y2 - y_ctr) * sin(_theta) + x_ctr);
    vertex[idx * dim + 3] = round((y2 - y_ctr) * cos(_theta) - (x2 - x_ctr) * sin(_theta) + y_ctr);
    vertex[idx * dim + 4] = round((x3 - x_ctr) * cos(_theta) + (y3 - y_ctr) * sin(_theta) + x_ctr);
    vertex[idx * dim + 5] = round((y3 - y_ctr) * cos(_theta) - (x3 - x_ctr) * sin(_theta) + y_ctr);
    vertex[idx * dim + 6] = round((x4 - x_ctr) * cos(_theta) + (y4 - y_ctr) * sin(_theta) + x_ctr);
    vertex[idx * dim + 7] = round((y4 - y_ctr) * cos(_theta) - (x4 - x_ctr) * sin(_theta) + y_ctr);
    vertex[idx * dim + 8] = - _theta;

    //area[idx] = (box[idx * n + 2] - box[idx * n + 0] + 1) * (box[idx * n + 3] - box[idx * n + 1] + 1);
}

static inline __device__ int in_poly(const float x, const float y, const float *vertex)
{
    int nr_edges_crossing = 0;
    for(int i = 0; i < 8; i += 2)
    {
        float x_cur = vertex[i];
        float x_next = vertex[(i + 2) % 8];
        float y_cur = vertex[i + 1];
        float y_next = vertex[(i + 3) % 8];
        int parallel = (y_cur == y_next);
        // for an upward edge
        int t1 = (x < x_cur + (y - y_cur) * (x_next - x_cur) * 1 / (y_next - y_cur + parallel));
        int intersect =
            (y_next > y_cur) &
            (
                ((y == y_cur) & ((x < x_cur) | (x == x_cur))) |
                (
                    (y > y_cur) & (y < y_next) & t1
                )
            );

        // for a downward edge
        int t2 = (x < x_next + (y - y_next) * (x_next - x_cur) * 1 / (y_next - y_cur + parallel));
        intersect |=
            (y_next < y_cur) &
            (
                ((y == y_next) & ((x < x_next) | (x == x_next))) |
                (
                    (y > y_next) & (y < y_cur) &  t2
                )
            );
        nr_edges_crossing += intersect & (!parallel);

    }
    return (nr_edges_crossing & 1);
}

__global__ void cuda_testpoint(float *vertex,
                             float *gt_vertex,
                             int *cirbox,
                             bool *all_areas,
                             int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= K) {
        return;
    }

    //int w = cirbox[1] - cirbox[0] + 1;
    int h = cirbox[3] - cirbox[2] + 1;
    int i = cirbox[0] + idx / h;
    int j = cirbox[2] + idx % h;
    int in1 = in_poly(i, j, vertex);
    int in2 = in_poly(i, j, gt_vertex);
    all_areas[K * 0 + idx] = (bool) in1 & in2; //area_inter;
    all_areas[K * 1 + idx] = (bool) in1 | in2; //area_union;
    all_areas[K * 2 + idx] = (bool) in1; //area_anchor;
}

static inline __device__ float in_overlap(float *cur_box,
					  float *gt_box,
                                          float sample_range,
					  int ignore_union)
{
    //circum rectangle for current vertex
    float x_min = 9999, x_max = -9999, y_min = 9999, y_max = -9999;
    for(int i = 0; i < 8; i += 2)
    {
        if(cur_box[i] < x_min)
            x_min = cur_box[i];
        else if(cur_box[i] > x_max)
            x_max = cur_box[i];
        if(cur_box[i + 1] < y_min)
            y_min = cur_box[i + 1];
        else if(cur_box[i + 1] > y_max)
            y_max = cur_box[i + 1];
    }

    //circum rectangle for gt vertex
    float qx_min = 9999, qx_max = -9999, qy_min = 9999, qy_max = -9999;
    for(int i = 0; i < 8; i += 2)
    {
        if(gt_box[i] < qx_min)
            qx_min = gt_box[i];
        else if(gt_box[i] > qx_max)
            qx_max = gt_box[i];
        if(gt_box[i + 1] < qy_min)
            qy_min = gt_box[i + 1];
        else if(gt_box[i + 1] > qy_max)
            qy_max = gt_box[i + 1];
    }

    //no overlap between two circum rectangels
    if((x_max < qx_min) || (y_max < qy_min)
            || (qx_max < x_min) || (qy_max < y_min))
    {
        return 0.0;
    }

    float area_inter = 0.0, area_union = 0.0, area_anchor = 0.0;

    float range = sample_range; //sampling up to 'range' points for each side of the box
    float xstep = (x_max - x_min + 1.0) / range;
    float ystep = (y_max - y_min + 1.0) / range;
    xstep = xstep < 1 ? 1.0 : xstep;
    ystep = ystep < 1 ? 1.0 : ystep;

    for(float i = x_min; i <= x_max; i += xstep)
    {
        for(float j = y_min; j <= y_max; j += ystep)
        {
            int in1 = in_poly(i, j, cur_box);
            int in2 = in_poly(i, j, gt_box);
            area_inter += in1 & in2;
            area_union += in1 | in2;
            area_anchor += in1;
        }
    }

    //skip some special cases for faster implementation
    if(area_inter <= 0){ //no overlap
        return 0.0;
    }
    if(ignore_union == 0) {
        float qxstep = (qx_max - qx_min + 1.0) / range;
        float qystep = (qy_max - qy_min + 1.0) / range;
        qxstep = qxstep < 1 ? 1 : qxstep;
        qystep = qystep < 1 ? 1 : qystep;

        for(float i = qx_min; i <= qx_max; i += qxstep)
        {
            for(float j = qy_min; j <= qy_max; j += qystep)
            {
                if((i >= x_min) && (i <= x_max) && (j >= y_min) && (j <= y_max))
                    continue;
                int in1 = in_poly(i, j, cur_box);
                int in2 = in_poly(i, j, gt_box);
                area_inter += in1 & in2;
                area_union += in1 | in2;
            }
        }
    }
    float area_div = ignore_union ? area_anchor: area_union;
    float overlap = area_inter>0 ? area_inter / area_div : 0.0;
    return overlap;
}

__global__ void cuda_overlap(float *vertex,
                             float *gt_vertex,
                             float *overlaps,
                             int K, int N, int dim,
                             float range, int ignore_union)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gt_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if(idx >= K || gt_idx >= N){
        return;
    }

    float *cur_box = vertex + idx * dim;
    float *gt_box = gt_vertex + gt_idx * dim;

    float overlap = in_overlap(cur_box, gt_box, range, ignore_union);
    overlaps[idx * N + gt_idx] = overlap;
}

__global__ void cuda_nms_overlap(float *vertex,
                                 float *overlaps,
                                 int K, int dim,
                                 float range)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gt_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if(idx >= gt_idx || idx >= K || gt_idx >= K ){
        return;
    }

    float *cur_box = vertex + idx * dim;
    float *gt_box =  vertex + gt_idx * dim;

    float overlap = in_overlap(cur_box, gt_box, range, 0);
    overlaps[idx * K + gt_idx] = overlap;
}

__global__ void cuda_area(float *vertex, float *areas, int K, int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= K)
    {
        return;
    }

    //circum rectangle for current vertex
    float x_min = 9999, x_max = -9999, y_min = 9999, y_max = -9999;
    for(int i = 0; i < 8; i += 2)
    {
        if(vertex[idx * dim + i] < x_min)
            x_min = vertex[idx * dim + i];
        else if(vertex[idx * dim + i] > x_max)
            x_max = vertex[idx * dim + i];
        if(vertex[idx * dim + i + 1] < y_min)
            y_min = vertex[idx * dim + i + 1];
        else if(vertex[idx * dim + i + 1] > y_max)
            y_max = vertex[idx * dim + i + 1];
    }

    float cur_area = 0.0;
    float *cur_vertex = vertex + idx * dim;
    for(float i = x_min; i <= x_max; i += 1.0)
    {
        for(float j = y_min; j <= y_max; j += 1.0)
        {
            int in = in_poly(i, j, cur_vertex);
            cur_area += in;
        }
    }
    areas[idx] = cur_area;
}

"""
######################END OF CUDA FUNCS#######################
with cuda_context():
    mod = SourceModule(kernel_overlap)
    cuda_transfer = mod.get_function("cuda_transfer_one")
    cuda_area = mod.get_function("cuda_area")
    cuda_overlap = mod.get_function("cuda_overlap")
    cuda_nms_overlap = mod.get_function("cuda_nms_overlap")
    cuda_testpoint = mod.get_function("cuda_testpoint")

###########################transfer 5 values to 8 point values#################################

def overlap_coord_transfer(gt_boxes):
    if gt_boxes is None:
        return np.array([])
    if len(gt_boxes)<=0:
        return gt_boxes
    K, n = gt_boxes.shape
    if n >= 8:
        return gt_boxes[:,:8]
    vertex = np.ones((K, 9), dtype = np.float32)*(-1.0)

    with cuda_context():
        cuda_transfer(cuda.In(gt_boxes.astype(np.float32)),
                      cuda.Out(vertex),
                      np.int32(n),np.int32(K),
                      block=(THREAD_NUM,1,1),grid=(K/THREAD_NUM+1,1))
    return vertex.astype(np.float)

#####################################################################
def gpu_areas(vertex):
    K, dim = vertex.shape #vertex.shape (K,8)
    if K <= 0:
        return None
    areas = np.zeros((K,1), dtype=np.float32)

    with cuda_context():
        cuda_area(cuda.In(vertex.astype(np.float32)),
                cuda.Out(areas), np.int32(K), np.int32(dim),
                block=(THREAD_NUM,1,1),grid=(K/THREAD_NUM+1,1))
    return areas.astype(np.float)

def circum_box(vertex_one):
    x_min = 9999; x_max = -9999; y_min = 9999; y_max = -9999;
    for i in range(0, 8, 2):
        if vertex_one[i] < x_min:
            x_min = vertex_one[i];
        elif vertex_one[i] > x_max :
            x_max = vertex_one[i];
        if vertex_one[i + 1] < y_min:
            y_min = vertex_one[i + 1];
        elif vertex_one[i + 1] > y_max:
            y_max = vertex_one[i + 1];
    return [x_min, x_max, y_min, y_max]


def testpoint_overlaps(vertex, gt_vertex):
    K, dim = vertex.shape
    N, dim2 = gt_vertex.shape
    assert dim >= 8
    assert dim2 >=8
    vertex = vertex[:, :8]
    gt_vertex = gt_vertex[:, :8]
    overlaps = np.zeros((K,N), dtype=np.float)

    #Maximum circumscribed bounding box
    gt_cirbox=[]
    anchor_cirbox=[]
    for query in gt_vertex:
        gt_cirbox.append(circum_box(query))
    for anchor in vertex:
        anchor_cirbox.append(circum_box(anchor))

    #calculate overlaps for all pairs
    with cuda_context():
        for i, anchor in enumerate(vertex):
            x_min, x_max, y_min, y_max = anchor_cirbox[i]
            for j, query in enumerate(gt_vertex):
                qx_min, qx_max, qy_min, qy_max = gt_cirbox[j]

                #no overlap between two circum rectangels
                if ((x_max < qx_min) or (y_max < qy_min)
                        or (qx_max < x_min) or (qy_max < y_min)):
                    overlaps[i,j] = 0
                    continue

                cirbox = np.zeros((4,), dtype=np.int32)
                cirbox[0] = x_min if x_min < qx_min else qx_min
                cirbox[1] = x_max if x_max > qx_max else qx_max
                cirbox[2] = y_min if y_min < qy_min else qy_min
                cirbox[3] = y_max if y_max > qy_max else qy_max

                #cuda calculate the overlap
                num_kernels = (cirbox[1] - cirbox[0] + 1) * (cirbox[3] - cirbox[2] + 1)
                all_areas = np.zeros((num_kernels*3, ), dtype=np.bool) #inter_area,union_area,anchor_area

                cuda_testpoint(cuda.In(anchor.astype(np.float32)),
                               cuda.In(query.astype(np.float32)),
                               cuda.In(cirbox), #width
                               cuda.Out(all_areas),
                               np.int32(num_kernels),
                               block=(THREAD_NUM,1,1),grid=(num_kernels/THREAD_NUM+1,1,1))
                inter_area = np.sum(all_areas[num_kernels*0:num_kernels*1])
                union_area = np.sum(all_areas[num_kernels*1:num_kernels*2])
                anchor_area = np.sum(all_areas[num_kernels*2:num_kernels*3])
                overlaps[i,j] = 1.0*inter_area / union_area if union_area>0 else 0
    return overlaps

def loopboxes_overlaps(vertex, gt_vertex, sample_range, ignore_union):
    K, dim = vertex.shape
    N, dim2 = gt_vertex.shape
    assert dim >= 8
    assert dim2 >=8
    vertex = vertex[:, :8]
    gt_vertex = gt_vertex[:, :8]
    overlaps = np.zeros((K, N), dtype=np.float32)

    with cuda_context():
        cuda_overlap(cuda.In(vertex.astype(np.float32)),cuda.In(gt_vertex.astype(np.float32)),
                cuda.Out(overlaps), np.int32(K), np.int32(N), np.int32(8),
                np.float32(sample_range), np.int32(ignore_union),
                block=(THREAD_NUM,1,1),grid=(K/THREAD_NUM+1,N,1))
    return overlaps.astype(np.float)

def loopnms_overlaps(vertex, sample_range):
    K, dim = vertex.shape
    assert dim >=8
    vertex = vertex[:, :8]
    overlaps = np.zeros((K, K), dtype=np.float32)

    with cuda_context():
        cuda_nms_overlap(cuda.In(vertex.astype(np.float32)),cuda.Out(overlaps),
                np.int32(K), np.int32(8), np.float32(sample_range),
                block=(THREAD_NUM,1,1),grid=(K/THREAD_NUM+1,K,1))
    for i in range(K):
        for j in range(i+1):
            if j < i:
                overlaps[i, j] = overlaps[j, i]
            elif j== i:
                overlaps[i, j] = 1.0
    return overlaps.astype(np.float)


def preprocess_boxes(boxes, gt_boxes):
    boxes  = overlap_coord_transfer(boxes)
    gt_boxes = overlap_coord_transfer(gt_boxes)
    return boxes, gt_boxes

def opt_overlaps(boxes, gt_boxes):
    boxes, gt_boxes=preprocess_boxes(boxes, gt_boxes)
    overlaps = testpoint_overlaps(boxes, gt_boxes)
    return overlaps

def nms_overlaps(boxes, sample_range=100):
    boxes  = overlap_coord_transfer(boxes)
    overlaps = loopnms_overlaps(boxes, sample_range)
    return overlaps

def plg_nms_overlaps(pred_boxes, sample_range=None):
    pred_boxes  = overlap_coord_transfer(pred_boxes)
    boxes = [plg.Polygon(box.reshape((4,2))) for box in pred_boxes]
    overlaps = np.zeros((len(boxes), len(boxes)), dtype=np.float32)

    for k in range(len(boxes)):
        for n in range(k,len(boxes)):
            inter = boxes[k] & boxes[n]
            area_inter = inter.area() if len(inter)>0 else 0
            area_union = boxes[k].area() + boxes[n].area() - area_inter
            overlaps[k,n] = area_inter / area_union
            overlaps[n,k] = overlaps[k,n]
    return overlaps

def plg_nms_area(pred_boxes, sample_range=None):
    pred_boxes  = overlap_coord_transfer(pred_boxes)
    boxes = [plg.Polygon(box.reshape((4,2))) for box in pred_boxes]
    overlaps = np.zeros((len(boxes), len(boxes)), dtype=np.float32)

    for k in range(len(boxes)):
        for n in range(k,len(boxes)):
            inter = boxes[k] & boxes[n]
            area_inter = inter.area() if len(inter)>0 else 0
            overlaps[k,n] = area_inter / boxes[n].area() if boxes[n].area()>0 else 1
            overlaps[n,k] = area_inter / boxes[k].area() if boxes[k].area()>0 else 1
    return overlaps



def bbox_overlaps(boxes, gt_boxes, sample_range=100):
    boxes, gt_boxes=preprocess_boxes(boxes, gt_boxes)
    overlaps = loopboxes_overlaps(boxes, gt_boxes, sample_range, ignore_union=0)
    return overlaps

def bbox_intersection(boxes, gt_boxes, sample_range=100):
    boxes, gt_boxes=preprocess_boxes(boxes, gt_boxes)
    overlaps = loopboxes_overlaps(boxes, gt_boxes, sample_range, ignore_union=1)
    return overlaps

def plg_overlaps(pred_boxes, gt_boxes):
    pred_boxes, gt_boxes=preprocess_boxes(pred_boxes, gt_boxes)
    boxes = [plg.Polygon(box.reshape((4,2))) for box in pred_boxes]
    gts = [plg.Polygon(box.reshape((4,2))) for box in gt_boxes]
    overlaps = np.zeros((len(boxes), len(gts)), dtype=np.float32)

    for k in range(len(boxes)):
        for n in range(len(gts)):
            inter = boxes[k] & gts[n]
            area_inter = inter.area() if len(inter)>0 else 0
            area_union = boxes[k].area() + gts[n].area() - area_inter
            overlaps[k,n] = area_inter / area_union
    return overlaps

if __name__ == '__main__':
#    boxes = [[3,10,33,16,0,0.4], [9,17,25,28,0.1,0.3], [7,20,22,27,0.9,0.5], [7,20,22,27,0.9,0.5]]
#    gt_boxes = [[2,11,30,15,0.3,0.7]]
    boxes = [[18, 16, 265, 11, 1466, 952, 180, 55],
	     [16, 88, 265, 87, 965, 928, 167, 128],
	     [14, 25, 256, 25, 957, 958, 105, 58],
	     [17, 86, 824, 83, 985, 814, 18, 116],
	     [17, 91, 265, 89, 965, 926, 167, 127],
	     [92, 80, 159, 84, 960, 921, 92, 117],
	     [16, 24, 172, 24, 973, 962, 107, 62],
	     [9, 80, 159, 85, 960, 821, 95, 117],
	     [13, 22, 156, 23, 956, 959, 104, 58],
	     [11, 22, 157, 23, 957, 959, 102, 59]]

    import time
    start = time.time()
    a = nms_overlaps(np.array(boxes*80)*1.0)
    finish = time.time()
    b = plg_nms_overlaps(np.array(boxes*80)*1.0)
    print 'cuda func, time: ', finish - start,' \n', a
    print 'plg  func, time: ', time.time() - finish, '\n', b

