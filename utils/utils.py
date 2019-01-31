#coding=utf-8
import numpy as np
import torch
import os
import torch.nn.functional as F
def class_weights():
    """
    COCO train2014每个样本类的频率
    是用于处理样本不均衡.
    “样本偏斜是指数据集中正负类样本数量不均，比如正类样本有10000个，负类样本只有100个，
    这就可能使得超平面被“推向”负类（因为负类数量少，分布得不够广），影响结果的准确性。
    """
    weights = 1 / torch.FloatTensor(
        [187437, 4955, 30920, 6033, 3838, 4332, 3160, 7051, 7677, 9167, 1316, 1372, 833, 6757, 7355, 3302, 3776, 4671,
         6769, 5706, 3908, 903, 3686, 3596, 6200, 7920, 8779, 4505, 4272, 1862, 4698, 1962, 4403, 6659, 2402, 2689,
         4012, 4175, 3411, 17048, 5637, 14553, 3923, 5539, 4289, 10084, 7018, 4314, 3099, 4638, 4939, 5543, 2038, 4004,
         5053, 4578, 27292, 4113, 5931, 2905, 11174, 2873, 4036, 3415, 1517, 4122, 1980, 4464, 1190, 2302, 156, 3933,
         1877, 17630, 4337, 4624, 1075, 3468, 135, 1380])
    weights /= weights.sum()
    # tensor([1.4458e-04, 5.4690e-03, 8.7642e-04, 4.4918e-03, 7.0606e-03, 6.2555e-03,
    #         8.5756e-03, 3.8433e-03, 3.5299e-03, 2.9561e-03, 2.0592e-02, 1.9751e-02,
    #         3.2532e-02, 4.0105e-03, 3.6844e-03, 8.2068e-03, 7.1766e-03, 5.8015e-03,
    #         4.0034e-03, 4.7492e-03, 6.9342e-03, 3.0010e-02, 7.3518e-03, 7.5358e-03,
    #         4.3708e-03, 3.4216e-03, 3.0868e-03, 6.0153e-03, 6.3433e-03, 1.4554e-02,
    #         5.7682e-03, 1.3812e-02, 6.1546e-03, 4.0695e-03, 1.1282e-02, 1.0078e-02,
    #         6.7544e-03, 6.4907e-03, 7.9445e-03, 1.5896e-03, 4.8073e-03, 1.8621e-03,
    #         6.9077e-03, 4.8924e-03, 6.3182e-03, 2.6873e-03, 3.8613e-03, 6.2816e-03,
    #         8.7444e-03, 5.8428e-03, 5.4867e-03, 4.8888e-03, 1.3297e-02, 6.7679e-03,
    #         5.3629e-03, 5.9193e-03, 9.9292e-04, 6.5886e-03, 4.5690e-03, 9.3283e-03,
    #         2.4252e-03, 9.4322e-03, 6.7143e-03, 7.9352e-03, 1.7863e-02, 6.5742e-03,
    #         1.3686e-02, 6.0705e-03, 2.2772e-02, 1.1772e-02, 1.7371e-01, 6.8901e-03,
    #         1.4437e-02, 1.5371e-03, 6.2483e-03, 5.8605e-03, 2.5208e-02, 7.8139e-03,
    #         2.0073e-01, 1.9637e-02])

    return weights


def xyxy2xywh(x):  # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):  # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)
    return y


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)


def build_targets(pred_boxes, pred_conf, pred_cls, target, anchor_wh, nA, nC, nG, batch_report):
    """return tx, ty, tw, th, tconf, tcls, nCorrect, nT:number of targets """
    nB = len(target)  # number of images in batch
    nT = [len(x) for x in target]  # targets per image
    tx = torch.zeros(nB, nA, nG, nG)  # nB:batch size(4)
    ty = torch.zeros(nB, nA, nG, nG)  # nA:number of anchors(3),
    tw = torch.zeros(nB, nA, nG, nG)  # nG:number of grid points(13) = img_dim/stride
    th = torch.zeros(nB, nA, nG, nG)
    tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0)  # 在函数后面加 _  是改变自身的意思
    tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0)  # nC = number of classes
    TP = torch.ByteTensor(nB, max(nT)).fill_(0)
    FP = torch.ByteTensor(nB, max(nT)).fill_(0)
    FN = torch.ByteTensor(nB, max(nT)).fill_(0)

    TC = torch.ShortTensor(nB, max(nT)).fill_(-1)  # target category  目标类别

    for b in range(nB):
        nTb = nT[b]  # number of targets
        if nTb == 0:
            continue
        t = target[b]
        if batch_report:
            FN[b, :nTb] = 1

        # 转换为相对于框的位置
        TC[b, :nTb], gx, gy, gw, gh = t[:, 0].long(), t[:, 1] * nG, t[:, 2] * nG, t[:, 3] * nG, t[:, 4] * nG
        # 获取网格框索引并防止溢出（即13个锚点上的13.01）
        '''
        clamp表示夹紧，夹住的意思，torch.clamp(input,min,max,out=None)-> Tensor
        将input中的元素限制在[min,max]范围内并返回一个Tensor
        '''
        gi = torch.clamp(gx.long(), min=0, max=nG - 1)
        gj = torch.clamp(gy.long(), min=0, max=nG - 1)

        # iou of targets-anchors (using wh only)
        box1 = t[:, 3:5] * nG
        # box2 = anchor_grid_wh[:, gj, gi]
        box2 = anchor_wh.unsqueeze(1).repeat(1, nTb, 1)

        # torch.prod(input): 返回所有元素的乘积
        inter_area = torch.min(box1, box2).prod(2)
        iou_anch = inter_area / (gw * gh + box2.prod(2) - inter_area + 1e-16)

        # Sekect best iou_pred and anchor
        iou_anch_best, a = iou_anch.max(0)  # best anchor [0-2] for each target

        # Select best unique target-anchor combinations
        if nTb > 1:
            iou_order = np.argsort(-iou_anch_best)  # best to worst

            # Unique anchor selection(slower but retains original order)
            u = torch.cat((gi, gj, a), 0).view(3, -1).numpy()
            _, first_unique = np.unique(u[:, iou_order], axis=1, return_index=True)  # 第一个独特的指数

            i = iou_order[first_unique]
            # 最佳anchor必须与目标共享重要的共性（iou）
            i = i[iou_anch_best[i] > 0.10]
            if len(i) == 0:
                continue

            a, gj, gi, t = a[i], gj[i], gi[i], t[i]
            if len(t.shape) == 1:
                t = t.view(1, 5)
        else:
            if iou_anch_best < 0.10:
                continue
            i = 0

        tc, gx, gy, gw, gh = t[:, 0].long(), t[:, 1] * nG, t[:, 2] * nG, t[:, 3] * nG, t[:, 4] * nG

        # Coordinates  坐标
        # b : number of images in batch
        # a : anchor
        tx[b, a, gj, gi] = gx - gi.float()
        ty[b, a, gj, gi] = gy - gj.float()

        # Width and height(yolo method)
        tw[b, a, gj, gi] = torch.log(gw / anchor_wh[a, 0])
        th[b, a, gj, gi] = torch.log(gh / anchor_wh[a, 1])

        # One-hot encoding of label
        tcls[b, a, gj, gi, tc] = 1
        tconf[b, a, gj, gi] = 1

        if batch_report:
            # predicted classes and confidence
            tb = torch.cat((gx - gw / 2, gy - gh / 2, gx + gw / 2, gy + gh / 2)).view(4, -1).t()  # target boxes
            pcls = torch.argmax(pred_cls[b, a, gj, gi], 1).cpu()
            pconf = torch.sigmoid(pred_conf[b, a, gj, gi]).cpu()
            iou_pred = bbox_iou(tb, pred_boxes[b, a, gj, gi].cpu())

            TP[b, i] = (pconf > 0.5) & (iou_pred > 0.5) & (pcls == tc)
            FP[b, i] = (pconf > 0.5) & (TP[b, i] == 0)
            FN[b, i] = pconf <= 0.5
    return tx, ty, tw, th, tconf, tcls, TP, FP, FN, TC


def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, 'r')
    names = fp.read().split('\n')[:-1]
    return names


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Method originally from https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # lists/pytorch to numpy
    tp, conf, pred_cls, target_cls = np.array(tp), np.array(conf), np.array(pred_cls), np.array(target_cls)

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(np.concatenate((pred_cls, target_cls), 0))

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = sum(target_cls == c)  # Number of ground truth objects
        n_p = sum(i)  # Number of predicted objects

        if (n_p == 0) and (n_gt == 0):
            continue
        elif (n_p == 0) or (n_gt == 0):
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = np.cumsum(1 - tp[i])
            tpc = np.cumsum(tp[i])

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(tpc[-1] / (n_gt + 1e-16))

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(tpc[-1] / (tpc[-1] + fpc[-1]))

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    return np.array(ap), unique_classes.astype('int32'), np.array(r), np.array(p)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    output = [None for _ in range(len(prediction))]
    for image_i, pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        # Get score and class with highest confidence

        # cross-class NMS (experimental)
        cross_class_nms = False
        if cross_class_nms:
            # thresh = 0.85
            thresh = nms_thres
            a = pred.clone()
            _, indices = torch.sort(-a[:, 4], 0)  # sort best to worst
            a = a[indices]
            radius = 30  # area to search for cross-class ious
            for i in range(len(a)):
                if i >= len(a) - 1:
                    break

                close = (torch.abs(a[i, 0] - a[i + 1:, 0]) < radius) & (torch.abs(a[i, 1] - a[i + 1:, 1]) < radius)
                close = close.nonzero()

                if len(close) > 0:
                    close = close + i + 1
                    iou = bbox_iou(a[i:i + 1, :4], a[close.squeeze(), :4].reshape(-1, 4), x1y1x2y2=False)
                    bad = close[iou > thresh]

                    if len(bad) > 0:
                        mask = torch.ones(len(a)).type(torch.ByteTensor)
                        mask[bad] = 0
                        a = a[mask]
            pred = a

        x, y, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        a = w * h  # area
        ar = w / (h + 1e-16)  # aspect ratio

        log_w, log_h, log_a, log_ar = torch.log(w), torch.log(h), torch.log(a), torch.log(ar)

        # n = len(w)
        # shape_likelihood = np.zeros((n, 60), dtype=np.float32)
        # x = np.concatenate((log_w.reshape(-1, 1), log_h.reshape(-1, 1)), 1)
        # from scipy.stats import multivariate_normal
        # for c in range(60):
        # shape_likelihood[:, c] = multivariate_normal.pdf(x, mean=mat['class_mu'][c, :2], cov=mat['class_cov'][c, :2, :2])

        class_prob, class_pred = torch.max(F.softmax(pred[:, 5:], 1), 1)

        v = ((pred[:, 4] > conf_thres) & (class_prob > .3))
        v = v.nonzero().squeeze()
        if len(v.shape) == 0:
            v = v.unsqueeze(0)

        pred = pred[v]
        class_prob = class_prob[v]
        class_pred = class_pred[v]

        # If none are remaining => process next image
        nP = pred.shape[0]
        if not nP:
            continue

        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        box_corner = pred.new(nP, 4)
        xy = pred[:, 0:2]
        wh = pred[:, 2:4] / 2
        box_corner[:, 0:2] = xy - wh
        box_corner[:, 2:4] = xy + wh
        pred[:, :4] = box_corner

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_prob, class_pred)
        detections = torch.cat((pred[:, :5], class_prob.float().unsqueeze(1), class_pred.float().unsqueeze(1)), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda(prediction.device)

        nms_style = 'OR'  # 'AND' or 'OR' (classical)
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []

            if nms_style == 'OR':  # Classical NMS
                while detections_class.shape[0]:
                    # Get detection with highest confidence and save as max detection
                    max_detections.append(detections_class[0].unsqueeze(0))
                    # Stop if we're at the last detection
                    if len(detections_class) == 1:
                        break
                    # Get the IOUs for all boxes with lower confidence
                    ious = bbox_iou(max_detections[-1], detections_class[1:])

                    # Remove detections with IoU >= NMS threshold
                    detections_class = detections_class[1:][ious < nms_thres]

            elif nms_style == 'AND':  # 'AND'-style NMS, at least two boxes must share commonality to pass, single boxes erased
                while detections_class.shape[0]:
                    if len(detections_class) == 1:
                        break

                    ious = bbox_iou(detections_class[:1], detections_class[1:])

                    if ious.max() > 0.5:
                        max_detections.append(detections_class[0].unsqueeze(0))

                    # Remove detections with IoU >= NMS threshold
                    detections_class = detections_class[1:][ious < nms_thres]

            if len(max_detections) > 0:
                max_detections = torch.cat(max_detections).data
                # Add max detections to outputs
                output[image_i] = max_detections if output[image_i] is None else torch.cat(
                    (output[image_i], max_detections))

    return output


def write_cfg(cfgfile, cfg):
    with open(cfgfile, 'r') as f:
        lines = f.read().split('\n')  # store the lines in a list
        lines = [x for x in lines if len(x) > 0]  # get read of the empty lines
        lines = [x for x in lines if x[0] != '#']  # get rid of comments
        # lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces\

    block = {}
    blocks = []
    # D:/yolotest/cfg/yolov3.cfg
    # prunedcfg = os.path.join('./'.join(cfgfile.split("/")[0:-1]), "prune_" + cfgfile.split("/")[-1])
    prunedcfg = os.path.join("prune_" + cfgfile.split("/")[-1])
    for line in lines:
        if line[0] == "[":  # This marks the start of a new block
            if len(block) != 0:  # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)  # add it the blocks list
                block = {}  # re-init the block
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    x = 0
    # print(blocks[1])
    for block in blocks:
        if 'batch_normalize' in block:
            block['filters'] = cfg[x]
            x = x + 1
    ##
    with open(prunedcfg, 'w') as f:
        for block in blocks:
            for i in block:
                if i == "type":
                    f.write('\n')
                    f.write("[" + block[i] + "]\n")
                    for j in block:
                        if j != "type":
                            f.write(j + "=" + str(block[j]) + '\n')
    print('save pruned cfg file in %s' % prunedcfg)
    return prunedcfg

def route_problem(model,ind):
    ds = list(model.children())
    dsas = list(ds[0].children())

    # print('-----------',dsas[90])
    sum1 = 0
    # print(dsas[90].named_children())
    for k in range(ind+1):
        # print('k:',k)
        for i in dsas[k].named_children():
            # print('i:',i)
            if "_".join(i[0].split("_")[0:-1]) == 'conv_with_bn':
                sum1 = sum1 + 1
    #print(sum1)
    return sum1-1


def dontprune(model):

    dontprune=[]
    nnlist = model.module_list
    for i in range(len(nnlist)):
        for name in nnlist[i].named_children():
            if name[0].split("_")[0] == 'shortcut':
                if 'conv' in list(nnlist[name[1].froms+i].named_children())[0][0]:
                    dontprune.append(name[1].froms+i)
                else:
                    dontprune.append(name[1].froms + i-1)
                dontprune.append(i-1)
    return dontprune