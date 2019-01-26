import numpy as np
import torch


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
