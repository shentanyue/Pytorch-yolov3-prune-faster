# coding=utf-8
from utils import parse_config
import torch.nn as nn
import torch
from utils import utils
from collections import defaultdict
import numpy as np
from utils import torch_utils
import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'


def create_modules(module_blocks):
    """
    根据module_block中的模块配置构造层模块的模块列表
    """
    net_hyperparams = module_blocks.pop(0)  # 获取网络的配置信息，即【net】
    # print(hyperparams)
    output_filters = [int(net_hyperparams['channels'])]
    module_list = nn.ModuleList()

    for index, module_block in enumerate(module_blocks):
        modules = nn.Sequential()

        # 卷积层
        if module_block['type'] == 'convolutional':
            filters = int(module_block['filters'])
            kernel_size = int(module_block['size'])
            stride = int(module_block['stride'])
            pad = (kernel_size - 1) // 2 if int(module_block['pad']) else 0
            activation = module_block['activation']

            try:
                batch_normalize = int(module_block['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            # 判断是否有BN层，有则增加BN层
            if batch_normalize:
                # 增加卷积层
                conv = nn.Conv2d(in_channels=output_filters[-1],
                                 out_channels=filters,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=pad,
                                 bias=bias)
                modules.add_module('conv_with_bn_%d' % index, conv)
                bn = nn.BatchNorm2d(filters)
                modules.add_module('batch_norm_%d' % index, bn)
            else:
                # 增加卷积层
                conv = nn.Conv2d(in_channels=output_filters[-1],
                                 out_channels=filters,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=pad,
                                 bias=bias)
                modules.add_module('conv_without_bn_%d' % index, conv)

            # 判断激活曾是否为Leaky，是则增加。
            if activation == 'leaky':
                modules.add_module('leaky%d' % index, nn.LeakyReLU(0.1))

        # 上采样层
        # [upsample]
        # stride = 2
        elif module_block['type'] == 'upsample':
            stride = int(module_block['stride'])
            upsample = nn.Upsample(scale_factor=stride,
                                   mode='nearest')
            modules.add_module('upsample_%d' % index, upsample)

        # Route层
        #   [route]                      [route]
        #
        #   layers = -1, 61    或        layers = -4
        elif module_block['type'] == 'route':

            layers = [int(x) for x in module_block['layers'].split(',')]
            filters = sum([output_filters[layers_i] for layers_i in layers])
            modules.add_module('route_%d' % index, Route(layers))
            # module_block["layers"] = module_block["layers"].split(',')
            # # Start of a route
            # start = int(module_block["layers"][0])
            # # end, if there exists one.
            # try:
            #     end = int(module_block["layers"][1])
            # except:
            #     end = 0
            # # Positive anotation
            # if start > 0:
            #     start = start - index
            # if end > 0:
            #     end = end - index
            # route = Route([start, end])
            # modules.add_module("route_{0}".format(index), route)
            # if end < 0:
            #     filters = output_filters[index + start] + output_filters[index + end]
            # else:
            #     filters = output_filters[index + start]

        # Shortcut层
        #   [shortcut]
        #   from=-3
        #   activation = linear
        elif module_block['type'] == 'shortcut':
            froms = int(module_block['from'])
            filters = output_filters[int(module_block['from'])]
            modules.add_module('shortcut_%d' % index, ShortcutLayer(froms))


        # Yolo层
        #     [yolo]
        #     mask = 6, 7, 8
        #     anchors = 10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326
        #     classes = 80
        #     num = 9
        #     jitter = .3
        #     ignore_thresh = .7
        #     truth_thresh = 1
        #     random = 1
        elif module_block['type'] == 'yolo':
            anchor_idxs = [int(x) for x in module_block['mask'].split(',')]
            # 提取anchor
            anchors = [float(x) for x in module_block['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_block['classes'])
            img_height = int(net_hyperparams['height'])
            # 定义检测层
            yolo_layer = YOLOLayer(anchors, num_classes, img_height, anchor_idxs)
            modules.add_module('yolo_%d' % index, yolo_layer)

        module_list.append(modules)
        output_filters.append(filters)

    return net_hyperparams, module_list


class Route(nn.Module):
    def __init__(self, layers):
        super(Route, self).__init__()
        self.layers = layers


class ShortcutLayer(nn.Module):
    def __init__(self, froms):
        super(ShortcutLayer, self).__init__()
        self.froms = froms


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):

    def __init__(self, anchors, nC, img_dim, anchor_idxs):
        super(YOLOLayer, self).__init__()

        anchors = [(a_w, a_h) for a_w, a_h in anchors]  # (pixels)
        nA = len(anchors)

        self.anchors = anchors
        self.nA = nA  # number of anchors (3)
        self.nC = nC  # number of classes (80)
        self.bbox_attrs = 5 + nC
        self.img_dim = img_dim  # from hyperparams in cfg file, NOT from parser

        if anchor_idxs[0] == (nA * 2):  # 6
            stride = 32
        elif anchor_idxs[0] == nA:  # 3
            stride = 16
        else:
            stride = 8

        # Build anchor grids
        nG = int(self.img_dim / stride)  # number grid points
        self.grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).float()
        self.grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).float()
        self.scaled_anchors = torch.FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, nA, 1, 1))
        self.weights = utils.class_weights()

        self.loss_means = torch.ones(6)
        self.tx, self.ty, self.tw, self.th = [], [], [], []

    def forward(self, p, targets=None, batch_report=False, var=None):
        device = torch_utils.select_device(cuda_num=7)
        with torch.cuda.device(device):

            FT = torch.cuda.FloatTensor if p.is_cuda else torch.FloatTensor

            bs = p.shape[0]  # batch size
            nG = p.shape[2]  # number of grid points
            stride = self.img_dim / nG

            if p.is_cuda and not self.grid_x.is_cuda:
                self.grid_x, self.grid_y = self.grid_x.cuda(), self.grid_y.cuda()
                self.anchor_w, self.anchor_h = self.anchor_w.cuda(), self.anchor_h.cuda()
                self.weights, self.loss_means = self.weights.cuda(), self.loss_means.cuda()

            # p.view(12, 255, 13, 13) -- > (12, 3, 13, 13, 80)  # (bs, anchors, grid, grid, classes + xywh)
            p = p.view(bs, self.nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()  # prediction

            # Get outputs
            x = torch.sigmoid(p[..., 0])  # Center x
            y = torch.sigmoid(p[..., 1])  # Center y

            # Width and height (yolo method)
            w = p[..., 2]  # Width
            h = p[..., 3]  # Height
            width = torch.exp(w.data) * self.anchor_w
            height = torch.exp(h.data) * self.anchor_h

            # Width and height (power method)
            # w = torch.sigmoid(p[..., 2])  # Width
            # h = torch.sigmoid(p[..., 3])  # Height
            # width = ((w.data * 2) ** 2) * self.anchor_w
            # height = ((h.data * 2) ** 2) * self.anchor_h

            # Add offset and scale with anchors (in grid space, i.e. 0-13)
            pred_boxes = FT(bs, self.nA, nG, nG, 4)
            pred_conf = p[..., 4]  # Conf
            pred_cls = p[..., 5:]  # Class

            # Training
            if targets is not None:
                MSELoss = nn.MSELoss()
                BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
                CrossEntropyLoss = nn.CrossEntropyLoss()

                if batch_report:
                    gx = self.grid_x[:, :, :nG, :nG]
                    gy = self.grid_y[:, :, :nG, :nG]
                    pred_boxes[..., 0] = x.data + gx - width / 2
                    pred_boxes[..., 1] = y.data + gy - height / 2
                    pred_boxes[..., 2] = x.data + gx + width / 2
                    pred_boxes[..., 3] = y.data + gy + height / 2

                tx, ty, tw, th, mask, tcls, TP, FP, FN, TC = \
                    utils.build_targets(pred_boxes, pred_conf, pred_cls, targets, self.scaled_anchors, self.nA, self.nC, nG,
                                        batch_report)
                tcls = tcls[mask]
                if x.is_cuda:
                    tx, ty, tw, th, mask, tcls = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda(), mask.cuda(), tcls.cuda()

                # Compute losses
                nT = sum([len(x) for x in targets])  # number of targets
                nM = mask.sum().float()  # number of anchors (assigned to targets)
                # print("mask:-----------",nM)
                nB = len(targets)  # batch size
                k = nM / nB
                if nM > 0:
                    lx = k * MSELoss(x[mask], tx[mask])
                    ly = k * MSELoss(y[mask], ty[mask])
                    lw = k * MSELoss(w[mask], tw[mask])
                    lh = k * MSELoss(h[mask], th[mask])

                    # self.tx.extend(tx[mask].data.numpy())
                    # self.ty.extend(ty[mask].data.numpy())
                    # self.tw.extend(tw[mask].data.numpy())
                    # self.th.extend(th[mask].data.numpy())
                    # print([np.mean(self.tx), np.std(self.tx)],[np.mean(self.ty), np.std(self.ty)],[np.mean(self.tw), np.std(self.tw)],[np.mean(self.th), np.std(self.th)])
                    # [0.5040668, 0.2885492] [0.51384246, 0.28328574] [-0.4754091, 0.57951087] [-0.25998235, 0.44858757]
                    # [0.50184494, 0.2858976] [0.51747805, 0.2896323] [0.12962963, 0.6263085] [-0.2722081, 0.61574113]
                    # [0.5032071, 0.28825334] [0.5063132, 0.2808862] [0.21124361, 0.44760725] [0.35445485, 0.6427766]
                    # import matplotlib.pyplot as plt
                    # plt.hist(self.x)

                    # lconf = k * BCEWithLogitsLoss(pred_conf[mask], mask[mask].float())

                    lcls = (k / 4) * CrossEntropyLoss(pred_cls[mask], torch.argmax(tcls, 1))
                    # lcls = (k * 10) * BCEWithLogitsLoss(pred_cls[mask], tcls.float())
                else:
                    lx, ly, lw, lh, lcls, lconf = FT([0]), FT([0]), FT([0]), FT([0]), FT([0]), FT([0])

                # lconf += k * BCEWithLogitsLoss(pred_conf[~mask], mask[~mask].float())
                lconf = (k * 64) * BCEWithLogitsLoss(pred_conf, mask.float())

                # Sum loss components
                balance_losses_flag = False
                if balance_losses_flag:
                    k = 1 / self.loss_means.clone()
                    loss = (lx * k[0] + ly * k[1] + lw * k[2] + lh * k[3] + lconf * k[4] + lcls * k[5]) / k.mean()

                    self.loss_means = self.loss_means * 0.99 + \
                                      FT([lx.data, ly.data, lw.data, lh.data, lconf.data, lcls.data]) * 0.01
                else:
                    loss = lx + ly + lw + lh + lconf + lcls

                # Sum False Positives from unassigned anchors
                FPe = torch.zeros(self.nC)
                if batch_report:
                    i = torch.sigmoid(pred_conf[~mask]) > 0.5
                    if i.sum() > 0:
                        FP_classes = torch.argmax(pred_cls[~mask][i], 1)
                        FPe = torch.bincount(FP_classes, minlength=self.nC).float().cpu()  # extra FPs

                return loss, loss.item(), lx.item(), ly.item(), lw.item(), lh.item(), lconf.item(), lcls.item(), \
                       nT, TP, FP, FPe, FN, TC

            else:
                pred_boxes[..., 0] = x.data + self.grid_x
                pred_boxes[..., 1] = y.data + self.grid_y
                pred_boxes[..., 2] = width
                pred_boxes[..., 3] = height

                # If not in training phase return predictions
                output = torch.cat((pred_boxes.view(bs, -1, 4) * stride,
                                    torch.sigmoid(pred_conf.view(bs, -1, 1)), pred_cls.view(bs, -1, self.nC)), -1)
                return output.data


#
# class YOLOLayer(nn.Module):
#     """Placeholder for 'YOLO'  用于检测"""
#
#     def __init__(self, anchors, num_classes, img_dim, anchor_idxs):
#         super(YOLOLayer, self).__init__()
#
#         anchors = [(a_w, a_h) for a_w, a_h in anchors]
#         num_anchors = len(anchors)
#
#         self.anchors = anchors
#         self.nA = num_anchors  # number of anchors  (3)
#         self.nC = num_classes  # number of classes  (80)
#         self.bbox_attrs = 5 + num_classes  # (85)
#         self.img_dim = img_dim  # from net_hyperparams in cfg file,Not from parser
#
#         if anchor_idxs[0] == (num_anchors * 2):  # 6
#             stride = 32
#         elif anchor_idxs[0] == num_anchors:  # 3
#             stride = 16
#         else:
#             stride = 8
#
#         # 建立anchor坐标
#         nG = int(self.img_dim / stride)  # number of grid points
#         self.grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).float()
#         self.grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).float()
#         self.scaled_anchors = torch.FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in anchors])
#         self.anchors_w = self.scaled_anchors[:, 0:1].view((1, num_anchors, 1, 1))
#         self.anchors_h = self.scaled_anchors[:, 1:2].view((1, num_anchors, 1, 1))
#         self.weights = utils.class_weights()
#
#         self.loss_means = torch.ones(6)
#         self.tx, self.ty, self.tw, self.th = [], [], [], []
#
#     def forward(self, p, targets=None, batch_report=False, var=None):
#         FT = torch.cuda.FloatTensor if p.is_cuda else torch.FloatTensor
#         # p(12,255,13,13)
#         bs = p.shape[0]  # batch size
#         nG = p.shape[2]  # number of grid points
#         stride = self.img_dim / nG
#
#         if p.is_cuda and not self.grid_x.is_cuda:
#             self.grid_x, self.grid_y = self.grid_x.cuda(), self.grid_y.cuda()
#             self.anchors_w, self.anchors_h = self.anchors_w.cuda(), self.anchors_h.cuda()
#             self.weights, self.loss_means = self.weights.cuda(), self.loss_means.cuda()
#
#         # p.view(12, 3, 85, 13, 13) -- > (12, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
#         p = p.view(bs, self.nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous  # prediction
#
#         # Get outputs                 #由于permute倒置了，所以xywh在0,1,2,3
#         x = torch.sigmoid(p[..., 0])  # Center x
#         y = torch.sigmoid(p[..., 1])  # Center y
#         # print("====================")
#         # print(x)
#         # print("--------------------")
#         # print(y)
#         # print("====================")
#
#         # Width and height (yolo method)
#         w = p[..., 2]  # width
#         h = p[..., 3]  # height
#         width = torch.exp(w.data) * self.anchors_w
#         height = torch.exp(h.data) * self.anchors_h
#
#         # Width and height (power method)
#         # w = torch.sigmoid(p[..., 2])  # Width
#         # h = torch.sigmoid(p[..., 3])  # Height
#         # width = ((w.data * 2) ** 2) * self.anchor_w
#         # height = ((h.data * 2) ** 2) * self.anchor_h
#
#         # Add offset and scale with anchors (in grid space,i.e. 0-13)
#         pred_boxes = FT(bs, self.nA, nG, nG, 4)
#         pred_conf = p[..., 4]  # Conf 置信度
#         pred_cls = p[..., 5]  # Class 类别   详见bbox_attrs
#
#         # Training
#         if targets is not None:
#             MSELoss = nn.MSELoss()
#             BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
#             CrossEntropyLoss = nn.CrossEntropyLoss()
#
#             if batch_report:
#                 gx = self.grid_x[:, :, :nG, :nG]
#                 gy = self.grid_y[:, :, :nG, :nG]
#                 pred_boxes[..., 0] = x.data + gx - width / 2
#                 pred_boxes[..., 1] = y.data + gy - height / 2
#                 pred_boxes[..., 2] = x.data + gx + width / 2
#                 pred_boxes[..., 3] = y.data + gy + height / 2
#
#             # TC:target category
#             # t: target
#             tx, ty, tw, th, tconf, tcls, TP, FP, FN, TC = \
#                 utils.build_targets(pred_boxes, pred_conf, pred_cls, targets,
#                                     self.scaled_anchors, self.nA,
#                                     self.nC, nG, batch_report)
#             tcls = tcls[tconf]
#             if x.is_cuda:
#                 tx, ty, tw, th, tconf, tcls = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda(), tconf.cuda(), tcls.cuda()
#             mask = tconf
#             # 计算损失 Compute losses
#             nT = sum([len(x) for x in targets])  # number of targets
#             nM = mask.sum().float()  # number of anchors(assigned to targets)
#             nB = len(targets)  # batch size
#             k = nM / nB
#             if nM > 0:
#                 lx = k * MSELoss(x[mask], tx[mask])
#                 ly = k * MSELoss(y[mask], ty[mask])
#                 lw = k * MSELoss(w[mask], tw[mask])
#                 lh = k * MSELoss(h[mask], th[mask])
#                 lcls = (k / 4) * CrossEntropyLoss(pred_cls[mask], torch.argmax(tcls, 1))
#             else:
#                 lx, ly, lw, lh, lcls, lconf = FT([0]), FT([0]), FT([0]), FT([0]), FT([0]), FT([0])
#
#             lconf = (k * 64) * BCEWithLogitsLoss(pred_conf, mask.float())
#
#             # Sum loss components
#             balance_losses_flag = False
#             if balance_losses_flag:
#                 k = 1 / self.loss_means.clone()
#                 loss = (lx * k[0] + ly * k[1] + lw * k[2] + lh * k[3] + lconf * k[4] + lcls * k[5]) / k.mean()
#                 self.loss_means = self.loss_means * 0.99 + \
#                                   FT([lx.data, ly.data, lw.data, lh.data, lconf.data, lcls.data]) * 0.01
#             else:
#                 loss = lx + ly + lw + lh + lconf + lcls
#
#             # Sum False Positives from unassigned anchors
#             FPe = torch.zeros(self.nC)
#             if batch_report:
#                 i = torch.sigmoid(pred_conf[~mask]) > 0.5
#                 if i.sum() > 0:
#                     FP_classes = torch.argmax(pred_cls[~mask][i], 1)
#                     FPe = torch.bincount(FP_classes, minlength=self.nC).float().cpu()  # 返回每个数的频数  格外的FPs
#
#             return loss, loss.item(), lx.item(), ly.item(), lw.item(), lh.item(), lconf.item(), lcls.item(), \
#                    nT, TP, FP, FPe, FN, TC
#         else:
#             pred_boxes[..., 0] = x.data + self.grid_x
#             pred_boxes[..., 1] = y.data + self.grid_y
#             pred_boxes[..., 2] = width
#             pred_boxes[..., 3] = height
#
#             # 如果没有在训练阶段返回预测
#             output = torch.cat((pred_boxes.view(bs, -1, 4) * stride,
#                                 torch.sigmoid(pred_conf.view(bs, -1, 1)),
#                                 pred_cls.view(bs, -1, self.nC)), -1)
#             return output.data


class Darknet(nn.Module):
    '''Yolov3 object detection model'''

    def __init__(self, cfgfile_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_blocks = parse_config.parse_model_config(cfgfile_path)
        self.module_blocks[0]['height'] = img_size
        self.net_hyperparams, self.module_list = create_modules(self.module_blocks)
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0])
        self.loss_names = ['loss', 'x', 'y', 'w', 'h', 'conf', 'cls', 'nT', 'TP', 'FP', 'FPe', 'FN', 'TC']

    def forward(self, x, targets=None, batch_report=False, var=0):
        is_training = targets is not None
        output = []
        self.losses = defaultdict(float)
        layer_outputs = []

        for i, (module_block, module) in enumerate(zip(self.module_blocks, self.module_list)):
            if module_block['type'] in ['convolutional', 'upsample']:
                x = module(x)  # x=nn.conv(x) or  x=nn.upsample(x)
            elif module_block['type'] == 'route':
                layer_i = [int(x) for x in module_block['layers'].split(',')]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_block['type'] == 'shortcut':
                layer_i = int(module_block['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_block['type'] == 'yolo':

                # Train phase:get loss
                if is_training:
                    x, *losses = module[0](x, targets, batch_report, var)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                else:
                    x = module(x)
                output.append(x)
            layer_outputs.append(x)

        if is_training:
            if batch_report:
                self.losses['TC'] /= 3  # TC:target category  目标类别
                metrics = torch.zeros(3, len(self.losses['FPe']))  # TP,FP,FN    # metrics: 指标

                ui = np.unique(self.losses['TC'])[1:]
                for i in ui:
                    j = self.losses['TC'] == float(i)
                    metrics[0, i] = (self.losses['TP'][j] > 0).sum().float()  # TP
                    metrics[1, i] = (self.losses['FP'][j] > 0).sum().float()  # FP
                    metrics[2, i] = (self.losses['FN'][j] == 3).sum().float()  # FN
                metrics[1] += self.losses['FPe']

                self.losses['TP'] = metrics[0].sum()
                self.losses['FP'] = metrics[1].sum()
                self.losses['FN'] = metrics[2].sum()
                self.losses['metrics'] = metrics
            else:
                self.losses['TP'] = 0
                self.losses['FP'] = 0
                self.losses['FN'] = 0

            self.losses['nT'] /= 3
            self.losses['TC'] = 0
        return sum(output) if is_training else torch.cat(output, 1)

    def load_weights(self, weights_path, cutoff=-1):
        # Parses and loads the weights stored in 'weights_path'
        # @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        if weights_path.endswith('darknet53.conv.74'):
            cutoff = 75

        # Open the weights file
        fp = open(weights_path, 'rb')
        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

        # Needed to write header when saving weights
        self.header_info = header
        self.seen = header[3]
        weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
        fp.close()

        ptr = 0
        for i, (module_block, module) in enumerate(zip(self.module_blocks[:cutoff], self.module_list[:cutoff])):
            if module_block['type'] == 'convolutional':
                conv_layer = module[0]
                try:
                    batch_normalize = int(self.module_blocks[i]["batch_normalize"])
                except:
                    batch_normalize = 0
                if batch_normalize:
                    # Load BN bias,weights,running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()  # torch.numel() 返回一个tensor变量内所有元素个数，可以理解为矩阵内元素的个数
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weight
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w
        print("done!")

    # def save_weights(self, path, cutoff=-1):
    #     fp = open(path, 'wb')
    #     self.header_info[3] = self.seen
    #     self.header_info.tofile(fp)
    #
    #     # Iterate throught layers
    #     for i, (module_block, module) in enumerate(zip(self.module_blocks[:cutoff], self.module_list[:cutoff])):
    #         if module_block['type'] == 'convolutional':
    #             conv_layer = module[0]
    #             # IF BATCHNORM,LOAD BN FIRST
    #             if module_block['batch_normalize']:
    #                 bn_layer = module[1]
    #                 bn_layer.bias.data.cpu().numpy().tofile(fp)
    #                 bn_layer.weight.data.cpu().numpy().tofile(fp)
    #                 bn_layer.running_mean.data.cpu().numpy().tofile(fp)
    #                 bn_layer.running_var.data.cpu().numpy().tofile(fp)
    #             # LOAD CONV BIAS
    #             else:
    #                 conv_layer.bias.data.cpu().numpy().tofile(fp)
    #             # LOAD CONV WEIGHTS
    #             conv_layer.weight.data.cpu().numpy().tofile(fp)
    #
    #     fp.close()

    # def save_weights(self, path, cutoff=-1):
    #     """save layers between 0 and cutoff (cutoff = -1 -> all are saved)"""
    #     fp = open(path, 'wb')
    #     self.header_info[3] = self.seen
    #     self.header_info.tofile(fp)
    #     for i in range(len(self.module_list[:cutoff])):
    #         module_type = self.module_blocks[i + 1]["type"]
    #         # If module_type is convolutional load weights
    #         # Otherwise ignore.
    #         if module_type == "convolutional":
    #             model = self.module_list[i]
    #             try:
    #                 batch_normalize = int(self.module_blocks[i + 1]["batch_normalize"])
    #             except:
    #                 batch_normalize = 0
    #             conv = model[0]
    #             if (batch_normalize):
    #                 bn = model[1]
    #                 bn.bias.data.cpu().numpy().tofile(fp)
    #                 bn.weight.data.cpu().numpy().tofile(fp)
    #                 bn.running_mean.data.cpu().numpy().tofile(fp)
    #                 bn.running_var.data.cpu().numpy().tofile(fp)
    #             else:
    #                 conv.bias.data.cpu().numpy().tofile(fp)
    #             conv.weight.data.cpu().numpy().tofile(fp)
    #     fp.close()

    def save_weights(self, path, cutoff=-1):
        fp = open(path, 'wb')
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_blocks, module) in enumerate(zip(self.module_blocks[:cutoff], self.module_list[:cutoff])):
            if module_blocks['type'] == 'convolutional':
                try:
                    batch_normalize = int(self.module_blocks[i]["batch_normalize"])
                except:
                    batch_normalize = 0
                # print('batchnorm:', batch_normalize)
                conv_layer = module[0]
                # If batch norm, load bn first
                if batch_normalize:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)

                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()


if __name__ == '__main__':
    cfgfile = "cfg\yolov3.cfg"
    blocks = parse_config.parse_model_config(cfgfile)
    a = create_modules(blocks)
    print(a)
