from utils import parse_config
import torch.nn as nn
import torch
from utils import utils


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
                modules.add_module('leaky%d' % index, nn.LeakyReLU(0, 1, inplace=True))

        # 上采样层
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
            modules.add_module('route_%d' % index, EmptyLayer())

        # Shortcut层
        #   [shortcut]
        #   from=-3
        #   activation = linear
        elif module_block['type'] == 'shortcut':
            filters = output_filters[int(module_block['from'])]
            modules.add_module('shortcut_%d' % index, EmptyLayer())

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


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """Placeholder for 'YOLO'  用于检测"""

    def __init__(self, anchors, num_classes, img_dim, anchor_idxs):
        super(YOLOLayer, self).__init__()

        anchors = [(a_w, a_h) for a_w, a_h in anchors]
        num_anchors = len(anchors)

        self.anchors = anchors
        self.nA = num_anchors  # number of anchors  (3)
        self.nC = num_classes  # number of classes  (80)
        self.bbox_attrs = 5 + num_classes  # (85)
        self.img_dim = img_dim  # from net_hyperparams in cfg file,Not from parser

        if anchor_idxs[0] == (num_anchors * 2):  # 6
            stride = 32
        elif anchor_idxs[0] == num_anchors:  # 3
            stride = 16
        else:
            stride = 8

        # 建立anchor坐标
        nG = int(self.img_dim / stride)  # number of grid points
        self.grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).float()
        self.grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).float()
        self.scaled_anchors = torch.FloatTensor([a_w / stride, a_h / stride] for a_w, a_h in anchors)
        self.anchors_w = self.scaled_anchors[:, 0:1].view((1, num_anchors, 1, 1))
        self.anchors_h = self.scaled_anchors[:, 1:2].view((1, num_anchors, 1, 1))
        self.weights = utils.class_weights()

        self.loss_means = torch.ones(6)
        self.tx, self.ty, self.tw, self.th = [], [], [], []

    def forward(self, p, targets=None, batch_report=False, var=None):
        FT = torch.cuda.FloatTensor if p.is_cuda else torch.FloatTensor
        # p(12,255,13,13)
        bs = p.shape[0]  # batch size
        nG = p.shape[2]  # number of grid points
        stride = self.img_dim / nG

        if p.is_cuda and not self.grid_x.is_cuda:
            self.grid_x, self.grid_y = self.grid_x.cuda(), self.grid_y.cuda()
            self.anchors_w, self.anchors_h = self.anchors_w.cuda(), self.anchors_h.cuda()
            self.weights, self.loss_means = self.weights.cuda(), self.loss_means.cuda()

        # p.view(12, 3, 85, 13, 13) -- > (12, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous  # prediction

        # Get outputs                 #由于permute倒置了，所以xywh在0,1,2,3
        x = torch.sigmoid(p[..., 0])  # Center x
        y = torch.sigmoid(p[..., 1])  # Center y
        print("====================")
        print(x)
        print("--------------------")
        print(y)
        print("====================")

        # Width and height (yolo method)
        w = p[..., 2]  # width
        h = p[..., 3]  # height
        width = torch.exp(w.data) * self.anchors_w
        height = torch.exp(h.data) * self.anchors_h

        # Width and height (power method)
        # w = torch.sigmoid(p[..., 2])  # Width
        # h = torch.sigmoid(p[..., 3])  # Height
        # width = ((w.data * 2) ** 2) * self.anchor_w
        # height = ((h.data * 2) ** 2) * self.anchor_h

        # Add offset and scale with anchors (in grid space,i.e. 0-13)
        pred_boxes = FT(bs, self.nA, nG, nG, 4)
        pred_conf = p[..., 4]  # Conf 置信度
        pred_cls = p[..., 5]  # Class 类别   详见bbox_attrs

        # Training
        if targets is not None:
            MSELoss = nn.MSELoss()
            BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
            CrossEntropyLoss = nn.CrossEntropyLoss()

            if batch_report:
                gx = self.grid_x[:, :, :nG, :nG]
                gy = self.grad_y[:, :, :nG, :nG]
                pred_boxes[..., 0] = x.data + gx - width / 2
                pred_boxes[..., 1] = y.data + gy - height / 2
                pred_boxes[..., 2] = x.data + gx + width / 2
                pred_boxes[..., 3] = y.data + gx + height / 2

            # TC:target category
            # t: target
            tx, ty, tw, th, tconf, tcls, TP, FP, FN, TC = \
                utils.build_targets(pred_boxes, pred_conf, pred_cls, targets,
                                    self.scaled_anchors, self.nA,
                                    self.nC, nG, batch_report)
            tcls = tcls[tconf]
            if x.is_cuda:
                tx, ty, tw, th, tconf, tcls = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda(), tconf.cuda(), tcls.cuda()
            mask = tconf
            # 计算损失 Compute losses
            nT = sum([len(x) for x in targets])  # number of targets
            nM = mask.sum().float()  # number of anchors(assigned to targets)
            nB = len(targets)  # batch size
            k = nM / nB
            if nM > 0:
                lx = k * MSELoss(x[mask], tx[mask])
                ly = k * MSELoss(y[mask], ty[mask])
                lw = k * MSELoss(w[mask], tw[mask])
                lh = k * MSELoss(h[mask], th[mask])
                lcls = (k / 4) * CrossEntropyLoss(pred_cls[mask], torch.argmax(tcls, 1))
            else:
                lx, ly, lw, lh, lcls, lconf = FT([0]), FT([0]), FT([0]), FT([0]), FT([0]), FT([0])

            lconf = (k * 64) * BCEWithLogitsLoss(pred_conf, mask.float())

            # Sum loss components
            balance_losses_flag = False
            if balance_losses_flag:
                k = 1 / self.loss_means.clone()
                loss = (lx * k[0] + ly * k[1], lw * k[2], lh * k[3], lconf * k[4], lcls * k[5]) / k.mean()
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
                    FPe = torch.bincount(FP_classes, minlength=self.nC).float().cpu()  # 返回每个数的频数  格外的FPs

            return loss, loss.item(), lx.item(), ly.item(), lw.item(), lh.item(), lconf.item(), lcls.item(), \
                   nT, TP, FP, FPe, FN, TC
        else:
            pred_boxes[...,0]=x.data+self.grid_x
            pred_boxes[...,1]=y.data+self.grid_y
            pred_boxes[...,2]=width
            pred_boxes[...,3]=height

            #如果没有在训练阶段返回预测
            output =torch.cat((pred_boxes.view(bs,-1,4)*stride,
                               torch.sigmoid(pred_conf.view(bs,-11)),
                               pred_cls.view(bs,-1,self.nC)),-1)
            return output.data


class Darknet(nn.Module):
    '''Yolov3 object detection model'''
    def __init__(self):
        pass







if __name__ == '__main__':
    cfgfile = "cfg\yolov3.cfg"
    blocks = parse_config.parse_model_config(cfgfile)
    create_modules(blocks)
