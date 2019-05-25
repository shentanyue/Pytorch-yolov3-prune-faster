import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from model import *
from utils.datasets import *
from utils.utils import *
from utils import torch_utils
import torch
import os
from collections import OrderedDict
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
                                 bias=True)
                modules.add_module('conv_with_bn_%d' % index, conv)
                # bn = nn.BatchNorm2d(filters)
                # modules.add_module('batch_norm_%d' % index, bn)
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


"""  Functions  """


def merge(params, name, layer):
    # global variables
    global weights, bias
    global bn_param

    if layer == 'Convolution':
        # save weights and bias when meet conv layer
        print(1)
        if 'weight' in name:
            weights = params.data
            bias = torch.zeros(weights.size()[0])
        elif 'bias' in name:
            bias = params.data
        bn_param = {}


    elif layer == 'BatchNorm':
        # save bn params
        bn_param[name.split('.')[-1]] = params.data
        # print(bn_param)
        # running_var is the last bn param in pytorch
        if 'running_var' in name:
            # let us merge bn ~
            print(2)
            tmp = bn_param['weight'] / torch.sqrt(bn_param['running_var'] + 1e-5)
            weights = tmp.view(tmp.size()[0], 1, 1, 1) * weights
            bias = tmp * (bias - bn_param['running_mean']) + bn_param['bias']

            return weights, bias

    return None, None


class Darknet_new(nn.Module):
    '''Yolov3 object detection model'''

    def __init__(self, cfgfile_path, img_size=416):
        super(Darknet_new, self).__init__()
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
                batch_normalize = 0
                # if batch_normalize:
                #     # Load BN bias,weights,running mean and running variance
                #     bn_layer = module[1]
                #     num_b = bn_layer.bias.numel()  # Number of biases
                #     # Bias
                #     bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                #     bn_layer.bias.data.copy_(bn_b)
                #     ptr += num_b
                #     # Weight
                #     bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                #     bn_layer.weight.data.copy_(bn_w)
                #     ptr += num_b
                #     # Running Mean
                #     bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                #     bn_layer.running_mean.data.copy_(bn_rm)
                #     ptr += num_b
                #     # Running Var
                #     bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                #     bn_layer.running_var.data.copy_(bn_rv)
                #     ptr += num_b
                # else:
                #     # Load conv. bias
                #     num_b = conv_layer.bias.numel()  # torch.numel() 返回一个tensor变量内所有元素个数，可以理解为矩阵内元素的个数
                #     conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                #     conv_layer.bias.data.copy_(conv_b)
                #     ptr += num_b
                # Load conv. weight
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w
        print("done!")

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


"""  Main functions  """
# import pytorch model

SAVE = True
device = torch.device('cuda')
net_config_path = 'mul_sparsity/yolov3_5.cfg'
img_size = 416
# new_weights_path = '/data_1/shenty/yolo_model/sparsity_weights/'
# best_weights_file = os.path.join(new_weights_path, 'yolov3_sparsity_20.weights')
new_weights_path = 'mul_sparsity/yolov3_5.weights'
best_weights_file = os.path.join(new_weights_path, 'best.pt')

# Initialize model
pytorch_net = Darknet(net_config_path, img_size)

# load weights
print('Finding trained model weights...')
checkpoint = torch.load(best_weights_file, map_location='cpu')
# pytorch_net.load_state_dict(checkpoint['model'])
# pytorch_net.to(device).train()
pytorch_net.load_weights(best_weights_file)

# go through pytorch net
print('Going through pytorch net weights...')
new_weights = OrderedDict()
inner_product_flag = False
for name, params in checkpoint['model'].items():
    print(name, len(params.size()))
    if 'conv_without_bn' in name:
        new_weights[name] = params
        continue
    elif len(params.size()) == 4:
        _, _ = merge(params, name, 'Convolution')
        prev_layer = name
    elif len(params.size()) == 1:
        w, b = merge(params, name, 'BatchNorm')
        if w is not None:
            new_weights[prev_layer] = w
            new_weights[prev_layer.replace('weight', 'bias')] = b
    else:
        print('None')
    #     # inner product layer
    #     # if meet inner product layer,
    #     # the next bias weight can be misclassified as 'BatchNorm' layer as len(params.size()) == 1
    #     new_weights[name] = params
    # inner_product_flag = True

# align names in new_weights with pytorch model
# after move BatchNorm layer in pytorch model,
# the layer names between old model and new model will mis-align
pytorch_net_key_list = list(pytorch_net.state_dict().keys())
new_weights_key_list = list(new_weights.keys())
print(len(pytorch_net_key_list))
print(len(new_weights_key_list))
print('Aligning weight names...')
module_blocks = parse_config.parse_model_config(net_config_path)
module_blocks[0]['height'] = img_size
net_hyperparams, module_list = create_modules(module_blocks)

pytorch_net_key_list = list(module_list.state_dict().keys())
new_weights_key_list = list(new_weights.keys())
print(len(pytorch_net_key_list))
print(len(new_weights_key_list))
# print(new_weights_key_list)

# assert len(pytorch_net_key_list) == len(new_weights_key_list)
for index in range(len(pytorch_net_key_list)):
    print(pytorch_net_key_list[index])
    # print(new_weights_key_list[index])
    new_weights[pytorch_net_key_list[index]] = new_weights.pop(new_weights_key_list[index])
    # print(new_weights)

# save new weights
weighs_file = './4_sparsity_merged.weights'
if SAVE:
    torch.save(new_weights, weighs_file)

import argparse
import time

from model import *
from utils.datasets import *
from utils.utils import *

data_config_path = 'cfg/coco.data'
data_config = parse_config.parse_data_config(data_config_path)
images_path = './data/samples'
batch_size = 1
conf_thres = 0.25
nms_thres = 0.45

new_model = Darknet_new(net_config_path, img_size)
new_model.load_weights(weighs_file)
new_model.to(device).eval()

# Set Dataloader
classes = load_classes(data_config['names'])  # Extracts class labels from file
dataloader = load_images(images_path, batch_size=batch_size, img_size=img_size)

imgs = []  # Stores image paths
img_detections = []  # Stores detections for each image index

total_time = 0
for i in range(3):
    for i, (img_paths, img) in enumerate(dataloader):
        print('%g/%g' % (i + 1, len(dataloader)), end=' ')
        prev_time = time.time()
        # Get detections
        with torch.no_grad():
            pred = new_model(torch.from_numpy(img).unsqueeze(0).to(device))
            pred = pred[pred[:, :, 4] > conf_thres]

            if len(pred) > 0:
                detections = non_max_suppression(pred.unsqueeze(0), conf_thres, nms_thres)
                img_detections.extend(detections)
                imgs.extend(img_paths)

        print('Batch %d... Done. (%.3fs)' % (i, time.time() - prev_time))
        total_time = (time.time() - prev_time) + total_time

total_time = total_time / (3 * 14)
print('total_time:', total_time)
