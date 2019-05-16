#coding=utf-8
import argparse
import sys
import time

from model3 import *
from utils.datasets import *
from utils.utils import *
from utils import torch_utils
import torch
import os
# os.environ['CUDA_VISIBLE_DEVICES']='8'
# Import test.py to get mAP after each epoch
import test
os.environ['OMP_NUM_THREADS'] = '6'
DARKNET_WEIGHTS_FILENAME = 'darknet53.conv.74'
DARKNET_WEIGHTS_URL = 'https://pjreddie.com/media/files/{}'.format(DARKNET_WEIGHTS_FILENAME)


def updateBN(model, s):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            # print(1)
            m.weight.grad.data.add_(s * torch.sign(m.weight.data))  # L1 Sparsity


def train(
        net_config_path,
        data_config_path,
        img_size=416,
        resume=False,
        epochs=100,
        batch_size=16,
        weights_path='weights',
        report=False,
        multi_scale=False,
        freeze_backbone=True,
        var=0,
        s=0.0001,
):
    device = torch_utils.select_device(cuda_num=7)
    print("Using device: \"{}\"".format(device))

    if not multi_scale:
        torch.backends.cudnn.benchmark = True

    # os.makedirs(weights_path, exist_ok=True)
    latest_weights_file = os.path.join(weights_path, 'latest.pt')
    best_weights_file = os.path.join(weights_path, 'best.pt')

    # Configure run
    data_config = parse_config.parse_data_config(data_config_path)
    num_classes = int(data_config['classes'])
    train_path = data_config['train']

    # Initialize model
    model = Darknet(net_config_path, img_size)

    # Get dataloader
    if multi_scale:  # pass maximum multi_scale size
        img_size = 608

    dataloader = load_images_and_labels(train_path, batch_size=batch_size, img_size=img_size,
                                        multi_scale=multi_scale, augment=True)

    lr0 = 0.001
    if resume:
        checkpoint = torch.load(best_weights_file, map_location='cpu')

        model.load_state_dict(checkpoint['model'])
            # print('Using ', torch.cuda.device_count(), ' GPUs')
            # model = nn.DataParallel(model)
        model.to(device).train()

        # # Transfer learning (train only YOLO layers)
        # for i, (name, p) in enumerate(model.named_parameters()):
        #     if p.shape[0] != 650:  # not YOLO layer
        #         p.requires_grad = False

        # Set optimizer
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr0, momentum=.9)

        start_epoch = checkpoint['epoch'] + 1
        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_loss = checkpoint['best_loss']

        del checkpoint  # current, saved

    else:
        start_epoch = 0
        best_loss = float('inf')

        # Initialize model with darknet53 weights(optional)
        # def_weight_file = os.path.join(weights_path, DARKNET_WEIGHTS_FILENAME)
        # if not os.path.isfile(def_weight_file):
        #     os.system('wget {} -P {}'.format(
        #         DARKNET_WEIGHTS_URL,
        #         weights_path))
        # assert os.path.isfile(def_weight_file)


        def_weight_file=weights_path

        model.load_weights(def_weight_file)

        model.to(device).train()

        # Set optimizer
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr0, momentum=.9)

    model_info(model)
    t0, t1 = time.time(), time.time()
    mean_recall, mean_precision = 0, 0
    for epoch in range(epochs):
        epoch += start_epoch
        print(('%8s%12s' + '%10s' * 14) % ('Epoch', 'Batch', 'x', 'y', 'w', 'h', 'conf', 'cls', 'total', 'P', 'R',
                                           'nTargets', 'TP', 'FP', 'FN', 'time'))

        if epoch > 10:
            lr = lr0 / 10
        else:
            lr = lr0
        for g in optimizer.param_groups:
            g['lr'] = lr

        # Freeze darknet53.conv.74 layers for first epoch
        if freeze_backbone:
            if epoch == 0:
                for i, (name, p) in enumerate(model.named_parameters()):
                    if int(name.split('.')[1]) < 75:  # if layer < 75
                        p.requires_grad = False
            elif epoch == 1:
                for i, (name, p) in enumerate(model.named_parameters()):
                    if int(name.split('.')[1]) < 75:  # if layer < 75
                        p.requires_grad = True

        ui = -1
        rloss = defaultdict(float)  # running loss
        metrics = torch.zeros(3, num_classes)
        optimizer.zero_grad()
        for i, (imgs, targets) in enumerate(dataloader):
            if sum([len(x) for x in targets]) < 1:  # if no targets continue
                continue

            # SGD burn-in
            if (epoch == 0) & (i <= 1000):
                lr = lr0 * (i / 1000) ** 4
                for g in optimizer.param_groups:
                    g['lr'] = lr

            # compute loss,compute gradient,update parameters
            loss = model(imgs.to(device), targets, batch_report=report, var=var)
            loss.backward()

            # Sparsity L1 loss
            updateBN(model, 0.0001)

            # 累积批次
            accumulated_batches = 4  # accumulate gradient for 4 batches before optimizing
            if ((i + 1) % accumulated_batches == 0) or (i == len(dataloader) - 1):
                optimizer.step()
                optimizer.zero_grad()

            # Running epoch-means of tracked metrics
            ui += 1
            for key, val in model.losses.items():
                rloss[key] = (rloss[key] * ui + val) / (ui + 1)

            if report:
                TP, FP, FN = metrics
                metrics += model.losses['metrics']

                # Precision
                precision = TP / (TP + FP)
                k = (TP + FP) > 0
                # print('k1:{}'.format(k))
                if k.sum() > 0:
                    mean_precision = precision[k].mean()

                # Recall
                recall = TP / (TP + FN)
                k = (TP + FN) > 0
                # print('k2:{}'.format(k))
                if k.sum() > 0:
                    mean_recall = recall[k].mean()

            s = ('%8s%12s' + '%10.3g' * 14) % (
                '%g/%g' % (epoch, epochs - 1), '%g/%g' % (i, len(dataloader) - 1), rloss['x'],
                rloss['y'], rloss['w'], rloss['h'], rloss['conf'], rloss['cls'],
                rloss['loss'], mean_precision, mean_recall, model.losses['nT'], model.losses['TP'],
                model.losses['FP'], model.losses['FN'], time.time() - t1)
            t1 = time.time()
            print(s)

        # Update best loss
        loss_per_target = rloss['loss'] / rloss['nT']
        if loss_per_target < best_loss:
            best_loss = loss_per_target

        # Save latest checkpoint
        # checkpoint = {'epoch': epoch,
        #               'best_loss': best_loss,
        #               'model': model.state_dict(),
        #               'optimizer': optimizer.state_dict()}
        # torch.save(checkpoint, latest_weights_file)
        model.save_weights("%s/%s/yolov3_sparsity_%d.weights" % ('prune_refine','percent_30',epoch))
        print("save weights in %s/%s/yolov3_sparsity_%d.weights" % ('prune_refine','percent_30',epoch))
        # Save best checkpoint

        # # Save best checkpoint
        # if best_loss == loss_per_target:
        #     os.system('cp {} {}'.format(
        #         latest_weights_file,
        #         best_weights_file,
        #     ))

        # # Save backup weights every 5 epochs
        # if (epoch > 0) & (epoch % 5 == 0):
        #     backup_file_name = 'backup{}.pt'.format(epoch)
        #     backup_file_path = os.path.join(weights_path, backup_file_name)
        #     os.system('cp {} {}'.format(
        #         latest_weights_file,
        #         backup_file_path,
        #     ))
            # model.save_weights("%s/yolov3_sparsity_%d.weights" % ('sparsity_weights_5', epoch)) ]
            # print("save weights in %s/yolov3_sparsity_%d.weights" % ('sparsity_weights_5', epoch))
        # Calculate mAP
        mAP, R, P = test.test(
            net_config_path,
            data_config_path,
            latest_weights_file,
            batch_size=batch_size,
            img_size=img_size,
        )

        # Write epoch results
        with open('results.txt', 'a') as file:
            file.write(s + '%11.3g' * 3 % (mAP, P, R) + '\n')

    # Save final model
    dt = time.time() - t0
    print('Finished %g epochs in %.2fs (%.2fs/epoch)' % (epoch, dt, dt / (epoch + 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--data-config', type=str, default='cfg/coco.data', help='path to data config file')
    parser.add_argument('--multi-scale', action='store_true', help='random image sizes per batch 320 - 608')
    parser.add_argument('--img-size', type=int, default=32 * 13, help='pixels')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--report', action='store_true', help='report TP, FP, FN, P and R per batch (slower)')
    parser.add_argument('--freeze', action='store_true', help='freeze darknet53.conv.74 layers for first epoche')
    parser.add_argument('--var', type=float, default=0, help='optional test variable')
    parser.add_argument('--s', type=float, default=0.0001, help='sparity')

    parser.add_argument('--cfg', type=str, default='prune_cfg/prune_0.3_yolov3.cfg', help='cfg file path')
    parser.add_argument('--weights-path', type=str, default='prune_weights/prune_0.3_yolov3_sparsity_95.weights',
                        help='path to store weights')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    torch_utils.init_seeds()

    torch.cuda.empty_cache()
    train(
        opt.cfg,
        opt.data_config,
        img_size=opt.img_size,
        resume=opt.resume,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        weights_path=opt.weights_path,
        report=opt.report,
        multi_scale=opt.multi_scale,
        freeze_backbone=opt.freeze,
        var=opt.var,
        s=opt.s,
    )
