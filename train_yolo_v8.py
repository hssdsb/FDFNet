import os
import sys
import argparse
import shutil
import time
import random
import gc
import json
from distutils.version import LooseVersion
import scipy.misc
import logging

import matplotlib as mpl

mpl.use('Agg')
from matplotlib import pyplot as plt

from PIL import Image
import numpy as np
import torch
torch.cuda.current_device()
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

from dataset.referit_loader_v8 import *

from model_v8.ground_model_v8 import *

# from utils.parsing_metrics import *
from utils.utils import *


def train_loss_fig(losses, name):
    plt.figure(figsize=(10, 6))
    epoch = len(losses)
    plt.plot(range(1, epoch + 1), losses, label='Train Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.savefig(f"./logs/{name}_loss_curve.png")
    plt.show()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    # print(optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'])
    if args.power != 0.:
        lr = lr_poly(args.lr, i_iter, args.nb_epoch, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr / 10


def save_checkpoint(state, is_best, filename='default'):
    if filename == 'default':
        filename = 'model_%s_batch%d' % (args.dataset, args.batch_size)

    checkpoint_name = './saved_models/%s_checkpoint.pth.tar' % (filename)
    best_name = './saved_models/%s_model_best.pth.tar' % (filename)

    torch.save(state, checkpoint_name)
    if is_best:
        shutil.copyfile(checkpoint_name, best_name)


def main():
    parser = argparse.ArgumentParser(description='Dataloader test')
    parser.add_argument('--gpu', default='0', help='gpu id')
    parser.add_argument('--workers', default=4, type=int, help='num workers for data loading')
    parser.add_argument('--nb_epoch', default=100, type=int, help='training epoch')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--power', default=0.9, type=float, help='lr poly power')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--size_average', dest='size_average',
                        default=False, action='store_true', help='size_average')
    parser.add_argument('--size', default=512, type=int, help='image size')
    parser.add_argument('--data_root', type=str, default='./ln_data/',
                        help='path to ReferIt splits data folder')
    parser.add_argument('--split_root', type=str, default='data',
                        help='location of pre-parsed dataset info')
    parser.add_argument('--dataset', default='referit', type=str,
                        help='referit/flickr/unc/unc+/gref_umd')
    parser.add_argument('--time', default=20, type=int,
                        help='maximum time steps (lang length) per batch')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrain', default='', type=str, metavar='PATH',
                        help='pretrain support load state_dict that are not identical, '
                             'while have no loss saved as resume')
    parser.add_argument('--optimizer', default='adamw', help='optimizer: sgd, adam, RMSprop, adamw')
    parser.add_argument('--print_freq', '-p', default=400, type=int,
                        metavar='N', help='print frequency (default: 1e3)')
    parser.add_argument('--savename', default='output', type=str, help='Name head for saved model')
    parser.add_argument('--seed', default=13, type=int, help='random seed')
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='bert model')
    parser.add_argument('--test', dest='test', default=False, action='store_true', help='test')

    global args, anchors_full
    args = parser.parse_args()

    print('----------------------------------------------------------------------')
    print(sys.argv[0])
    print(args)
    print('----------------------------------------------------------------------')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # fix seed
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed + 1)
    torch.manual_seed(args.seed + 2)
    torch.cuda.manual_seed_all(args.seed + 3)

    eps = 1e-10

    # save logs
    if args.savename == 'default':
        args.savename = 'model_%s_batch%d' % (args.dataset, args.batch_size)
    if not os.path.exists('./logs'):
        os.mkdir('logs')
    logging.basicConfig(level=logging.DEBUG, filename="./logs/%s" % args.savename, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")

    input_transform = Compose([
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ReferDataset(data_root=args.data_root,
                                 split_root=args.split_root,
                                 dataset=args.dataset,
                                 split='train',
                                 imsize=args.size,
                                 transform=input_transform,
                                 max_query_len=args.time,
                                 augment=True,
                                 bert_model=args.bert_model)
    val_dataset = ReferDataset(data_root=args.data_root,
                               split_root=args.split_root,
                               dataset=args.dataset,
                               split='val',
                               imsize=args.size,
                               transform=input_transform,
                               max_query_len=args.time,
                               bert_model=args.bert_model)
    # note certain dataset does not have 'test' set:
    # 'unc': {'train', 'val', 'trainval', 'testA', 'testB'}
    test_dataset = ReferDataset(data_root=args.data_root,
                                split_root=args.split_root,
                                dataset=args.dataset,
                                testmode=True,
                                split='test',
                                imsize=args.size,
                                transform=input_transform,
                                max_query_len=args.time,
                                bert_model=args.bert_model)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, drop_last=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            pin_memory=True, drop_last=True, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             pin_memory=True, drop_last=True, num_workers=args.workers)

    # Model
    model = grounding_model(bert_model=args.bert_model)
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    freeze_layer_names = [".dfl"]
    # freeze_layer_names = [".dfl", "textmodel"]
    for k, v in model.module.named_parameters():
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        if any(x in k for x in freeze_layer_names):
            print(f"Freezing layer '{k}'")
            logging.info(f"Freezing layer '{k}'")
            v.requires_grad = False

    if args.pretrain:
        if os.path.isfile(args.pretrain):
            pretrained_dict = torch.load(args.pretrain)['state_dict']
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            assert (len([k for k, v in pretrained_dict.items()]) != 0)
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print("=> loaded pretrain model at {}"
                  .format(args.pretrain))
            logging.info("=> loaded pretrain model at {}"
                         .format(args.pretrain))
        else:
            print(("=> no pretrained file found at '{}'".format(args.pretrain)))
            logging.info("=> no pretrained file found at '{}'".format(args.pretrain))
    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            logging.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # best_loss = checkpoint['best_loss']
            best_ac = checkpoint['best_ac']
            model.load_state_dict(checkpoint['state_dict'])
            # print(("=> loaded checkpoint (epoch {}) Loss{}"
            print(("=> loaded checkpoint (epoch {}) Accuracy{}"
                   # .format(checkpoint['epoch'], best_loss)))
                   .format(checkpoint['epoch'], best_ac)))
            # logging.info("=> loaded checkpoint (epoch {}) Loss{}"
            logging.info("=> loaded checkpoint (epoch {}) Accuracy{}"
                         # .format(checkpoint['epoch'], best_loss))
                         .format(checkpoint['epoch'], best_ac))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))
            logging.info(("=> no checkpoint found at '{}'".format(args.resume)))

    print('Num of parameters:', sum([param.nelement() for param in model.parameters()]))
    logging.info('Num of parameters:%d' % int(sum([param.nelement() for param in model.parameters()])))

    visu_param = model.module.visumodel.parameters()
    rest_param = [param for param in model.parameters() if param not in visu_param]
    """"""
    text_param = model.module.textmodel.parameters()
    """"""
    visu_param = list(model.module.visumodel.parameters())
    sum_visu = sum([param.nelement() for param in visu_param])
    sum_text = sum([param.nelement() for param in model.module.textmodel.parameters()])
    sum_fusion = sum([param.nelement() for param in rest_param]) - sum_text
    print('visu, text, fusion module parameters:', sum_visu, sum_text, sum_fusion)
    logging.info('visu, text, fusion module parameters:%d, %d, %d' % (sum_visu, sum_text, sum_fusion))

    visu_small_param = []
    visu_large_param = []
    for m in model.module.visumodel.model:
        if isinstance(m, nn.Sequential):
            for s in m:
                if s.large_lr:
                    for p in s.parameters():
                        visu_large_param.append(p)
                else:
                    for p in s.parameters():
                        visu_small_param.append(p)
            continue
        if m.large_lr:
            for p in m.parameters():
                visu_large_param.append(p)
        else:
            for p in m.parameters():
                visu_small_param.append(p)
    sum_visu_small = sum([param.nelement() for param in visu_small_param])
    sum_visu_large = sum([param.nelement() for param in visu_large_param])
    print('visu_small_lr, visu_large_lr module parameters:', sum_visu_small, sum_visu_large)
    logging.info('visu_small_lr, visu_large_lr module parameters:%d, %d' % (sum_visu_small, sum_visu_large))

    # optimizer; rmsprop default
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD([
            {'params': rest_param},
            {'params': visu_large_param},
            {'params': visu_small_param, 'lr': args.lr / 10.}
        ], lr=args.lr, momentum=0.99)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=0.0005)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW([
            # {'params': rest_param},
            {'params': visu_large_param},
            {'params': visu_small_param, 'lr': args.lr / 10.},
            {'params': text_param, 'lr': args.lr / 10.}
        ], lr=args.lr, weight_decay=0.0001)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    # training and testing
    best_accu = -float('Inf')
    if args.test:
        _ = test_epoch(test_loader, model, args.size_average)
        exit(0)

    if args.resume:
        EPOCH = args.start_epoch
    else:
        EPOCH = 0

    stop_cet = 0
    train_losses = []
    scaler = torch.amp.GradScaler(device='cuda')
    for epoch in range(EPOCH, args.nb_epoch, 1):
        adjust_learning_rate(optimizer, epoch)
        train_loss_avg = train_epoch(train_loader, model, optimizer, epoch, args.size_average, args.size, scaler)
        train_losses.append(train_loss_avg)
        accu_new = validate_epoch(val_loader, model, args.size_average)
        # remember best accu and save checkpoint
        is_best = accu_new > best_accu
        best_accu = max(accu_new, best_accu)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_ac': accu_new,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename=args.savename)
        if accu_new > 0.9:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_ac': accu_new,
                'optimizer': optimizer.state_dict(),
            }, False, filename=str(epoch))
        print(f"train loss:{train_loss_avg}")
        logging.info('train loss: %f' % train_loss_avg)
        if is_best:
            stop_cet = 0
        else:
            stop_cet = stop_cet + 1
        if stop_cet >= 50:
            break
    print('\nBest Accu: %f\n' % best_accu)
    logging.info('\nBest Accu: %f\n' % best_accu)
    train_loss_fig(train_losses, args.savename)


def train_epoch(train_loader, model, optimizer, epoch, size_average, im_size, scaler):
    batch_time = AverageMeter()
    losses = AverageMeter()
    box_losses = AverageMeter()
    cls_losses = AverageMeter()
    dfl_losses = AverageMeter()
    token_1_num = 0
    token_2_num = 0
    token_3_num = 0
    token_4_num = 0

    model.train()
    end = time.time()

    for batch_idx, (ori_h, ori_w, img_path, imgs, word_id, word_mask, bbox) in enumerate(train_loader):
        for i in range(word_mask.shape[0]):
            word_num = word_mask[i].sum().item() - 2
            if word_num == 1 or word_num == 2:
                token_1_num += 1
            if word_num == 3 or word_num == 4:
                token_2_num += 1
            if word_num >= 5 and word_num <= 7:
                token_3_num += 1
            if word_num >= 8:
                token_4_num += 1
        ori_h = ori_h.cuda()
        ori_w = ori_w.cuda()
        imgs = imgs.cuda()  # (32 3 256 256)
        word_id = word_id.cuda()  # (32 20)
        word_mask = word_mask.cuda()  # (32 20)
        bbox = bbox.cuda()  # (32 4)
        image = Variable(imgs)
        word_id = Variable(word_id)
        word_mask = Variable(word_mask)
        bbox = Variable(bbox)
        bbox = torch.clamp(bbox, min=0, max=1)

        optimizer.zero_grad()
        with autocast():
            loss, loss_item = model(ori_h, ori_w, im_size, img_path, image, word_id, word_mask, bbox)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.update(loss.item()/imgs.size(0), imgs.size(0))
        box_losses.update(loss_item[0], imgs.size(0))
        cls_losses.update(loss_item[1], imgs.size(0))
        dfl_losses.update(loss_item[2], imgs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            print_str = 'Epoch: [{0}][{1}/{2}]\t' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                        'Box_Loss {box_loss.val:.4f} ({box_loss.avg:.4f})\t' \
                        'Cls_Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t' \
                        'Dfl_Loss {dfl_loss.val:.4f} ({dfl_loss.avg:.4f})\t' \
                .format(epoch, batch_idx, len(train_loader), batch_time=batch_time, loss=losses, box_loss=box_losses, cls_loss=cls_losses, dfl_loss=dfl_losses)
            print(print_str)
            logging.info(print_str)
    print(f"num of token1-2/3-4/5-7/8+:{token_1_num},{token_2_num},{token_3_num},{token_4_num}")
    logging.info("%f,%f,%f,%f" % (token_1_num, token_2_num, token_3_num, token_4_num))
    return losses.avg


def validate_epoch(val_loader, model, size_average, mode='val'):
    batch_time = AverageMeter()
    acc = AverageMeter()
    miou = AverageMeter()

    model.eval()
    end = time.time()

    for batch_idx, (h_list, w_list, path_list, imgs, word_id, word_mask, bbox) in enumerate(val_loader):
        imgs = imgs.cuda()
        word_id = word_id.cuda()
        word_mask = word_mask.cuda()
        bbox = bbox.cuda()
        image = Variable(imgs)
        word_id = Variable(word_id)
        word_mask = Variable(word_mask)
        bbox = Variable(bbox)
        bbox = torch.clamp(bbox, min=0, max=args.size - 1)

        with torch.no_grad():
            # Note LSTM does not use word_mask
            results = model(h_list, w_list, args.size, path_list, image, word_id, word_mask, bbox)

        feats = results[0]

        _, max_idx = torch.max(feats[:, 4, :], dim=1)
        B = feats.shape[0]
        batch_max_ids = torch.arange(B, device=feats.device)
        max_pre_bbox = feats[batch_max_ids, :4, max_idx]
        # metrics
        bbox = bbox * torch.tensor(args.size)
        iou = bbox_iou(max_pre_bbox, bbox.data, x1y1x2y2=True)
        accu = np.sum(np.array((iou.data.cpu().numpy() > 0.5), dtype=float)) / args.batch_size

        acc.update(accu, imgs.size(0))
        miou.update(iou.data[0], imgs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            print_str = '[{0}/{1}]\t' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'Accu {acc.val:.4f} ({acc.avg:.4f})\t' \
                        'Mean_iu {miou.val:.4f} ({miou.avg:.4f})\t' \
                .format(batch_idx, len(val_loader), batch_time=batch_time, acc=acc, miou=miou)
            print(print_str)
            logging.info(print_str)
    print(acc.avg, float(miou.avg))
    logging.info("acc/miou/:%f,%f" % (acc.avg, float(miou.avg)))
    return acc.avg


def test_epoch(val_loader, model, size_average, mode='test'):
    batch_time = AverageMeter()
    acc = AverageMeter()
    miou = AverageMeter()

    visual_model = copy.deepcopy(model.module.visumodel)
    assert isinstance(visual_model, torch.nn.Module), "Error: weights must be a torch.nn.Module object!"
    visual_model = visual_model.fuse(verbose=False)
    model.module.visumodel = visual_model

    model.eval()

    token_num_1 = 0
    token_num_2 = 0
    token_num_3 = 0
    token_num_4 = 0
    false_token_num_1 = 0
    false_token_num_2 = 0
    false_token_num_3 = 0
    false_token_num_4 = 0

    for batch_idx, (imgs, word_id, word_mask, bbox, ratio, dw, dh, im_id) in enumerate(val_loader):
        if (word_mask.sum().item()-2) == 1 or (word_mask.sum().item()-2) == 2:
            token_num_1 += 1
        if (word_mask.sum().item()-2) == 3 or (word_mask.sum().item()-2) == 4:
            token_num_2 += 1
        if (word_mask.sum().item()-2) >= 5 and (word_mask.sum().item()-2) <= 7:
            token_num_3 += 1
        if (word_mask.sum().item()-2) >= 8:
            token_num_4 += 1

        imgs = imgs.cuda()
        word_id = word_id.cuda()
        word_mask = word_mask.cuda()
        bbox = bbox.cuda()
        image = Variable(imgs)
        word_id = Variable(word_id)
        word_mask = Variable(word_mask)
        bbox = Variable(bbox)
        bbox = torch.clamp(bbox, min=0, max=args.size - 1)

        with torch.no_grad():
            # Note LSTM does not use word_mask
            start = time.time()
            results = model(dw, dh, args.size, ratio, image, word_id, word_mask, image)
            end = time.time()
            batch_time.update(end - start)

        feats = results[0]

        _, max_idx = torch.max(feats[0, 4, :], dim=0)
        max_pre_bbox = feats[:, :4, max_idx]
        # metrics
        bbox = bbox * torch.tensor(args.size)
        iou = bbox_iou(max_pre_bbox, bbox.data, x1y1x2y2=True)
        accu = np.sum(np.array((iou.data.cpu().numpy() > 0.5), dtype=float)) / 1

        acc.update(accu, imgs.size(0))
        miou.update(iou.data[0], imgs.size(0))

        if batch_idx % args.print_freq == 0:
            print_str = '[{0}/{1}]\t' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'Accu {acc.val:.4f} ({acc.avg:.4f})\t' \
                        'Mean_iu {miou.val:.4f} ({miou.avg:.4f})\t' \
                .format(batch_idx, len(val_loader), batch_time=batch_time, acc=acc, miou=miou)
            print(print_str)
            logging.info(print_str)

        if miou.val <= 0.5:
            if (word_mask.sum().item()-2) == 1 or (word_mask.sum().item()-2) == 2:
                false_token_num_1 += 1
            if (word_mask.sum().item()-2) == 3 or (word_mask.sum().item()-2) == 4:
                false_token_num_2 += 1
            if (word_mask.sum().item()-2) >= 5 and (word_mask.sum().item()-2) <= 7:
                false_token_num_3 += 1
            if (word_mask.sum().item()-2) >= 8:
                false_token_num_4 += 1

    print(acc.avg, miou.avg)
    print(f"num of 1-2/3-4/5-7/8+:{token_num_1},{token_num_2},{token_num_3},{token_num_4}")
    print(f"false num of 1-2/3-4/5-7/8+:{false_token_num_1},{false_token_num_2},{false_token_num_3},{false_token_num_4}")
    print(f"false rate:{false_token_num_1/token_num_1},{false_token_num_2/token_num_2},{false_token_num_3/token_num_3},{false_token_num_4/token_num_4}")
    logging.info("acc/miou/:%f,%f" % (acc.avg, float(miou.avg)))
    logging.info("num of 1-2/3-4/5-7/8+:%f,%f,%f,%f" % (token_num_1, token_num_2, token_num_3, token_num_4))
    logging.info("false num of 1-2/3-4/5-7/8+:%f,%f,%f,%f" % (false_token_num_1, false_token_num_2, false_token_num_3, false_token_num_4))
    logging.info("success rate:%f,%f,%f,%f" % (1-(false_token_num_1/token_num_1), 1-(false_token_num_2/token_num_2), 1-(false_token_num_3/token_num_3), 1-(false_token_num_4/token_num_4)))
    return acc.avg


if __name__ == "__main__":
    main()
