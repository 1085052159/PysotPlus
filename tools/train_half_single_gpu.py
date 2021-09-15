# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import time
import math
import json
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from apex import amp
from torchvision.models.googlenet import Inception

from pysot.models.init_weight import init_weights
from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.log_helper import init_log, print_speed, add_file_handler
from pysot.utils.misc import describe
from pysot.utils.model_load import load_pretrain, restore_from
from pysot.utils.average_meter import AverageMeter
from pysot.models.model_builder import ModelBuilder
from pysot.datasets.dataset import TrkDataset
from pysot.core.config import cfg

logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--cfg', type=str, default='config.yaml',
                    help='configuration of tracking')
parser.add_argument('--seed', type=int, default=123456,
                    help='random seed')
parser.add_argument('--local_rank', type=int, default=0,
                    help='compulsory for pytorch launcer')
args = parser.parse_args()


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_data_loader():
    logger.info("build train dataset")
    # train_dataset
    train_dataset = TrkDataset()
    logger.info("build dataset done")

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=cfg.TRAIN.NUM_WORKERS,
                              pin_memory=True)
    return train_loader


def build_opt_lr(model, current_epoch=0):
    for param in model.backbone.parameters():
        param.requires_grad = False
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    if current_epoch >= cfg.BACKBONE.TRAIN_EPOCH:
        for layer in cfg.BACKBONE.TRAIN_LAYERS:
            for param in getattr(model.backbone, layer).parameters():
                param.requires_grad = True
            for m in getattr(model.backbone, layer).modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()

    trainable_params = []
    trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                           model.backbone.parameters()),
                          'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]

    if cfg.ADJUST.ADJUST:
        trainable_params += [{'params': model.neck.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]

    trainable_params += [{'params': model.rpn_head.parameters(),
                          'lr': cfg.TRAIN.BASE_LR}]

    if cfg.MASK.MASK:
        trainable_params += [{'params': model.mask_head.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]

    if cfg.REFINE.REFINE:
        trainable_params += [{'params': model.refine_head.parameters(),
                              'lr': cfg.TRAIN.LR.BASE_LR}]

    optimizer = torch.optim.SGD(trainable_params,
                                momentum=cfg.TRAIN.MOMENTUM,
                                weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    lr_scheduler = build_lr_scheduler(optimizer, epochs=cfg.TRAIN.EPOCH)
    lr_scheduler.step(cfg.TRAIN.START_EPOCH)
    return optimizer, lr_scheduler


def log_grads(model, tb_writer, tb_index):
    def weights_grads(model):
        grad = {}
        weights = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad[name] = param.grad
                weights[name] = param.data
        return grad, weights

    grad, weights = weights_grads(model)
    feature_norm, rpn_norm = 0, 0
    for k, g in grad.items():
        _norm = g.data.norm(2)
        weight = weights[k]
        w_norm = weight.norm(2)
        if 'feature' in k:
            feature_norm += _norm ** 2
        else:
            rpn_norm += _norm ** 2

        tb_writer.add_scalar('grad_all/' + k.replace('.', '/'),
                             _norm, tb_index)
        tb_writer.add_scalar('weight_all/' + k.replace('.', '/'),
                             w_norm, tb_index)
        tb_writer.add_scalar('w-g/' + k.replace('.', '/'),
                             w_norm / (1e-20 + _norm), tb_index)
    tot_norm = feature_norm + rpn_norm
    tot_norm = tot_norm ** 0.5
    feature_norm = feature_norm ** 0.5
    rpn_norm = rpn_norm ** 0.5

    tb_writer.add_scalar('grad/tot', tot_norm, tb_index)
    tb_writer.add_scalar('grad/feature', feature_norm, tb_index)
    tb_writer.add_scalar('grad/rpn', rpn_norm, tb_index)


def train(train_loader, model, optimizer, lr_scheduler, tb_writer):
    cur_lr = lr_scheduler.get_cur_lr()

    average_meter = AverageMeter()

    def is_valid_number(x):
        return not (math.isnan(x) or math.isinf(x) or x > 1e4)

    num_per_epoch = len(train_loader.dataset) // \
                    cfg.TRAIN.EPOCH // cfg.TRAIN.BATCH_SIZE
    start_epoch = cfg.TRAIN.START_EPOCH
    epoch = start_epoch
    snapshot_dir = os.path.join(cfg.RESULTS_BASE_PATH, cfg.TRAIN.SNAPSHOT_DIR)

    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)

    end = time.time()
    for idx, data in enumerate(train_loader):
        if epoch != idx // num_per_epoch + start_epoch:
            epoch = idx // num_per_epoch + start_epoch
            torch.save(
                {'epoch': epoch,
                 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict()},
                snapshot_dir + '/checkpoint_e%d.pth' % (epoch),
                _use_new_zipfile_serialization=False)

            if epoch == cfg.TRAIN.EPOCH:
                torch.save(
                    {'epoch': epoch,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()},
                    snapshot_dir + '/checkpoint_e%d.pth' % (epoch),
                    _use_new_zipfile_serialization=False)
                return

            if cfg.BACKBONE.TRAIN_EPOCH == epoch:
                logger.info('start training backbone. release occupied memory')
                optimizer, lr_scheduler = build_opt_lr(model, epoch)
                model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

            lr_scheduler.step(epoch)
            cur_lr = lr_scheduler.get_cur_lr()
            logger.info('epoch: {}'.format(epoch + 1))

        tb_idx = idx
        if idx % num_per_epoch == 0 and idx != 0:
            for idx, pg in enumerate(optimizer.param_groups):
                logger.info('epoch {} lr {}'.format(epoch + 1, pg['lr']))
                tb_writer.add_scalar('lr/group{}'.format(idx + 1),
                                     pg['lr'], tb_idx)

        data_time = time.time() - end
        tb_writer.add_scalar('time/data', data_time, tb_idx)

        outputs = model(data)
        loss = outputs['total_loss']

        if is_valid_number(loss.data.item()):
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            if cfg.TRAIN.LOG_GRADS:
                log_grads(model, tb_writer, tb_idx)

            # clip gradient
            clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
            optimizer.step()

        batch_time = time.time() - end
        batch_info = {}
        batch_info['batch_time'] = batch_time
        batch_info['data_time'] = data_time
        for k, v in sorted(outputs.items()):
            batch_info[k] = v.data.item()

        average_meter.update(**batch_info)

        for k, v in batch_info.items():
            tb_writer.add_scalar(k, v, tb_idx)

        if (idx + 1) % cfg.TRAIN.PRINT_FREQ == 0:
            info = "Epoch: [{}][{}/{}] lr: {:.6f}\n".format(
                epoch + 1, (idx + 1) % num_per_epoch,
                num_per_epoch, cur_lr)
            for cc, (k, v) in enumerate(batch_info.items()):
                if cc % 2 == 0:
                    info += ("\t{:s}\t").format(
                        getattr(average_meter, k))
                else:
                    info += ("{:s}\n").format(
                        getattr(average_meter, k))
            logger.info(info)
            print_speed(idx + 1 + start_epoch * num_per_epoch,
                        average_meter.batch_time.avg,
                        cfg.TRAIN.EPOCH * num_per_epoch)
        end = time.time()


def main(config):
    # load cfg
    cfg.merge_from_file(config)
    print(cfg)
    log_dir = os.path.join(cfg.RESULTS_BASE_PATH, cfg.TRAIN.LOG_DIR)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    init_log('global', logging.INFO)
    if cfg.TRAIN.LOG_DIR:
        add_file_handler('global', os.path.join(log_dir, 'logs.txt'), logging.INFO)

    # create model
    model = ModelBuilder().train()
    # model = convert_syncbn_model(model)
    model = model.cuda()
    if cfg.TRAIN.INIT_WEIGHT:
        init_weights(model)
        logger.info("init weight using kaiming_normal_")

    # load pretrained backbone weights
    if cfg.BACKBONE.PRETRAINED:
        cur_path = os.path.dirname(os.path.realpath(__file__))
        backbone_path = os.path.join(cur_path, '../', cfg.BACKBONE.PRETRAINED)
        load_pretrain(model.backbone, backbone_path)

    # create tensorboard writer
    if cfg.TRAIN.LOG_DIR:
        tb_writer = SummaryWriter(log_dir)
    else:
        tb_writer = None

    # build dataset loader
    train_loader = build_data_loader()

    # build optimizer and lr_scheduler
    optimizer, lr_scheduler = build_opt_lr(model,
                                           cfg.TRAIN.START_EPOCH)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    # resume training
    if cfg.TRAIN.RESUME:
        logger.info("resume from {}".format(cfg.TRAIN.RESUME))
        assert os.path.isfile(cfg.TRAIN.RESUME), \
            '{} is not a valid file.'.format(cfg.TRAIN.RESUME)
        model, optimizer, cfg.TRAIN.START_EPOCH = \
            restore_from(model, optimizer, cfg.TRAIN.RESUME)
    # load pretrain
    elif cfg.TRAIN.PRETRAINED:
        load_pretrain(model, cfg.TRAIN.PRETRAINED)

    logger.info(lr_scheduler)
    logger.info("model prepare done")
    # logger.info(describe(model))

    # start training
    train(train_loader, model, optimizer, lr_scheduler, tb_writer)


if __name__ == '__main__':
    seed_torch(123456)
    base_path = "D:/PycharmWorkspaces/PysotPlus/experiments"
    config = "siamrpn_r50_l234_dwxcorr/5_union_rpnpp_l234.yaml"
    # config = "feature_confusion_2gpu_5_union/5_union_rpnpp_c234_feature_confusion_rpn_cls_sum_4.yaml"
    # config = "feature_confusion_2gpu_5_union/5_union_ban_c234_feature_confusion_rpn_cls_sum_4.yaml"
    torch.autograd.set_detect_anomaly(True)
    main("%s/%s" % (base_path, config))

    # config = "feature_confusion_2gpu_5_union/rpnpp.yaml"
    # cfg.merge_from_file("%s/%s" % (base_path, config))
    # model = ModelBuilder()
    # print(sum([p.numel() for p in model.parameters()]))
    # keys = list(model.connect.state_dict().keys())
    # print(keys)
    # print(model.connect.state_dict()[keys[0]])
    # load_pretrain(model, "C:/Users/YSQPCL/Desktop/checkpoint_e19.pth")
    # print(model.connect.state_dict()[keys[0]])
    # print(describe(model))
    # print(model)
