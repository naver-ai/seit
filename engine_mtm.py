#  SeiT++
#  Copyright (c) 2024-present NAVER Cloud Corp.
#  CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

from typing import Iterable, Optional

import sys
import math
import torch

import util.misc as misc
import util.lr_sched as lr_sched

from timm.data import Mixup
from timm.utils import accuracy


def pretrain_one_epoch(model: torch.nn.Module, codebook: torch.nn.Module,
                       data_loader: Iterable, optimizer: torch.optim.Optimizer,
                       device: torch.device, epoch: int, loss_scaler,
                       tf_color=None, tf_base=None, tf_tokenadapt=None, args=None,
                       print_freq=20):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            samples = codebook(samples).detach()
            if tf_color is not None:
                samples = tf_color(samples)

            samples, samples_ta = samples.chunk(2)

            samples_ta = model.module.forward_conversion(samples_ta)

            samples_ta = tf_tokenadapt(samples_ta)

            samples_ta = model.module.forward_inversion(samples_ta)
            samples_ta = codebook(samples_ta).detach()

            samples = tf_base(samples)

            samples = torch.cat((samples, samples_ta), dim=0)

            loss = model(misc.convert_to_codes(samples, codebook.weight.data))

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        grad_norm = loss_scaler(loss, optimizer, parameters=model.parameters(),
                                update_grad=(data_iter_step + 1) % accum_iter == 0)
        loss_scale_value = loss_scaler.state_dict()["scale"]
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        weight_decay_value = 0
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def finetune_one_epoch(model: torch.nn.Module, codebook: torch.nn.Module, criterion: torch.nn.Module,
                       data_loader: Iterable, optimizer: torch.optim.Optimizer,
                       device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                       mixup_fn: Optional[Mixup] = None, tf_color=None, tf_base=None, tf_tokenadapt=None,
                       args=None, print_freq=20):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    accum_iter = args.accum_iter

    mixup_ta_fn = Mixup(
        mixup_alpha=0.8, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
        prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
        label_smoothing=args.smoothing, num_classes=args.nb_classes)

    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            samples = codebook(samples).detach()
            samples = tf_color(samples)

            samples, samples_ta = samples.chunk(2)
            targets, targets_ta = targets.chunk(2)

            samples_ta = model.module.forward_conversion(samples_ta)

            samples_ta = tf_tokenadapt(samples_ta)
            samples_ta, targets_ta = mixup_ta_fn(samples_ta, targets_ta)

            samples_ta = model.module.forward_inversion(samples_ta)
            samples_ta = codebook(samples_ta).detach()

            samples = tf_base(samples)
            samples, targets = mixup_fn(samples, targets)

            samples = torch.cat((samples, samples_ta), dim=0)
            targets = torch.cat((targets, targets_ta), dim=0)

            outputs = model(misc.convert_to_codes(samples, codebook.weight.data))
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=False,
                                update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, codebook, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            images = codebook(images)
            output = model(misc.convert_to_codes(images, codebook.weight.data))
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
