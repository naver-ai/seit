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

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mtm as models
from token_transform import build_codebook, build_geo_transform, build_color_transform
from datasets import build_dataset

from engine_mtm import pretrain_one_epoch

from util.model_saver import ModelSaver

import warnings
warnings.filterwarnings("ignore", message="Argument interpolation should be of type InterpolationMode instead of int. " "Please, use InterpolationMode enum.")


def get_args_parser():
    parser = argparse.ArgumentParser('MTM pre-training', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_token_vit_base_patch16_dec512d4b_k2s2_vitvqgan_32', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # MASK params
    parser.add_argument('--mask_ratio_min', type=float, default=0.4,
                        help='Minimum mask ratio')
    parser.add_argument('--mask_ratio_max', type=float, default=1.0,
                        help='Maximum mask ratio')
    parser.add_argument('--mask_ratio_mu', type=float, default=0.45,
                        help='Mask ratio distribution peak')
    parser.add_argument('--mask_ratio_std', type=float, default=0.25,
                        help='Mask ratio distribution std')
    parser.add_argument('--grad_clip', type=float, default=3.0,
                        help='Gradient clip')

    # Token dataset params
    parser.add_argument('--token_patch_size', default=8, type=int,
                        help='token patch size')
    parser.add_argument('--codebook-path', default='tokens/codebook.ckpt', type=str,
                        help='path to pretrained codebook')
    parser.add_argument('--train-token-file', default='tokens/imagenet1k-train-token.data.bin.gz', type=str, help='Token file path')
    parser.add_argument('--train-label-file', default='tokens/imagenet1k-train-label.txt', type=str, help='Token file path')

    # Token Augmentation parameters
    parser.add_argument('--tokenadapt-path', default='weights/tokenadapt.ckpt', type=str, metavar='MODEL',
                        help='path to tokenadapt weight')
    parser.add_argument('--rrc-ratio', type=float, default=0.2,
                        help='RRC min ratio')  # RRC ratio
    parser.add_argument('--src', action='store_true')
    parser.add_argument('--rrc', action='store_true')
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--affine', action='store_true')
    parser.add_argument('--token-coloradapt-prob', type=float, default=0.0)
    parser.add_argument('--token-noise-prob', type=float, default=0.0,
                        help='token noise prob')
    parser.add_argument('--token-noise-scale', type=float, default=1.0,
                        help='token noise scale')
    parser.add_argument('--token-noise-std', type=float, default=1.0,
                        help='token noise std')

    parser.add_argument('--eda-rc-prob', type=float, default=0.0)
    parser.add_argument('--token-synonym-dict', default="tokens/synonyms.json", type=str)
    parser.add_argument('--token-synonym-thres', type=int, default=5)
    parser.add_argument('--replace-token-prob', type=float, default=0.25)
    parser.add_argument('--eda-sc-prob', type=float, default=0.0)
    parser.add_argument('--swap-token-scale', type=float, default=0.25)

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--save_periods', nargs='+', default=['last2', 'every_200_epochs'],
                        help='periods for save')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # Extra parameters
    parser.add_argument('--print_freq', default=100, type=int,
                        help='frequency for print operation')

    # Additional parameters
    parser.add_argument('--smoothing', type=float, default=0.0, help='Label smoothing (default: 0.0)')
    parser.add_argument('--bce_loss', action='store_true')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    cdb, num_tokens = build_codebook(args)
    cdb.to(device)

    saver = None
    if args.output_dir:
        saver = ModelSaver(checkpoint_dir=args.output_dir, target='local',
                           periods=args.save_periods)

    dataset_train, _ = build_dataset(is_train=True, num_tokens=num_tokens, args=args)
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    tf_color = build_color_transform(True, args)
    tf_base = build_geo_transform(False, args)
    tf_tokenadapt = build_geo_transform(True, args)

    # define the model
    model = models.__dict__[args.model](mask_ratio_mu=args.mask_ratio_mu, mask_ratio_std=args.mask_ratio_std,
                                        mask_ratio_min=args.mask_ratio_min, mask_ratio_max=args.mask_ratio_max, smoothing=args.smoothing, bce_loss=args.bce_loss)

    if args.tokenadapt_path is not None:
        ta_state_dict = torch.load(args.tokenadapt_path, map_location='cpu')
        model.load_state_dict(ta_state_dict, strict=False)
        for name, p in model.named_parameters():
            if 'conversion_function' in name or 'reverse_transform' in name:
                p.requires_grad = False

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = pretrain_one_epoch(
            model, cdb, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            tf_color=tf_color, tf_base=tf_base, tf_tokenadapt=tf_tokenadapt, args=args,
            print_freq=args.print_freq
        )

        save_flag = misc.is_main_process()
        if args.output_dir and save_flag:
            save_dict = misc.save_dict(args=args, model=model,
                                       model_without_ddp=model_without_ddp,
                                       optimizer=optimizer,
                                       loss_scaler=loss_scaler, epoch=epoch)
            saver.save(step=epoch,
                       num_steps=args.epochs,
                       state=save_dict,
                       summary={'epoch': '%d/%d' % (epoch+1, args.epochs), **train_stats, })

        log_stats = {**{f'pretrain/{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}

        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
