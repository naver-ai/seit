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
import math
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mtm as models
from token_transform import build_codebook, build_geo_transform, build_color_transform
from datasets import build_dataset

from engine_mtm import finetune_one_epoch, evaluate
from util.model_saver import ModelSaver

import warnings
warnings.filterwarnings("ignore", message="Argument interpolation should be of type InterpolationMode instead of int. Please, use InterpolationMode enum.")


def get_args_parser():
    parser = argparse.ArgumentParser('MTM fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='deit_base_token_32', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='weight decay (default: 0.1)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.0)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 0.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--save_periods', nargs='+',
                        default=['last2', 'best'],
                        help='periods for save')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Token dataset params
    parser.add_argument('--token_patch_size', default=8, type=int,
                        help='token patch size')
    parser.add_argument('--codebook-path', default='tokens/codebook.ckpt', type=str,
                        help='path to pretrained codebook')
    parser.add_argument('--train-token-file', default='tokens/imagenet1k-train-token.data.bin.gz', type=str, help='Token file path')
    parser.add_argument('--train-label-file', default='tokens/imagenet1k-train-label.txt', type=str, help='Token file path')
    parser.add_argument('--val-token-file', default='tokens/imagenet1k-val-token.data.bin.gz', type=str, help='Token file path')
    parser.add_argument('--val-label-file', default='tokens/imagenet1k-val-label.txt', type=str, help='Token file path')

    # Token Augmentation parameters
    parser.add_argument('--tokenadapt-path', default='weights/tokenadapt.ckpt', type=str, metavar='MODEL',
                        help='path to tokenadapt weight')
    parser.add_argument('--rrc-ratio', type=float, default=0.08,
                        help='RRC min ratio')  # RRC ratio (imagenet default: 0.08)
    parser.add_argument('--src', action='store_true')  # simple random crop
    parser.add_argument('--rrc', action='store_true')  # random random crop
    parser.add_argument('--eda-rc-prob', type=float, default=0.0)
    parser.add_argument('--token-synonym-dict', default="tokens/codebook-synonyms.json", type=str)
    parser.add_argument('--token-synonym-thres', type=int, default=5)
    parser.add_argument('--replace-token-prob', type=float, default=0.25)
    parser.add_argument('--eda-sc-prob', type=float, default=0.0)
    parser.add_argument('--swap-token-scale', type=float, default=0.25)
    parser.add_argument('--token-coloradapt-prob', type=float, default=0.25)
    parser.add_argument('--token-noise-prob', type=float, default=0.75,
                        help='token noise prob')
    parser.add_argument('--token-noise-scale', type=float, default=1.0,
                        help='token noise scale')
    parser.add_argument('--token-noise-std', type=float, default=1.0,
                        help='token noise std')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # Extra parameters
    parser.add_argument('--print_freq', default=200, type=int,
                        help='frequency for print operation')

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
        saver = ModelSaver(checkpoint_dir=args.output_dir,
                           target='local',
                           periods=args.save_periods)

    dataset_train, args.nb_classes = build_dataset(is_train=True, num_tokens=num_tokens, args=args)
    dataset_val, _ = build_dataset(is_train=False, num_tokens=num_tokens, args=args)

    print(dataset_train)
    print(dataset_val)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    tf_color = build_color_transform(True, args)
    tf_base = build_geo_transform(False, args)
    tf_tokenadapt = build_geo_transform(True, args)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    model = models.__dict__[args.model](
        drop_path_rate=args.drop_path,
    )

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    if args.tokenadapt_path is not None:
        ta_state_dict = torch.load(args.tokenadapt_path, map_location='cpu')
        model.load_state_dict(ta_state_dict, strict=False)
        for name, p in model.named_parameters():
            if 'conversion_function' in name or 'reverse_transform' in name:
                p.requires_grad = False

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * math.sqrt(eff_batch_size) / 8

    print("base lr: %.2e" % (args.lr * 8 / math.sqrt(eff_batch_size)))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                        no_weight_decay_list=model_without_ddp.no_weight_decay(),
                                        layer_decay=args.layer_decay
                                        )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    if args.eval:
        test_stats = evaluate(data_loader_val, model, cdb, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = finetune_one_epoch(
            model, cdb, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            tf_color=tf_color, tf_base=tf_base, tf_tokenadapt=tf_tokenadapt, args=args,
            print_freq=args.print_freq
        )
        if args.output_dir and misc.is_main_process():
            save_dict = misc.save_dict(args=args, model=model,
                                       model_without_ddp=model_without_ddp,
                                       optimizer=optimizer,
                                       loss_scaler=loss_scaler,
                                       epoch=epoch)

        test_stats = evaluate(data_loader_val, model, cdb, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        is_best = False
        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]
            is_best = True
        print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {
            **{f'finetune/train_{k}': v for k, v in train_stats.items()},
            **{f'finetune/test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch}

        if args.output_dir and misc.is_main_process():
            saver.save(step=epoch, num_steps=args.epochs, state=save_dict,
                       summary={'accuracy': test_stats["acc1"],
                                'epoch': '%d/%d' % (epoch + 1, args.epochs),
                                **log_stats, }, is_best=is_best)

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
