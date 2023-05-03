#  SeiT
#  Copyright (c) 2023-present NAVER Cloud Corp.
#  Apache-2.0
import sys
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from torchvision import transforms
from torchvision.datasets.folder import ImageFolder

from engine import tokenize_and_evaluate
from token_transform import build_codebook, CC

from timm.models import create_model
import models
import utils

import warnings
warnings.filterwarnings("ignore", message="Argument interpolation should be of type InterpolationMode instead of int. Please, use InterpolationMode enum.")


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)

    # Model parameters
    parser.add_argument('--model', default='deit_base_token_32', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--model-path', default='weights/trained-vit.ckpt', type=str, metavar='MODEL',
                        help='path to trained ViT weight')
    parser.add_argument('--codebook-path', default='weights/codebook.ckpt', type=str,
                        help='path to pretrained codebook')
    parser.add_argument('--vit-input-size', default=28, type=int, help='input image size for classifier')

    # Tokenizer parameters
    parser.add_argument('--tokenizer-path', default='weights/tokenizer.ckpt', type=str, metavar='MODEL',
                        help='path to tokenizer weight')
h    parser.add_argument('--tokenizer-code-path', type=str, metavar='MODEL',
                        help='path to tokenizer code (can be cloned from https://github.com/thuanz123/enhancing-transformers.git)')
    parser.add_argument('--tokenizer-input-size', default=256, type=int, help='input image size for tokenizer')

    # Dataset parameters
    parser.add_argument('--data-path', type=str, help="dataset path")
    parser.add_argument('--nb_classes', default=1000, type=int, help="number of classes")

    # etc.
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


def main(args):
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    codebook = build_codebook(args, is_train=False)

    sys.path.append(args.tokenizer_code_path)
    tokenizer = torch.load(args.tokenizer_path)
    tokenizer.to(device)

    token_transform = CC(args.vit_input_size)

    transform = transforms.Compose([
        transforms.Resize((args.tokenizer_input_size, args.tokenizer_input_size), interpolation=3),
        transforms.ToTensor(),
    ])

    dataset = ImageFolder(args.data_path, transform=transform)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()

        if args.dist_eval:
            if len(dataset) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    print(f"Creating model: {args.model}")

    model_args = {
        "model_name": args.model,
        "pretrained": False,
        "num_classes": args.nb_classes,
    }

    model_args["img_size"] = args.vit_input_size
    model = create_model(**model_args)

    ckpt = torch.load(args.model_path, map_location='cpu')["model"]
    model.load_state_dict(ckpt)
    model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    test_stats = tokenize_and_evaluate(data_loader, model, token_transform, codebook, tokenizer, device)
    print(f"Accuracy of the network on the {len(dataset)} test images: {test_stats['acc1']:.1f}%")

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
