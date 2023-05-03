#  SeiT
#  Copyright (c) 2023-present NAVER Cloud Corp.
#  Apache-2.0
import math
import gzip
import numpy as np

import torch

from token_transform import build_token_transform, build_transform


def build_dataset(is_train, num_tokens, args):
    if is_train:
        token_file = args.train_token_file
        label_file = args.train_label_file

    else:
        token_file = args.val_token_file
        label_file = args.val_label_file

    token_transform = build_token_transform(is_train, args)
    transform = build_transform(is_train, args)

    dataset = TokenDataset(token_file=token_file, label_file=label_file, num_tokens=num_tokens, token_transform=token_transform, transform=transform)
    nb_classes = 1000

    return dataset, nb_classes


class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, token_file: str, label_file: str, num_tokens: int, token_transform=None, transform=None):
        self.tokens, self.start_indices, self.labels = self.make_classes_from_imagelist(token_file, label_file)
        self.num_tokens = num_tokens

        self.token_transform = token_transform
        self.transform = transform

    def make_classes_from_imagelist(self, token_file, label_file):
        index_file = token_file.replace("data.bin", "index.npy")
        tokens, start_indices = load_from_gz(token_file, index_file)
        with open(label_file) as f:
            labels = [int(_l.strip()) for _l in f.readlines()]

        return tokens, start_indices, labels

    def __getitem__(self, index):
        token = get_data_index_of(self.tokens, self.start_indices, index)
        grid = (int(math.sqrt(len(token))),) * 2

        token = torch.LongTensor(token).reshape(*grid)

        if self.token_transform is not None:
            token = self.token_transform(token)

        if self.transform is not None:
            token = torch.nn.functional.one_hot(token, self.num_tokens).float().permute(2, 0, 1)
            token = self.transform(token)

        cid = self.labels[index]

        return token, cid

    def __len__(self):
        return len(self.labels)


def load_from_gz(token_data_fname, token_index_fname):
    with gzip.open(token_data_fname, 'rb') as fin:
        tokens = fin.read()
    with gzip.open(token_index_fname, 'rb') as fin:
        start_indices = np.load(fin)
    return tokens, start_indices


def get_data_index_of(tokens, start_indices, index):
    if len(start_indices) - 2 < index:
        raise IndexError(f'{len(start_indices)=} but got {index=}')
    cur_tokens = tokens[start_indices[index]:start_indices[index+1]]

    _token = 0
    converted_tokens = []
    for token in cur_tokens:
        if token == 255:
            _token = 255
            continue
        token += _token
        _token = 0
        converted_tokens.append(token)

    if len(converted_tokens) != 1024:
        raise ValueError(f'{converted_tokens=}')
    return converted_tokens
