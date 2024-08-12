#  SeiT++
#  Copyright (c) 2024-present NAVER Cloud Corp.
#  CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)

#  Mostly copy-paste from torchvision references.

import math
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


def build_token_transform(is_train, args):
    if is_train:
        eda_t = []
        eda_prob = []
        if args.eda_rc_prob:
            synonym_dict = json.load(open(args.token_synonym_dict))
            eda_t.append(ReplaceSynonym(synonym_dict, args.token_synonym_thres, args.replace_token_prob))
            eda_prob.append(args.eda_rc_prob)
        if args.eda_sc_prob:
            eda_t.append(SwapCode((0, args.swap_token_scale)))
            eda_prob.append(args.eda_sc_prob)

        if eda_t:
            eda_t.append(nn.Identity())
            eda_prob.append(1-sum(eda_prob))
            return transforms.RandomChoice(eda_t, eda_prob)
        else:
            return None

    else:
        return None


def build_color_transform(is_train, args):
    if is_train:
        color_t = []
        color_prob = []
        if args.token_noise_prob:
            color_t.append(Noise(scale=args.token_noise_scale, std=args.token_noise_std))
            color_prob.append(args.token_noise_prob)
        if args.token_coloradapt_prob:
            color_t.append(ColorAdapt())
            color_prob.append(args.token_coloradapt_prob)

        if color_t:
            color_t.append(nn.Identity())
            color_prob.append(1-sum(color_prob))
            return transforms.RandomChoice(color_t, color_prob)
        else:
            return None

    else:
        return None


def build_geo_transform(is_tokenadapt, args):
    input_size = args.input_size // args.token_patch_size
    t = []
    if is_tokenadapt:
        t.append(transforms.RandomHorizontalFlip())

        t.append(transforms.RandomApply(
            [Rotate(30), TranslateX(0.2), TranslateY(0.2), ShearX(15), ShearY(15)]
        ))

        t.append(RRC(size=input_size * 2, scale=(args.rrc_ratio, 1.0)))
    else:
        t.append(RRC(size=input_size, scale=(args.rrc_ratio, 1.0)))

    return transforms.Compose(t)


def build_basic_transform(is_train, args):
    input_size = args.input_size // args.token_patch_size

    if is_train:
        if args.src:
            t = [RC(size=input_size)]

        elif hasattr(args, "rrc") and not args.rrc:
            t = []

        else:
            t = [RRC(size=input_size, scale=(args.rrc_ratio, 1.0))]

    else:
        t = [CC(size=input_size)]

    return transforms.Compose(t)


def build_codebook(args, is_train=True):
    codebook = torch.load(args.codebook_path).cpu()

    if is_train:
        return Codebook(codebook), codebook.shape[0]

    else:

        return codebook


class Codebook(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.num_embed, self.embed_dim = int(weight.shape[0]), int(weight.shape[1])
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = x @ self.weight
        x = x.permute(0, 3, 1, 2)
        return x

    def __repr__(self):
        return self.__class__.__name__ + f"(weight=({self.num_embed}, {self.embed_dim}))"


class Noise(nn.Module):
    def __init__(self, prob=0.5, scale=1.0, std=1.0):
        super().__init__()
        self.prob = prob
        self.scale = scale
        self.std = std

    def forward(self, x):
        B, C, H, W = x.shape
        B_noise = int(B * self.prob)
        zero_ids = torch.randperm(B)[B_noise:]

        noise = (torch.randn(B, C, 1, 1) * self.scale) + (torch.randn(B, C, H, W) * self.std)
        noise[zero_ids] = 0.
        x += noise.to(x.device)

        return x

    def __repr__(self):
        return self.__class__.__name__ + f"(prob={self.prob}, scale={self.scale}, std={self.std})"


class ReplaceSynonym(nn.Module):
    def __init__(self, synonym_dict, thres=None, prob=0.5):
        super().__init__()
        self.synonym_dict = synonym_dict
        self.thres = thres
        self.prob = prob

    def forward(self, x):
        x_shape = x.shape
        replaced = [random.choice(self.synonym_dict[str(c.item())][:self.thres])
                    if torch.empty(1).uniform_(0, 1) < self.prob else c.item()
                    for c in x.flatten()]
        replaced = torch.LongTensor(replaced).reshape(x_shape)
        return replaced

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + f"(threshold=top{self.thres}, replace_prob={self.prob})"
        return format_string


class SwapCode(nn.Module):
    def __init__(self, patch_scale=(0, 0.25), num_swap=1):
        super().__init__()
        self.patch_scale = patch_scale
        self.num_swap = 1

    def get_params(self, x):
        h, w = x.shape[-2:]
        ms = min(h, w)
        ps = int(ms * torch.empty(1).uniform_(*self.patch_scale).item())
        while True:
            y0, x0 = random.randint(0, h-ps-1), random.randint(0, w-ps-1)
            y1, x1 = random.randint(0, h-ps-1), random.randint(0, w-ps-1)
            if math.sqrt((y1-y0)**2 + (x1-x0)**2) > ps:
                break

        return ps, y0, x0, y1, x1

    def forward(self, x):
        x_shape = x.shape
        ps, y0, x0, y1, x1 = self.get_params(x)
        x_ = x.clone().reshape(x_shape[-2:])
        x_[y0:y0+ps, x0:x0+ps] = x[y1:y1+ps, x1:x1+ps]
        x_[y1:y1+ps, x1:x1+ps] = x[y0:y0+ps, x0:x0+ps]

        return x_.reshape(x_shape)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + f"(scale={self.patch_scale})"
        return format_string


class RRC(nn.Module):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0), interpolation="bicubic"):
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def get_params(self, x, scale, ratio):
        height, width = x.shape[-2:]
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, x):
        if len(x.shape) == 3:
            i, j, h, w = self.get_params(x, self.scale, self.ratio)
            x = x[:, i:(i+h), j:(j+w)]
            x = F.interpolate(x.unsqueeze(0), size=self.size, mode=self.interpolation).squeeze()
        elif len(x.shape) == 4:
            x = torch.stack([self.forward(_x) for _x in x])
        else:
            raise ValueError(f"Invalid input shape: {x.shape}")

        return x

    def __repr__(self) -> str:
        interpolate_str = self.interpolation
        format_string = self.__class__.__name__ + f"(size={self.size}"
        format_string += f", scale={tuple(round(s, 4) for s in self.scale)}"
        format_string += f", ratio={tuple(round(r, 4) for r in self.ratio)}"
        format_string += f", interpolation={interpolate_str})"
        return format_string


class Resize(nn.Module):
    def __init__(self, size, interpolation="bicubic"):
        super().__init__()
        self.size = size
        self.interpolation = interpolation

    def forward(self, x):
        if isinstance(self.size, int):
            h, w = x.shape[-2:]
            if min(h, w) == self.size:
                return x
            elif h > w:
                ratio = h / w
                size = (int(self.size*ratio), self.size)
            else:
                ratio = w / h
                size = (self.size, int(self.size*ratio))

        if len(x.shape) == 3:
            x = F.interpolate(x.unsqueeze(0), size=size, mode=self.interpolation).squeeze()
        elif len(x.shape) == 4:
            x = F.interpolate(x, size=size, mode=self.interpolation)
        else:
            raise ValueError(f"Invalid input shape: {x.shape}")

        return x

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + f"(size={self.size})" + f"(interpolation={ self.interpolation.value})"
        return format_string


class CC(nn.Module):
    def __init__(self, size):
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def forward(self, x):
        h, w = x.shape[-2:]
        ystart = (h - self.size[0]) // 2
        xstart = (w - self.size[1]) // 2

        if len(x.shape) == 3:
            x = x[:, ystart:(ystart+self.size[0]), xstart:(xstart+self.size[1])]
        elif len(x.shape) == 4:
            x = x[:, :, ystart:(ystart+self.size[0]), xstart:(xstart+self.size[1])]
        else:
            raise ValueError(f"Invalid input shape: {x.shape}")

        return x

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + f"(size={self.size})"
        return format_string


class RC(nn.Module):
    def __init__(self, size):
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def forward(self, x):
        h, w = x.shape[-2:]
        ystart = torch.randint(0, h - self.size[0] + 1, size=(1,)).item()
        xstart = torch.randint(0, w - self.size[1] + 1, size=(1,)).item()

        if len(x.shape) == 3:
            x = x[:, ystart:(ystart+self.size[0]), xstart:(xstart+self.size[1])]
        elif len(x.shape) == 4:
            x = torch.stack([self.forward(_x) for _x in x])
        else:
            raise ValueError(f"Invalid input shape: {x.shape}")

        return x

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + f"(size={self.size})"
        return format_string


class ColorAdapt(nn.Module):
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        B, C, H, W = x.shape

        x_ca = x.clone()

        x_mu = x_ca.mean(dim=(2, 3), keepdim=True)
        x_std = x_ca.std(dim=(2, 3), keepdim=True)
        x_mu2 = x_ca.flip(0).mean(dim=(2, 3), keepdim=True)
        x_std2 = x_ca.flip(0).std(dim=(2, 3), keepdim=True)

        x_ca = (x_ca - x_mu) / x_std
        x_ca = x_ca * x_std2 + x_mu2

        ca_ids = torch.randperm(B)[:int(B * self.prob)]
        x[ca_ids] = x_ca[ca_ids]

        return x

    def __repr__(self):
        return self.__class__.__name__ + f"(prob={self.prob})"


class Rotate(nn.Module):
    def __init__(self, degrees, p=0.5):
        super().__init__()
        self.tf = transforms.RandomAffine(degrees=degrees, interpolation=2)
        self.p = p

    def forward(self, x):
        if torch.rand(1) < self.p:
            return self.tf(x)
        return x


class TranslateX(nn.Module):
    def __init__(self, value, p=0.5):
        super().__init__()
        self.tf = transforms.RandomAffine(degrees=0, translate=[value, 0], interpolation=2)
        self.p = p

    def forward(self, x):
        if torch.rand(1) < self.p:
            return self.tf(x)
        return x


class TranslateY(nn.Module):
    def __init__(self, value, p=0.5):
        super().__init__()
        self.tf = transforms.RandomAffine(degrees=0, translate=[0, value], interpolation=2)
        self.p = p

    def forward(self, x):
        if torch.rand(1) < self.p:
            return self.tf(x)
        return x


class ShearX(nn.Module):
    def __init__(self, value, p=0.5):
        super().__init__()
        self.tf = transforms.RandomAffine(degrees=0, shear=[0, value], interpolation=2)
        self.p = p

    def forward(self, x):
        if torch.rand(1) < self.p:
            return self.tf(x)
        return x


class ShearY(nn.Module):
    def __init__(self, value, p=0.5):
        super().__init__()
        self.tf = transforms.RandomAffine(degrees=0, shear=[0, 0, 0, value], interpolation=2)
        self.p = p

    def forward(self, x):
        if torch.rand(1) < self.p:
            return self.tf(x)
        return x
