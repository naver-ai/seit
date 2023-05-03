#  SeiT
#  Copyright (c) 2023-present NAVER Cloud Corp.
#  Apache-2.0
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model


class TokenVisionTransformer(VisionTransformer):
    def __init__(self, global_pool=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.global_pool = global_pool

    def reset_patch_embed_conv(self, *args, **kwargs):
        self.patch_embed.proj = nn.Conv2d(*args, **kwargs)

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = x[:, 0]

        return self.pre_logits(x)


@register_model
def deit_small_token_32(pretrained=False, **kwargs):
    model = TokenVisionTransformer(
        global_pool=True, in_chans=32, patch_size=2, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    model.reset_patch_embed_conv(32, 384, kernel_size=(4, 4), stride=2, padding=1)
    return model


@register_model
def deit_base_token_32(pretrained=False, **kwargs):
    model = TokenVisionTransformer(
        global_pool=True, in_chans=32, patch_size=2, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    model.reset_patch_embed_conv(32, 768, kernel_size=(4, 4), stride=2, padding=1)
    return model


@register_model
def deit_small_token_4(pretrained=False, **kwargs):
    model = TokenVisionTransformer(
        global_pool=True, in_chans=4, patch_size=2, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    model.reset_patch_embed_conv(4, 384, kernel_size=(4, 4), stride=2, padding=1)
    return model


@register_model
def deit_base_token_4(pretrained=False, **kwargs):
    model = TokenVisionTransformer(
        global_pool=True, in_chans=4, patch_size=2, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    model.reset_patch_embed_conv(4, 768, kernel_size=(4, 4), stride=2, padding=1)
    return model
