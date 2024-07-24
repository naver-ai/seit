#  SeiT++
#  Copyright (c) 2024-present NAVER Cloud Corp.
#  CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as stats

from timm.models.vision_transformer import VisionTransformer, _cfg, PatchEmbed, Block
from timm.models.registry import register_model
from util.pos_embed import get_2d_sincos_pos_embed
from layers import ViTEncoder as Encoder, ViTDecoder as Decoder


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss


class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        torch.nn.init.normal_(self.word_embeddings.weight, std=.02)

    def forward(self, input_ids):
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MlmLayer(nn.Module):

    def __init__(self, feat_emb_dim, word_emb_dim, vocab_size):
        super().__init__()

        self.word_emb_dim = word_emb_dim // 4

        self.fc = nn.Linear(feat_emb_dim, word_emb_dim)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(self.word_emb_dim)
        self.bias = nn.Parameter(torch.zeros(1, 1, vocab_size))

    def forward(self, x, word_embeddings):
        mlm_hidden = self.fc(x)

        logits = mlm_hidden[:, 1:, ...].reshape(mlm_hidden.shape[0], 14, 14, 2, 2, self.word_emb_dim)
        logits = torch.einsum('nhwpqc->nhpwqc', logits)
        logits = logits.reshape(logits.shape[0], 28*28, self.word_emb_dim)

        mlm_hidden = logits

        mlm_hidden = self.gelu(mlm_hidden)
        mlm_hidden = self.ln(mlm_hidden)
        word_embeddings = word_embeddings.transpose(0, 1)
        logits = torch.matmul(mlm_hidden, word_embeddings)
        logits = logits + self.bias

        return logits


class TokenVisionTransformer(VisionTransformer):
    def __init__(self, global_pool=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.global_pool = global_pool
        self.codebook_size = 391

        self.token_emb = BertEmbeddings(vocab_size=self.codebook_size,
                                        hidden_size=32,
                                        dropout=0.1)

        self.conversion_function = nn.Sequential(
            nn.Linear(32, 384),
            Decoder(image_size=64, patch_size=2, dim=384, depth=1, heads=6, mlp_dim=384*4, channels=32),
        )

        self.reverse_transform = nn.Sequential(
            Encoder(image_size=56, patch_size=2, dim=384, depth=1, heads=6, mlp_dim=384*4, channels=32),
            nn.Linear(384, 391),
        )

    def _reset_patch_embed_conv(self, *args, **kwargs):
        self.patch_embed.proj = nn.Conv2d(*args, **kwargs)

    def forward_conversion(self, x):
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.conversion_function(x)

        return x

    def forward_inversion(self, x):
        x = self.reverse_transform(x)
        x = x.argmax(dim=-1)
        x = torch.nn.functional.one_hot(x, 391).float().transpose(1, 2).reshape(x.shape[0], 391, 28, 28).detach()

        return x

    def forward_features(self, x):
        x = x.reshape(x.size(0), -1)
        input_embeddings = self.token_emb(x.detach())
        input_embeddings = input_embeddings.permute(0, 2, 1).reshape(input_embeddings.shape[0], 32, 28, 28)
        x = self.patch_embed(input_embeddings)

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

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


class MaskedAutoencoderTokenViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=28, patch_size=2, in_chans=32,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 mask_ratio_min=0.5, mask_ratio_max=1.0, mask_ratio_mu=0.55, mask_ratio_std=0.25,
                 token_embed_dim=32, codebook_size=391, smoothing=0.1, bce_loss=False):
        super().__init__()

        self.codebook_size = codebook_size
        self.token_embed_dim = token_embed_dim
        self.mask_token_label = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.token_emb = BertEmbeddings(vocab_size=self.codebook_size,
                                        hidden_size=self.token_embed_dim,
                                        dropout=0.1)

        # MTM variant masking ratio
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - mask_ratio_mu) / mask_ratio_std,
                                                    (mask_ratio_max - mask_ratio_mu) / mask_ratio_std,
                                                    loc=mask_ratio_mu, scale=mask_ratio_std)

        # MTM encoder specifics
        dropout_rate = 0.1
        self.patch_embed = PatchEmbed(img_size, patch_size, self.token_embed_dim, 768)
        self.num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # MTM decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim))  # learnable pos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  drop=dropout_rate, attn_drop=dropout_rate)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        # MlmLayer
        self.mlm_layer = MlmLayer(feat_emb_dim=decoder_embed_dim, word_emb_dim=patch_size**2 * self.token_embed_dim, vocab_size=self.codebook_size)

        self.smoothing = smoothing
        self.bce_loss = bce_loss

        if self.bce_loss:
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)

        self.initialize_weights()

        self.conversion_function = nn.Sequential(
            nn.Linear(32, 384),
            Decoder(image_size=64, patch_size=2, dim=384, depth=1, heads=6, mlp_dim=384*4, channels=32),
        )

        self.reverse_transform = nn.Sequential(
            Encoder(image_size=56, patch_size=2, dim=384, depth=1, heads=6, mlp_dim=384*4, channels=32),
            nn.Linear(384, 391),
        )

    def _reset_patch_embed_conv(self, *args, **kwargs):
        self.patch_embed.proj = nn.Conv2d(*args, **kwargs)

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_conversion(self, x):
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.conversion_function(x)

        return x

    def forward_inversion(self, x):
        x = self.reverse_transform(x)
        x = x.argmax(dim=-1)
        x = torch.nn.functional.one_hot(x, 391).float().transpose(1, 2).reshape(x.shape[0], 391, 28, 28).detach()

        return x

    def forward_encoder(self, x):
        x = x.reshape(x.size(0), -1)
        gt_indices = x.clone().detach().long()

        input_embeddings = self.token_emb(x.detach())
        input_embeddings = input_embeddings.permute(0, 2, 1).reshape(input_embeddings.shape[0], -1, 28, 28)
        input_embeddings = self.patch_embed(input_embeddings)

        # calculate the number of tokens to mask based on predefined ratios
        bsz, seq_len, emb_dim = input_embeddings.shape
        mask_ratio_min = self.mask_ratio_min

        num_dropped_tokens = int(np.ceil(seq_len * mask_ratio_min))
        num_masked_tokens = int(np.ceil(seq_len * self.mask_ratio_generator.rvs(1)[0]))

        # it is possible that two elements of the noise is the same, so do a while loop to avoid it
        while True:
            noise = torch.rand(bsz, seq_len, device=x.device)  # generate noise in [0, 1]
            sorted_noise, _ = torch.sort(noise, dim=1)  # ascend: small is remove, large is keep
            cutoff_drop = sorted_noise[:, num_dropped_tokens-1:num_dropped_tokens]
            cutoff_mask = sorted_noise[:, num_masked_tokens-1:num_masked_tokens]
            token_drop_mask = (noise <= cutoff_drop).float()
            token_all_mask = (noise <= cutoff_mask).float()

            if token_drop_mask.sum() == bsz*num_dropped_tokens and token_all_mask.sum() == bsz*num_masked_tokens:
                break

        input_embeddings = self.mask_token_label * token_all_mask.unsqueeze(-1) + input_embeddings * (1 - token_all_mask.unsqueeze(-1))
        input_embeddings = input_embeddings + self.pos_embed[:, 1:, :]

        # dropping
        token_keep_mask = 1 - token_drop_mask
        input_embeddings_after_drop = input_embeddings[token_keep_mask.nonzero(as_tuple=True)].reshape(bsz, -1, emb_dim)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(input_embeddings_after_drop.shape[0], -1, -1)
        token_drop_mask = torch.cat([torch.zeros(x.size(0), 1).cuda(), token_drop_mask], dim=1)
        token_all_mask = torch.cat([torch.zeros(x.size(0), 1).cuda(), token_all_mask], dim=1)
        input_embeddings_after_drop = torch.cat((cls_tokens, input_embeddings_after_drop), dim=1)

        # apply Transformer blocks
        x = input_embeddings_after_drop
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, gt_indices, token_drop_mask, token_all_mask


    def forward_decoder(self, x, token_drop_mask, token_all_mask):
        # embed tokens
        x = self.decoder_embed(x)

        # generate mask tokens
        mask_tokens = x[:, 0:1].repeat(1, token_all_mask.shape[1], 1)

        # put undropped tokens into original sequence
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - token_drop_mask).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        # set undropped but masked positions with mask
        x_after_pad = torch.where(token_all_mask.unsqueeze(-1).bool(), mask_tokens, x_after_pad)

        # add pos embed
        x = x_after_pad + self.decoder_pos_embed_learned

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        # predictor projection
        word_embeddings = self.token_emb.word_embeddings.weight.data.detach()
        x = self.mlm_layer(x, word_embeddings)

        return x


    def forward_loss(self, gt_indices, logits, mask):
        bsz, seq_len = gt_indices.size()
        # logits and mask are with seq_len+1 but gt_indices is with seq_len
        if self.bce_loss:
            gt_indices = gt_indices.reshape(bsz*seq_len)
            gt_indices = torch.nn.functional.one_hot(gt_indices, 391).float()
            loss = self.criterion(logits.reshape(bsz*seq_len, -1), gt_indices).mean(-1)
        else:
            loss = self.criterion(logits.reshape(bsz*seq_len, -1), gt_indices.reshape(bsz*seq_len))
        loss = loss.reshape(bsz, seq_len)

        h = w = 14
        p = 2
        loss = loss.reshape(shape=(loss.shape[0], 28, 28))
        loss = loss.reshape(shape=(loss.shape[0], h, p, w, p))
        loss = torch.einsum('nhpwq->nhwpq', loss)
        loss = loss.reshape(shape=(loss.shape[0], h * w, p**2))

        loss = loss.mean(-1)

        loss = (loss * mask[:, 1:]).sum() / mask[:, 1:].sum()  # mean loss on removed patches
        return loss


    def forward(self, x):
        latent, gt_indices, token_drop_mask, token_all_mask = self.forward_encoder(x)
        logits = self.forward_decoder(latent, token_drop_mask, token_all_mask)
        loss = self.forward_loss(gt_indices, logits, token_all_mask)
        return loss


def mae_token_vit_base_patch16_dec512d4b_k2s2_vitvqgan_32(**kwargs):
    model = MaskedAutoencoderTokenViT(
        img_size=28, patch_size=2, in_chans=32, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), token_embed_dim=32, **kwargs)
    model._reset_patch_embed_conv(32, 768, kernel_size=(2, 2), stride=2)
    return model


@register_model
def deit_base_token_32(pretrained=False, **kwargs):
    model = TokenVisionTransformer(img_size=28,
        global_pool=True, in_chans=32, patch_size=2, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    model._reset_patch_embed_conv(32, 768, kernel_size=(2, 2), stride=2)
    return model
