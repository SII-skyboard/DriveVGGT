# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

from torch import Tensor
from torch import nn
import torch.nn.functional as F
import random
from functools import lru_cache, partial

import torch

# from .flex_util import flex_atten

from torch.nn.attention.flex_attention import _mask_mod_signature
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
XFORMERS_AVAILABLE = False


def generate_sliding_window(window_size: int) -> _mask_mod_signature:

    def sliding_window(b, h, q_idx, kv_idx):
        del b, h # not used
        # return torch.abs(q_idx - kv_idx) <= window_size // 2
        return torch.abs(q_idx - kv_idx) >= 0

    sliding_window_mask = sliding_window
    sliding_window_mask.__name__ = f"sliding_window_{window_size}"
    return sliding_window_mask


class Attention_attenmap(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = False,  # use F.scaled_dot_product_attention or not
        rope=None,
        S = 742*6*3
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.v_norm = nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        sliding_window_mask = generate_sliding_window(window_size=512)
        
        self.block_mask = create_block_mask(
        sliding_window_mask, B=None, H=None, Q_LEN=S, KV_LEN=S, _compile=True
        )
        self.opt_flex_attention = torch.compile(partial(flex_attention, block_mask=self.block_mask))
        self.kernel_options = {
                            "BLOCK_M": 64,
                            "BLOCK_N": 64,
                            "BLOCK_M1": 32,
                            "BLOCK_N1": 64,
                            "BLOCK_M2": 64,
                            "BLOCK_N2": 32,
                        }

    def forward(self, x: Tensor, pos=None,q_idx=None) -> Tensor:
        # print(x.size())
        # print(pos.size())
        # print(q_idx)
        # assert 1==0

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        v = self.v_norm(v)
        # if self.rope is not None:
        #     q = self.rope(q, pos)
        #     k = self.rope(k, pos)

        B,H,S,D = q.size()

        # block_mask = create_block_mask(
        # sliding_window_mask, B=None, H=None, Q_LEN=S, KV_LEN=S, _compile=True
        # )
        
        x = self.opt_flex_attention(q.to(dtype=torch.float16), k.to(dtype=torch.float16), v.to(dtype=torch.float16), kernel_options=self.kernel_options)
        # x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


