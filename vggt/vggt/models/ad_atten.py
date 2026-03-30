import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any
import numpy as np
from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from torch import Tensor
import cv2 as cv
from vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri


class AD_atten(nn.Module):
    
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global","multiview"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
    ):
        super().__init__()
        self.seq = 5
        self.view_num = 6
        self.position_encoder = nn.Linear(9, embed_dim)
        self.multiview_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )
        self.seq_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )
        
    def forward(self, tokens, B, S, P, C, multiview_idx, pos, others, img_shape):
        
        tokens = tokens.view(B,self.seq,self.view_num,P,C).permute(0,2,1,3,4).contiguous().view(B*self.view_num,self.seq*P,C)
        if pos is not None :
            pos = pos.view(B,self.seq,self.view_num,P,2).permute(0,2,1,3,4).contiguous().view(B*self.view_num,self.seq*P,2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                    # print("frame",frame_idx)
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                    # print("global",global_idx)
                elif attn_type == "multiview":
                    tokens, multiview_idx, mv_intermediates = self._process_multiview_attention(
                        tokens, B, S, P, C, multiview_idx, pos=pos, others=others, img_shape=(H,W)
                    )
                    # print("multiview",multiview_idx)
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")


