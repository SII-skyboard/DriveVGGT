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

class Attention_MV(nn.Module):
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
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
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
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor,v: Tensor, pos=None) -> Tensor:
        
        B, Nq, P, C = x.shape
        B, Nv, P, C = v.shape
        q = x.reshape(B,Nq*P,C).reshape(B, Nq*P, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B,Nv*P,C).reshape(B, Nv*P, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k = self.q_norm(q), self.k_norm(v)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, Nq, P, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Sparse_attention(nn.Module):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=1,
        num_heads=16,
        mlp_ratio=4.0,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        qk_norm=True,
        rope_freq=-1,
        init_values=0.01,
    ):
        super().__init__()
        self.depth = 1
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None 
        self.use_reentrant = False # hardcoded to False
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
    def similarity(self,cam_tokens):
        # self.similarity(multiview_aggregated_tokens_pose) # 1 30 1024 cam pose token
        L = 5
        K = 6
        # mask single view token
        similarity_matrix = cam_tokens@cam_tokens.transpose(1,2)/32
        
        for index in range(6):
            chunk_index_start = L*index
            similarity_matrix[:,chunk_index_start:chunk_index_start+L,chunk_index_start:chunk_index_start+L] -= 100000 # B,30,30
        # similarity_matrix += 100000*torch.eye(30).to('cuda')
        similarity_matrix = torch.softmax(similarity_matrix,dim=-1) # 24*1 30 30
        values,indices = torch.topk(similarity_matrix, K, dim=-1)

        return values, indices

    def forward(self, tokens,idx):
        '''
        tokens: B 30 745 1024 (5*6:seq*cam_list)
        '''    
        cam_num = 6
        B, S, P, C = tokens.size() # 1 30 745 1024
        cam_tokens = tokens[:,:,0,:] # B 30 1024
        values, indices = self.similarity(cam_tokens)

        selected_tokens_list = []
        selected_tokens_list_bs = []
        for bs_index in range(B):
            for cam_index in range(S):
                selected_tokens = tokens[bs_index,indices[bs_index,cam_index,:],:,:].unsqueeze(0)
                selected_tokens_list.append(selected_tokens)
            selected_tokens_bs = torch.cat(selected_tokens_list,dim=0).unsqueeze(0)
            selected_tokens_list_bs.append(selected_tokens_bs)
        selected_tokens = torch.cat(selected_tokens_list_bs,dim=0).view(B*S,-1,C)  # 1*30 6 745 1024
        
        pos = None
        if self.depth == 1:
            idx = 0
            
        if self.training:
            selected_tokens = checkpoint(self.multiview_blocks[idx], selected_tokens, pos, use_reentrant=self.use_reentrant)
        else:
            selected_tokens = self.multiview_blocks[idx](selected_tokens, pos=pos)
            
        selected_tokens = torch.mean(selected_tokens.view(B*S,cam_num,-1,C)*values.view(B*S,-1).unsqueeze(-1).unsqueeze(-1),dim=1)  # 1*30 6 745 1024
        
        selected_tokens = selected_tokens.view(B,S,-1,P,C) # 1 30 8 745 1024
        selected_tokens = selected_tokens.view(B, S, P, C)
        
        return selected_tokens
        