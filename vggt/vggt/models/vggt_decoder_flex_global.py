# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator

from vggt.heads.camera_head import CameraHead,CameraHead_m,CameraHead_trans,CameraHead_decoder
from vggt.heads.dpt_head import DPTHead,DPTHead_m
from vggt.heads.track_head import TrackHead
from vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri,extri_to_pose_encoding
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter

from vggt.layers.utils.block_mv import Block_mv
from vggt.models.utils.seperate_camera_head import CameraHead_seperate
from torch.utils.checkpoint import checkpoint


class VGGT_decoder_flex_global(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True,cam_num=6):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        # self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1",intermediate_layer_idx=[0,1,2,3]) if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

        # add seq_multiview atten
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.camera_relative_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.cam_num = cam_num
        self.rel_pose_embed = nn.Linear(7,2048)
        self.layer_norm = nn.LayerNorm([2048], eps=1e-05, elementwise_affine=True)
        self.rope = RotaryPositionEmbedding2D(frequency=100) 
        self.position_getter = PositionGetter() if self.rope is not None else None
        self.patch_size = 14
        self.batch_norm = nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True)
        depth = 1
        self.depth = depth
        self.mv_blocks = nn.ModuleList(
            [
                Block_mv(
                    dim=embed_dim*2,
                    num_heads=16,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    init_values=0.01,
                    qk_norm=True,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )
        
    def camera_tokens_agg(self,camera_tokens,token_type,relative_pose_enc = None):
        B,S,C = camera_tokens.size()
        camera_tokens = camera_tokens.reshape(B,self.cam_num,S//self.cam_num,C)
        if token_type == "multiview":
            camera_tokens = torch.mean(camera_tokens,dim=2) # 1 6 1 2048
            # camera_tokens = (relative_pose_enc + camera_tokens)/2 # 1 6 1 2048
        if token_type == "frame":
            camera_tokens = torch.mean(camera_tokens,dim=1)
        return camera_tokens.unsqueeze(2) # B X 1 C

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None,others = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        # If without batch dimension, add it

        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        B,S,_,H,W = images.size()   
        f_num = S//self.cam_num
        images = images.reshape(B*self.cam_num,f_num,3,H,W) 
        
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        images = images.view(B,S,3,H,W)
        
        
        real_world_extrinics = others["extrinsics"] # B S 3 4
        real_world_extrinics_enc = extri_to_pose_encoding(real_world_extrinics) # B S 7(4+3)
        
        relative_pose_enc = real_world_extrinics_enc.reshape(B,self.cam_num,f_num,7)[:,:,0,:] # B 6 7 (4+3)
        
        relative_pose_enc_T, relative_pose_enc_R = relative_pose_enc[...,:3],relative_pose_enc[...,3:]

        # input norm
        mean_t = torch.mean(relative_pose_enc_T, dim=-1).unsqueeze(-1)
        std_t = torch.std(relative_pose_enc_T, dim=-1).unsqueeze(-1)
        std_t = torch.clamp(std_t,1e-5,1e2)
        norm_relative_pose_enc_T = (relative_pose_enc_T - mean_t)/std_t
        relative_pose_enc = torch.cat([norm_relative_pose_enc_T/10,relative_pose_enc_R],dim=-1)
        
        relative_pose_enc = self.rel_pose_embed(relative_pose_enc)
        relative_pose_enc = self.layer_norm(relative_pose_enc) # B 6 2048

        
        # # write when concerning scale 
        # relative_pose_enc_T, relative_pose_enc_R = relative_pose_enc[...,:3],relative_pose_enc[...,3:]
        # print(relative_pose_enc,relative_pose_enc.size())
        # mean_t = torch.mean(relative_pose_enc_T, dim=-1)
        # std_t = torch.std(relative_pose_enc_T, dim=-1)
        # norm_relative_pose_enc_T = (relative_pose_enc_T - mean_t)/std_t
        # norm_relative_pose_enc = torch.cat([relative_pose_enc_R,norm_relative_pose_enc_T])
        # print(norm_relative_pose_enc)
        
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)
            pos = pos + 1
            pos_special = torch.zeros(B * S, 2, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1) # 6 742 2
            
        last_pose_enc = aggregated_tokens_list[-1][...,0,:]# 6 9 2048
        selected_list = [4, 11, 17, 23]
        aggregated_tokens_list = [aggregated_tokens_list[i] for i in selected_list]
        layer_nums = len(aggregated_tokens_list)
        
        agg_layers_depths_tokens_list = []
        agg_layers_camera_frame_tokens_list = []
        agg_layers_camera_relative_tokens_list = []
        for layer in range(layer_nums):
            agg_frame_tokens_list = []
            aggregated_tokens = aggregated_tokens_list[layer] # 6 9 745 2048
            aggregated_tokens = self.batch_norm(aggregated_tokens.reshape(-1,745,2048).permute(0,2,1)).permute(0,2,1).reshape(6,-1,745,2048)
            
            _,_,N,C = aggregated_tokens.size()
            # for frame_index in range(f_num):
            #     frame_tokens = aggregated_tokens[:,frame_index,...].view(B,self.cam_num,N,C) # 1 6 745 2048
                
            #     frame_depth_tokens = frame_tokens[:,:,patch_start_idx:,...] # 1 6 740 2048
            #     frame_pose_tokens = frame_tokens[:,:,0,:].unsqueeze(2) # 1 6 1 2048
            #     frame_relative_tokens = relative_pose_enc.unsqueeze(2) # 1 6 1 2048
            #     last_frame_pose_enc = last_pose_enc[:,frame_index,:].reshape(B,self.cam_num,1,C) # 1 6 1 2048
            #     frame_tokens = torch.cat([frame_relative_tokens,last_frame_pose_enc+frame_pose_tokens,frame_depth_tokens],dim=2) # 1 6 742 2048
            #     atten_idx = 0
            #     for _ in range(4):
            #         frame_tokens, atten_idx = self._process_mv_attention(frame_tokens,B,6,742,2048,atten_idx,pos)
            #     agg_frame_tokens_list.append(frame_tokens.unsqueeze(2))

            frame_tokens = aggregated_tokens[:,:,...].view(B,-1,N,C) # 1 54 745 2048
            
            frame_depth_tokens = frame_tokens[:,:,patch_start_idx:,...] # 1 54 740 2048
            frame_pose_tokens = frame_tokens[:,:,0,:].unsqueeze(2) # 1 54 1 2048
            frame_relative_tokens = relative_pose_enc.unsqueeze(2).expand(B,self.cam_num,f_num,2048).reshape(B,-1,1,2048) # 1 54 1 2048
            last_frame_pose_enc = last_pose_enc[:,:,:].reshape(B,-1,1,C) # 1 54 1 2048
            frame_tokens = torch.cat([frame_relative_tokens,last_frame_pose_enc+frame_pose_tokens,frame_depth_tokens],dim=2) # 1 6 742 2048

            atten_idx = 0
            for _ in range(self.depth):
                frame_tokens, atten_idx = self._process_mv_attention(frame_tokens,B,S,742,2048,atten_idx,pos)
            
            frame_tokens = frame_tokens.reshape(B,S,-1,2048)
            
            # agg_frame_tokens_list.append(frame_tokens.unsqueeze(2))
            
            agg_frame_tokens = frame_tokens # 1 54 742 2048

            agg_layers_depths_tokens_list.append(agg_frame_tokens[...,2:,:]) # 4x B 54 740 2048

            agg_camera_frame_tokens = self.camera_tokens_agg(agg_frame_tokens[...,1,:],'frame') # 1 9 1 2048
            agg_camera_relative_tokens = self.camera_tokens_agg(agg_frame_tokens[...,0,:],'multiview',relative_pose_enc) # 1 6 1 2048

            agg_layers_camera_frame_tokens_list.append(agg_camera_frame_tokens) # 4x B 9 2048
            agg_layers_camera_relative_tokens_list.append(agg_camera_relative_tokens) # 4x B 6 2048

                 
        # norm_extrinics
        # print(others.keys())
        # aggregated_tokens_list: 24 *torchsize(1 36 3 280 518)
        predictions = {}
        with torch.cuda.amp.autocast(enabled=False):
            # if self.camera_head is not None:
                # seq_enc_list,multiview_enc_list = self.camera_head(aggregated_tokens_list)
                # predictions["pose_enc"] = {"seq":seq_enc_list[-1],"multiview":multiview_enc_list[-1]}  # pose encoding of the last iteration
                # predictions["pose_enc_list"] = {"seq":seq_enc_list,"multiview":multiview_enc_list}
                
                # pose_enc_list = self.camera_head(aggregated_tokens_list)
                # predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                # predictions["pose_enc_list"] = pose_enc_list
            if self.camera_head is not None:
                predictions["seq_enc_list"] = self.camera_head(agg_layers_camera_frame_tokens_list)
                predictions["mv_enc_list"] = self.camera_relative_head(agg_layers_camera_relative_tokens_list)
                predictions["mv_env"] = predictions["mv_enc_list"][-1]
                predictions["seq_enc"] = predictions["seq_enc_list"][-1]
                
                
                # frame_extrinsic, _ = pose_encoding_to_extri_intri(predictions["seq_enc_list"][-1], (H,W)) # 1 9 3 4
                # relative_extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["mv_enc_list"][-1], (H,W))# 1 6 3 4

                # a = torch.zeros([1,S,1,3]).to('cuda')
                # b = torch.ones([1,S,1,1]).to('cuda')
                # c = torch.cat([a,b],dim=-1) # 1 S 1 4
                # N_cam = 6
                # N_seq = S//N_cam
                # relative_pose = relative_extrinsic.unsqueeze(2).expand(1,N_cam,N_seq,3,4).reshape(1,-1,3,4)
                # relative_pose = torch.cat([relative_pose,c],dim=-2)
                # intrinsic = intrinsic.unsqueeze(2).expand(1,N_cam,N_seq,3,3).reshape(1,-1,3,3)
                # frame_pose = frame_extrinsic.unsqueeze(1).expand(1,N_cam,N_seq,3,4).reshape(1,-1,3,4)
                # frame_pose = torch.cat([frame_pose,c],dim=-2)

                # predictions["extrinsic"] = relative_pose.matmul(frame_pose)[...,:3,:]
                # predictions["intrinsic"] = intrinsic

                # extrinsics_mv, intrinsics = pose_encoding_to_extri_intri(predictions["mv_env"],(294,518))
                # extrinsics_seq, _ = pose_encoding_to_extri_intri(predictions["seq_enc"],(294,518))    
                # predictions["pose_enc"] = predictions["seq_enc"]
                
                # print(extrinsics_mv.size(),extrinsics_seq.size())
                # print(extrinsics_mv,extrinsics_seq)
                
            if self.depth_head is not None:
                # depth, depth_conf = self.depth_head(
                #     aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                # )
                depth, depth_conf = self.depth_head(
                    agg_layers_depths_tokens_list, images=images, patch_start_idx=0
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        # if not self.training:
        predictions["images"] = images  # store the images for visualization during inference
        # import sys
        # sys.path.append("/inspire/hdd/global_user/chenxinyan-240108120066/liuyanhao/vggt")
        # print("Converting pose encoding to extrinsic and intrinsic matrices...")
        # extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        # predictions["extrinsic"] = extrinsic
        # predictions["intrinsic"] = intrinsic

        # print("Processing model outputs...")
        # for key in predictions.keys():
        #     if isinstance(predictions[key], torch.Tensor):
        #         predictions[key] = predictions[key].detach().cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy

        # from demo_viser import viser_wrapper
        # viser_wrapper(predictions,use_point_map=False)
        # # viser_wrapper(pred_dict,use_point_map=False)
        # assert 1==0
        return predictions
    def _process_mv_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """

        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)
            
            
        if self.training:
            tokens = checkpoint(self.mv_blocks[global_idx],tokens, pos,0, use_reentrant=False)
        else:
            tokens = self.mv_blocks[global_idx](tokens, pos=pos,q_idx=0)

        # print(tokens.size())
        global_idx += 1

        return tokens, global_idx

class VGGT_decoder_raw(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None,others = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        # If without batch dimension, add it

        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        # aggregated_tokens_list: 24 *torchsize(1 36 3 280 518)
        predictions = {}
        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                # seq_enc_list,multiview_enc_list = self.camera_head(aggregated_tokens_list)
                # predictions["pose_enc"] = {"seq":seq_enc_list[-1],"multiview":multiview_enc_list[-1]}  # pose encoding of the last iteration
                # predictions["pose_enc_list"] = {"seq":seq_enc_list,"multiview":multiview_enc_list}
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        # if not self.training:
        predictions["images"] = images  # store the images for visualization during inference
        # import sys
        # sys.path.append("/inspire/hdd/global_user/chenxinyan-240108120066/liuyanhao/vggt")
        # print("Converting pose encoding to extrinsic and intrinsic matrices...")
        # extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        # predictions["extrinsic"] = extrinsic
        # predictions["intrinsic"] = intrinsic

        # print("Processing model outputs...")
        # for key in predictions.keys():
        #     if isinstance(predictions[key], torch.Tensor):
        #         predictions[key] = predictions[key].detach().cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy

        # from demo_viser import viser_wrapper
        # viser_wrapper(predictions,use_point_map=False)
        # # viser_wrapper(pred_dict,use_point_map=False)
        # assert 1==0
        return predictions
