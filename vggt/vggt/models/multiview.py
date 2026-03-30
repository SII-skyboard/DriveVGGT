# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead
from vggt.models.aggregator import Aggregator
from vggt.utils.pose_enc import extri_intri_to_pose_encoding

class MultiViewVGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None
        
        self.cam_list = ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']
        self.position_encoder = nn.Linear(9, embed_dim*2)
        
    def process_image_seq(self, x):
        # dict_keys(['seq_name', 'ids', 'images', 'depths', 'extrinsics', 'intrinsics', 'cam_points', 'world_points', 'point_masks'])
        # print(x['seq_name'])
        # print(x['ids'])
        len_pos = 6
        B,S,C,H,W = x['images'].size()
        x['images'] = x['images'].view(B,len_pos,-1,C,H,W).permute(0,2,1,3,4,5).contiguous().view(B,-1,C,H,W)
        x['depths'] = x['depths'].view(B,len_pos,-1,H,W).permute(0,2,1,3,4).contiguous().view(B,-1,H,W)
        x['extrinsics'] = x['extrinsics'].view(B,len_pos,-1,3,4).permute(0,2,1,3,4).contiguous().view(B,-1,3,4)
        x['intrinsics'] = x['intrinsics'].view(B,len_pos,-1,3,3).permute(0,2,1,3,4).contiguous().view(B,-1,3,3)
        x['cam_points'] = x['cam_points'].view(B,len_pos,-1,H,W,C).permute(0,2,1,3,4,5).contiguous().view(B,-1,H,W,C)
        x['world_points'] = x['world_points'].view(B,len_pos,-1,H,W,C).permute(0,2,1,3,4,5).contiguous().view(B,-1,H,W,C)
        x['point_masks'] = x['point_masks'].view(B,len_pos,-1,H,W).permute(0,2,1,3,4).contiguous().view(B,-1,H,W)
        
        return x
    def process_image_spatial(self, aggreagted_cam_token_seq_dict):
        aggregated_camera_list = []
        patch_index = aggreagted_cam_token_seq_dict['CAM_FRONT'][1]
        for layer in range(24):
            single_layer_token_list = []
            for cam_pos in self.cam_list:
                B,C,P,F = aggreagted_cam_token_seq_dict[cam_pos][0][layer].size()
                single_cam_token = aggreagted_cam_token_seq_dict[cam_pos][0][layer]
                single_layer_token_list.append(single_cam_token.unsqueeze(1))
            single_layer_token = torch.cat(single_layer_token_list,dim=1).view(B,-1,P,F)
            aggregated_camera_list.append(single_layer_token)
        
        return aggregated_camera_list,patch_index
            
        
        
    def process_relative_pose(self, others):
        intrinsics = others['intrinsics']
        extrinsics = others['extrinsics']

        relative_extrinics = extrinsics[:,:6,...] - extrinsics[:,0,...]
        relative_intrinsics = intrinsics[:,:6,...] - intrinsics[:,0,...]     #  B 6 H W
        return [relative_intrinsics,relative_extrinics]
        
    
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

        # images information transform to dimension with [seq * cam_pose]
        B, S, C, H, W = images.size()
        len_cam_pos = len(self.cam_list)
        others = self.process_image_seq(others)
        relative_pose = self.process_relative_pose(others)
        relative_pose = extri_intri_to_pose_encoding(relative_pose[1],relative_pose[0],(H,W))
        B,P,C_pos = relative_pose.size()
        relative_pose = relative_pose.view(B*P,C_pos)
        relative_pose = self.position_encoder(relative_pose)
        relative_pose = relative_pose.view(B,P,-1)

        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        seq_images = others["images"].view(B,-1,len(self.cam_list), C,H,W)
        aggreagted_cam_token_seq_dict = {
            'CAM_FRONT':None,
            'CAM_FRONT_LEFT':None,
            'CAM_FRONT_RIGHT':None,
            'CAM_BACK':None,
            'CAM_BACK_LEFT':None,
            'CAM_BACK_RIGHT':None
        }

        for i in range(len(self.cam_list)):
            seq_images_cam = seq_images[:,:,i,...] # B seq cam-th CHW
            aggregated_tokens_list, patch_start_idx = self.aggregator(seq_images_cam)
            # aggreagted_cam_token_seq_dict[][0] 
            for seq in range(len(aggregated_tokens_list)):
                aggregated_tokens_list[seq][:,:,0,:] = aggregated_tokens_list[seq][:,:,0,:] + relative_pose[:,i,:]
                # token[:, :, 0] + relative_pose[:,i,:] # B S  /B 1024
            aggreagted_cam_token_seq_dict[self.cam_list[i]] = [aggregated_tokens_list,patch_start_idx]     
            # print(aggregated_tokens_list[0].size())       # 1 6(seq) 745 2048
        
        aggregated_tokens_list,patch_start_idx = self.process_image_spatial(aggreagted_cam_token_seq_dict)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
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

        return predictions
