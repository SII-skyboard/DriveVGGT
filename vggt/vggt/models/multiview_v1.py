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
from vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri
from vggt.models.sparseatten import Sparse_attention

class MultiViewVGGT_v1(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None
        
        self.cam_list = ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']
        self.multiview_atten = Sparse_attention(depth=1)
        self.position_encoder = nn.Linear(9, embed_dim*2)
        
    def process_relative_pose(self, others):
        intrinsics = others['intrinsics']
        extrinsics = others['extrinsics']

        relative_extrinics = extrinsics[:,0::5,...] - extrinsics[:,0,...]
        relative_intrinsics = intrinsics[:,0::5,...] - intrinsics[:,0,...]     #  B 6 H W
 
        return [relative_intrinsics,relative_extrinics]
    
    def similarity(self,cam_tokens):
        # self.similarity(multiview_aggregated_tokens_pose) # 24*1 30 1024 cam pose token
        L = 5
        K = 8
        # mask single view token
        similarity_matrix = cam_tokens@cam_tokens.transpose(1,2)
        for index in range(len(self.cam_list)):
            chunk_index_start = L*index
            similarity_matrix[:,chunk_index_start:chunk_index_start+L,chunk_index_start:chunk_index_start+L] += 10000
        
        similarity_matrix = torch.softmax(similarity_matrix,dim=-1) # 24*1 30 30
        values,indices = torch.topk(similarity_matrix, K, dim=-1)

        return values, indices
    
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
        seq_len = S//len_cam_pos
        
        relative_pose = self.process_relative_pose(others)
        relative_pose = extri_intri_to_pose_encoding(relative_pose[1],relative_pose[0],(H,W))
        B,P,C_pos = relative_pose.size()
        relative_pose = relative_pose.view(B*P,C_pos)
        relative_pose = self.position_encoder(relative_pose)

        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        seq_images = others["images"].view(B,-1,len(self.cam_list), C,H,W).contiguous() # B 5 6 CHW

        multiview_aggregated_tokens_list = []
        for i in range(len(self.cam_list)):
            seq_images_cam = seq_images[:,:,i,...] # B seq cam-th CHW
            aggregated_tokens_list, patch_start_idx = self.aggregator(seq_images_cam)
            aggregated_tokens = torch.stack(aggregated_tokens_list,dim=0)
            multiview_aggregated_tokens_list.append(aggregated_tokens)

        multiview_aggregated_tokens = torch.stack(multiview_aggregated_tokens_list,dim=3)
        L,B,S,P,N,C = multiview_aggregated_tokens.size() # 24 1 5 6 750 1024
        relative_pose = relative_pose.view(B,P,-1).repeat(L,1,seq_len,1,1).view(L,B,S,P,-1) # B 6 1024 -> B 30 1024
        multiview_aggregated_tokens[:,:,:,:,0,:] = multiview_aggregated_tokens[:,:,:,:,0,:] + relative_pose
        multiview_aggregated_tokens = multiview_aggregated_tokens.view(L*B,S*P,N,C).contiguous() # 24*1 30 750 1024
        multiview_aggregated_tokens_pose = multiview_aggregated_tokens[:,:,0,:] # 24*1 30 1024 cam pose token
        values, indices = self.similarity(multiview_aggregated_tokens_pose) # 24*1 30 30 (0-1)
        multiview_aggregated_tokens = self.multiview_atten(multiview_aggregated_tokens,values, indices,idx=0)
        aggregated_tokens_list = torch.chunk(multiview_aggregated_tokens,L,dim=0)
        
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
