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
from vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri
from vggt.models.sparseatten import Sparse_attention

class MultiViewVGGT_v2(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True):
        super().__init__()
        self.cam_list = ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,depth=4)
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None
    
    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None,others = None):
        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        del others["images"]
        # images information transform to dimension with [seq * cam_pose] 
        B, S, C, H, W = images.size()
        len_cam_pos = len(self.cam_list)
        seq_len = S//len_cam_pos

        seq_images = images.view(B,-1, C,H,W).contiguous() # B 5 6 CHW

        aggregated_tokens_list, patch_start_idx = self.aggregator(seq_images,others)
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
