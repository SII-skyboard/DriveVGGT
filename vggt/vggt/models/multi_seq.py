# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator_low_resolution import Aggregator_low_resolution
from vggt.models.aggregator import Aggregator,MultiSeqTransformer

from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead,DPTHead_m
from vggt.heads.track_head import TrackHead
from vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri
import torch.nn.functional as F

class MultiSeq_Geo(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True):
        super().__init__()

        self.aggregator = Aggregator_low_resolution(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1",intermediate_layer_idx=[0,1,2,3]) if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1",intermediate_layer_idx=[0,1,2,3]) if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None
        # self.dim_change = nn.Linear(2 * embed_dim,embed_dim)
        # self.multiseq_transformer = MultiSeqTransformer(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,depth=4)

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None,others = None):
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        B,S,C_in,H,W = images.size()
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)
        # 1. geo fea initialization
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        # tokens selections

        tokens_list = []
        for layer in [4, 11, 17, 23]:
            # 1, 54, 45, 2048
            cam_pose_tokens = aggregated_tokens_list[layer][:,:,0:5,:].view(B*S,5,-1)
            geo_tokens = aggregated_tokens_list[layer][:,:,5:].view(B*S,5,8,2048).contiguous() # B*S  1024
            geo_tokens = F.interpolate(geo_tokens.permute(0,3,1,2), size=(20,37), mode='bilinear')
            geo_tokens = geo_tokens.permute(0,2,3,1).view(B*S,-1,2048)
            tokens = torch.cat((cam_pose_tokens,geo_tokens),dim=1).view(B,S,-1,2048)                   
            # tokens = self.dim_change(tokens)
            tokens_list.append(tokens)
        # token aggregation
        # B,S,P,C = cam_pose_tokens.size() #[1, 54, 45, 2048]
        # aggregated_tokens_list, patch_start_idx  = self.multiseq_transformer(tokens,[B,S,C_in,H,W])
        aggregated_tokens_list = tokens_list
        
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
