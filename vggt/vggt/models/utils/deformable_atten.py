import torch.nn as nn
import torch


class Deformable_atten(nn.Module):
    def __init__(self):
        super().__init__()
        self.selected_extra_intra = 3
        self.selected_points_num = 2
        self.selected_values_proj = nn.Linear(2048, self.selected_points_num * 6) # 2048->12
    
    # def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    #     # for debug and test only,
    #     # need to use cuda version instead
    #     N_, S_, M_, D_ = value.shape
    #     _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    #     value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    #     sampling_grids = 2 * sampling_locations - 1
    #     sampling_value_list = []
    #     for lid_, (H_, W_) in enumerate(value_spatial_shapes):
    #         # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
    #         value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
    #         # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
    #         sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
    #         # N_*M_, D_, Lq_, P_
    #         sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
    #                                         mode='bilinear', padding_mode='zeros', align_corners=False)
    #         sampling_value_list.append(sampling_value_l_)
    #     # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    #     attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    #     output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    #     return output.transpose(1, 2).contiguous()
    
    # def find_values(self,whole_tokens,intra_indices,inter_indices):
    #     whole_tokens = whole_tokens.permute(0,3,4,1,2).reshape(-1,2048,6,9)
    #     intra_indices = intra_indices.
    #     sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
    #     # N, Len_q, n_heads, n_levels, n_points, 2

    #     return selected_intra_values,selected_inter_values
    
    def forward(self,tokens):
        
        B, N_cam, N_f, P,C = tokens.size() # 1 6 9 742 2048
        print(tokens.size())
        assert 1==0
        # 1. get which tokens to atten
        # 1.1 selected_frame_index
        selected_tokens_fea = torch.mean(tokens,dim=-2) # 1 6 9 2048
        
        delta_value_indices_w = self.selected_values_proj(selected_tokens_fea) # 1 6 9 12 (0-1) delta pose
        delta_value_indices_w = delta_value_indices_w.view(B,N_cam,N_cam,self.selected_points_num,1) # 1 6 9 6 2 1
        delta_value_indices_h = torch.zeros_like(delta_value_indices_w).view(B,N_cam,N_cam,self.selected_points_num,1) # 1 6 9 6 2 1
        delta_value_indices = torch.cat([delta_value_indices_h,delta_value_indices_w],dim=-1)# 1 6 9 6 2 2
        
        value_index_h = torch.linespace(0.5,N_cam-0.5,N_cam)
        value_index_w = torch.linespace(0.5,N_f-0.5,N_f)
        delta_value_indices = 
        initial_value_indices_h = torch.ones_like(delta_value_indices_h) * frame_num / N_frame + 0.5 # 1 6 9 6 2 1        
        value_indices = frame_initial_value_indices + frame_delta_value_indices # 1 6 6 2 1
        
        multiview_value_indices = torch.linespace(0.5,N_cam-0.5,N_cam).to('cuda').reshape(1,1,N_cam,1,1).expand(B,N_cam,N_cam,self.selected_points_num,1)
        value_indices = torch.cat([multiview_value_indices,frame_value_indices],dim=-1)
        print(value_indices)
        assert 1==0
        # selected_values,selected_inter_values = self.find_values(whole_tokens,intra_indices,inter_indices) # 1 6 3+5*1 742 2048

        
        