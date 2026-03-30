import os
import torch
import numpy as np
import gzip
import json
import random
import logging
import warnings
from vggt.models.vggt import VGGT
from vggt.models.multiview_v2 import MultiViewVGGT_v2
from vggt.models.vggt_low_resolution import VGGT_low_resolution
from vggt.models.multi_seq import MultiSeq_Geo
from vggt.models.vggt_low_layer import VGGT_low_layer
from vggt.models.vggt_decoder import VGGT_decoder
from vggt.models.vggt_decoder_global import VGGT_decoder_global
from vggt.models.vggt import VGGT_rel
from vggt.utils.rotation import mat_to_quat
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import closed_form_inverse_se3
import time
# from ba import run_vggt_with_ba
import argparse
import cv2 
from vggt.utils.load_fn import load_and_preprocess_images,load_and_preprocess_images_pos


val = \
    ['scene-0003', 'scene-0012', 'scene-0013', 'scene-0014', 'scene-0015', 'scene-0016', 'scene-0017', 'scene-0018',
     'scene-0035', 'scene-0036', 'scene-0038', 'scene-0039', 'scene-0092', 'scene-0093', 'scene-0094', 'scene-0095',
     'scene-0096', 'scene-0097', 'scene-0098', 'scene-0099', 'scene-0100', 'scene-0101', 'scene-0102', 'scene-0103',
     'scene-0104', 'scene-0105', 'scene-0106', 'scene-0107', 'scene-0108', 'scene-0109', 'scene-0110', 'scene-0221',
     'scene-0268', 'scene-0269', 'scene-0270', 'scene-0271', 'scene-0272', 'scene-0273', 'scene-0274', 'scene-0275',
     'scene-0276', 'scene-0277', 'scene-0278', 'scene-0329', 'scene-0330', 'scene-0331', 'scene-0332', 'scene-0344',
     'scene-0345', 'scene-0346', 'scene-0519', 'scene-0520', 'scene-0521', 'scene-0522', 'scene-0523', 'scene-0524',
     'scene-0552', 'scene-0553', 'scene-0554', 'scene-0555', 'scene-0556', 'scene-0557', 'scene-0558', 'scene-0559',
     'scene-0560', 'scene-0561', 'scene-0562', 'scene-0563', 'scene-0564', 'scene-0565', 'scene-0625', 'scene-0626',
     'scene-0627', 'scene-0629', 'scene-0630', 'scene-0632', 'scene-0633', 'scene-0634', 'scene-0635', 'scene-0636',
     'scene-0637', 'scene-0638', 'scene-0770', 'scene-0771', 'scene-0775', 'scene-0777', 'scene-0778', 'scene-0780',
     'scene-0781', 'scene-0782', 'scene-0783', 'scene-0784', 'scene-0794', 'scene-0795', 'scene-0796', 'scene-0797',
     'scene-0798', 'scene-0799', 'scene-0800', 'scene-0802', 'scene-0904', 'scene-0905', 'scene-0906', 'scene-0907',
     'scene-0908', 'scene-0909', 'scene-0910', 'scene-0911', 'scene-0912', 'scene-0913', 'scene-0914', 'scene-0915',
     'scene-0916', 'scene-0917', 'scene-0919', 'scene-0920', 'scene-0921', 'scene-0922', 'scene-0923', 'scene-0924',
     'scene-0925', 'scene-0926', 'scene-0927', 'scene-0928', 'scene-0929', 'scene-0930', 'scene-0931', 'scene-0962',
     'scene-0963', 'scene-0966', 'scene-0967', 'scene-0968', 'scene-0969', 'scene-0971', 'scene-0972', 'scene-1059',
     'scene-1060', 'scene-1061', 'scene-1062', 'scene-1063', 'scene-1064', 'scene-1065', 'scene-1066', 'scene-1067',
     'scene-1068', 'scene-1069', 'scene-1070', 'scene-1071', 'scene-1072', 'scene-1073']

def align_pred_to_gt(
    pred_depth,
    gt_depth,
    valid_mask,
    min_valid_pixels = 100,
) -> tuple[float, float, np.ndarray]:
    """
    Aligns a predicted depth map to a ground truth depth map using scale and shift.
    The alignment is: gt_aligned_to_pred ≈ scale * pred_depth + shift.

    Args:
        pred_depth (np.ndarray): The HxW predicted depth map.
        gt_depth (np.ndarray): The HxW ground truth depth map.
        min_gt_depth (float): Minimum valid depth value for GT.
        max_gt_depth (float): Maximum valid depth value for GT.
        min_pred_depth (float): Minimum valid depth value for predictions.
        min_valid_pixels (int): Minimum number of valid overlapping pixels
                                 required to perform the alignment.
        robust_median_scale (bool): If True, uses median(gt/pred) for scale and then
                                    median(gt - scale*pred) for shift. Otherwise, uses
                                    least squares for both scale and shift simultaneously.

    Returns:
        tuple[float, float, np.ndarray]:
            - scale (float): The calculated scale factor. (NaN if alignment failed)
            - shift (float): The calculated shift offset. (NaN if alignment failed)
            - aligned_pred_depth (np.ndarray): The HxW predicted depth map after
                                               applying scale and shift. (Original pred_depth
                                               if alignment failed)
    """
    if pred_depth.shape != gt_depth.shape:
        raise ValueError(
            f"Predicted depth shape {pred_depth.shape} must match GT depth shape {gt_depth.shape}"
        )

    # Extract valid depth values
    gt_masked = gt_depth.reshape(-1)[valid_mask.reshape(-1)]
    pred_masked = pred_depth.reshape(-1)[valid_mask.reshape(-1)]
    if len(gt_masked) < min_valid_pixels:
        print(
            f"Warning: Not enough valid pixels ({len(gt_masked)} < {min_valid_pixels}) to align. "
            "Using all pixels."
        )
        gt_masked = gt_depth.reshape(-1)
        pred_masked = pred_depth.reshape(-1)

    # Handle case where pred_masked has no variance (e.g., all zeros or a constant value)
    if np.std(pred_masked) < 1e-6: # Small epsilon to check for near-constant values
        print(
            "Warning: Predicted depth values in the valid mask have near-zero variance. "
            "Scale is ill-defined. Setting scale=1 and solving for shift only."
        )
        scale = 1.0
        shift = np.mean(gt_masked) - np.mean(pred_masked) # or np.median(gt_masked) - np.median(pred_masked)
    else:
        A = np.vstack([pred_masked, np.ones_like(pred_masked)]).T
        try:
            x, residuals, rank, s_values = np.linalg.lstsq(A, gt_masked, rcond=None)
            scale, shift = x[0], x[1]
        except np.linalg.LinAlgError as e:
            print(f"Warning: Least squares alignment failed ({e}). Returning original prediction.")
            return np.nan, np.nan, pred_depth.copy()


    aligned_pred_depth = scale * pred_depth + shift
    return scale, shift, aligned_pred_depth



def build_pair_index(N, B=1):
    """
    Build indices for all possible pairs of frames.

    Args:
        N: Number of frames
        B: Batch size

    Returns:
        i1, i2: Indices for all possible pairs
    """
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]
    return i1, i2

def rotation_angle(rot_gt, rot_pred, batch_size=None, eps=1e-15):
    """
    Calculate rotation angle error between ground truth and predicted rotations.

    Args:
        rot_gt: Ground truth rotation matrices
        rot_pred: Predicted rotation matrices
        batch_size: Batch size for reshaping the result
        eps: Small value to avoid numerical issues

    Returns:
        Rotation angle error in degrees
    """
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)

    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)

    rel_rangle_deg = err_q * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg

def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """
    Normalize the translation vectors and compute the angle between them.

    Args:
        t_gt: Ground truth translation vectors
        t: Predicted translation vectors
        eps: Small value to avoid division by zero
        default_err: Default error value for invalid cases

    Returns:
        Angular error between translation vectors in radians
    """
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t

def translation_angle(tvec_gt, tvec_pred, batch_size=None, ambiguity=True):
    """
    Calculate translation angle error between ground truth and predicted translations.

    Args:
        tvec_gt: Ground truth translation vectors
        tvec_pred: Predicted translation vectors
        batch_size: Batch size for reshaping the result
        ambiguity: Whether to handle direction ambiguity

    Returns:
        Translation angle error in degrees
    """
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg

def se3_to_relative_pose_error(pred_se3, gt_se3, num_frames):
    """
    Compute rotation and translation errors between predicted and ground truth poses.
    This function assumes the input poses are world-to-camera (w2c) transformations.

    Args:
        pred_se3: Predicted SE(3) transformations (w2c), shape (N, 4, 4)
        gt_se3: Ground truth SE(3) transformations (w2c), shape (N, 4, 4)
        num_frames: Number of frames (N)

    Returns:
        Rotation and translation angle errors in degrees
    """
    pair_idx_i1, pair_idx_i2 = build_pair_index(num_frames)

    relative_pose_gt = gt_se3[pair_idx_i1].bmm(
        closed_form_inverse_se3(gt_se3[pair_idx_i2])
    )
    relative_pose_pred = pred_se3[pair_idx_i1].bmm(
        closed_form_inverse_se3(pred_se3[pair_idx_i2])
    )

    rel_rangle_deg = rotation_angle(
        relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3]
    )
    rel_tangle_deg = translation_angle(
        relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3]
    )

    return rel_rangle_deg, rel_tangle_deg

def setup_args():
    """Set up command-line arguments for the CO3D evaluation script."""
    parser = argparse.ArgumentParser(description='Test VGGT on CO3D dataset')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (only test on specific scene)')
    parser.add_argument('--use_ba', action='store_true', default=False, help='Enable bundle adjustment')
    parser.add_argument('--fast_eval', action='store_true', default=False, help='Only evaluate 10 sequences per scene')
    parser.add_argument('--min_num_images', type=int, default=50, help='Minimum number of images for a sequence')
    parser.add_argument('--num_frames', type=int, default=10, help='Number of frames to use for testing')
    parser.add_argument('--co3d_dir', type=str, required=False, help='Path to CO3D dataset')
    parser.add_argument('--co3d_anno_dir', type=str, required=False, help='Path to CO3D annotations')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--model_path', type=str, required=False, help='Path to the VGGT model checkpoint')
    parser.add_argument('--model_name', type=str, required=False, help='Path to the VGGT model checkpoint')
    
    return parser.parse_args()

def load_model(device, model_path,debug=False,raw_vggt=False,vggt_raw_para=False,name=None):
    """
    Load the VGGT model.

    Args:
        device: Device to load the model on
        model_path: Path to the model checkpoint

    Returns:
        Loaded VGGT model
    """

    if name is None:
        if raw_vggt is False:
            model = MultiViewVGGT_v2(enable_track=False,enable_point=False,enable_depth=True)
        else:
            if vggt_raw_para is True:
                model = VGGT()
            else:
                print("Initializing and loading VGGT model...")
                model = VGGT(enable_track=False,enable_point=False,enable_depth=True)
    elif name == "low_resolution":
        print("low_resolution")
        model = VGGT_low_resolution(enable_track=False,enable_point=False,enable_depth=True)
    elif name == "decoder":
        print("decoder")
        model = VGGT_decoder(enable_track=False,enable_point=False,enable_depth=True)
    elif name == "decoder_global":
        print("decoder_global")
        model = VGGT_decoder_global(enable_track=False,enable_point=False,enable_depth=True)
    elif name == "multiseq":
        print("multiseq")
        from vggt.models.vggt_low_resolution import VGGT_low_resolution_trans_enc,VGGT_low_resolution_seq
        model = VGGT_low_resolution_seq(enable_track=False,enable_point=False,enable_depth=True)
    elif name == "vggtrel":
        print("vggtrel")
        from vggt.models.vggt import VGGT_rel
        model = VGGT_rel(enable_track=False,enable_point=False,enable_depth=True)
    elif name == "fastdecoder":
        print("fastdecoder")
        from vggt.models.vggt_decoder import VGGT_fastdecoder
        model = VGGT_fastdecoder(enable_track=False,enable_point=False,enable_depth=True)
    elif name == "fastvggt":
        print("fastvggt")
        from vggt.models.vggtfast.vggt.models.vggt import Fast_VGGT
        model = Fast_VGGT(enable_track=False,enable_point=False,enable_depth=True)
    elif name == "streamvggt":
        print("streamvggt")
        from vggt.models.streamvggt.streamvggt.models.streamvggt import StreamVGGT
        model = StreamVGGT(enable_track=False,enable_point=False,enable_depth=True)
    elif name == "fastrel":
        print("fastrel")
        from vggt.models.vggtfast.vggt.models.vggt import Fast_VGGT_rel
        model = Fast_VGGT_rel(enable_track=False,enable_point=False,enable_depth=True)
    elif name == "a":
        from vggt.models.vggt_decoder_a import VGGT_decoder_a
        print("decoder_a")
        model = VGGT_decoder_a(enable_track=False,enable_point=False,enable_depth=True)
    # _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    # model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    
    if debug is True:
        model.eval()
        model = model.to(device)
    else:
        print(f"USING {model_path}")
        if raw_vggt is True:         
            if vggt_raw_para is True:
                model.load_state_dict(torch.load(model_path)) #for trained model add key "model" else None 
            else:
                model.load_state_dict(torch.load(model_path)["model"], False) #for trained model add key "model" else None 
        else:
            model.load_state_dict(torch.load(model_path)["model"]) #for trained model add key "model" else None 
        model.eval()
        model = model.to(device)
    return model

def set_random_seeds(seed):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class NuscenesDatasetEval():
    def __init__(
        self,
        nusc_info: str = "/inspire/hdd/global_user/chenxinyan-240108120066/liuyanhao/data_nuscenes/nuscene_json/",
        min_num_images: int = 24, 
    ):
        """
        Initialize the VKittiDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            VKitti_DIR (str): Directory path to VKitti data.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
            expand_range (int): Range for expanding nearby image selection.
            get_nearby_thres (int): Threshold for nearby image selection.
        """
        super().__init__()
        
        self.min_num_images = min_num_images
        self.cam_list = ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']
        
        self.nusc_info = nusc_info
        
        self.sequence_list = os.listdir(nusc_info)
        # self.sequence_list.sort()
        self.sequence_list_len = len(self.sequence_list)
        self.depth_max = 80
        
    def lidar2depthmap(
        self,
        pts_lidar,
        img,
        lidar2img,
        depth_max = 80
        ):
        # image size

        height, width,c = img.shape
        # point2image
        point_img = (lidar2img[:3,:3] @ pts_lidar[:,:3].T).T + lidar2img[:3,3]
        points = np.concatenate(
            [point_img[:, :2] / point_img[:, 2:3],
            point_img[:, 2:3]], 1).astype(np.float32)
        depth_map = np.zeros((height, width), dtype=np.float32)
        coor = np.round(points[:, :2])
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & \
                (coor[:, 1] >= 0) & (coor[:, 1] < height) & \
                (depth < depth_max) & (depth > 0)
        coor, depth = coor[kept1], depth[kept1]
        
        coor = coor.astype(np.int32)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        eps = 10e-6
        valid_mask = np.where(depth_map > eps,True,False)
        return depth_map,valid_mask

    def get_data(
        self,
        scene_name: str = "scene-0001",
        img_per_seq: int = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
        image_stat: int = 10,
        seq_step: int = 5
    ):
        """
        Retrieve data for a specific sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve.
            img_per_seq (int): Number of images per sequence.
            seq_name (str): Name of the sequence.
            ids (list): Specific IDs to retrieve.
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """
        
        # seq_index = random.randint(0, (self.sequence_list_len - 1))
        

        json_path = os.path.join(self.nusc_info, scene_name+".json")
        with open(json_path, "r") as f:
            scene1 = json.load(f)
        
        images = []
        images_path = []
        extris = []
        intris = []
        depths = []
        valid_masks = []
        image_len = seq_step
        camera_list = range(image_stat,image_stat+image_len)
                    
        for cam_pos in self.cam_list:
            for time_step in camera_list:
                lidar_path = scene1[scene_name][time_step]['lidar']['lidar_path']
        
                img_path = scene1[scene_name][time_step]['camera'][cam_pos]['image_path']
                K = np.array(scene1[scene_name][time_step]['camera'][cam_pos]['intrinsics'])
                cam2global = np.array(scene1[scene_name][time_step]['camera'][cam_pos]['cam2global'])
                lidar2cam = np.linalg.inv(cam2global)  # lidar -> cam As lidar is in the axis of global
                cam2img = np.eye(4)
                cam2img[:3,:3] = K
                lidar2img = cam2img @ lidar2cam 

                pts_lidar = np.load(lidar_path).reshape(-1,3)
                RGB_image = cv2.imread(img_path)
                depth_map,valid_mask = self.lidar2depthmap(pts_lidar,RGB_image,lidar2img)
                
                extri_opencv = cam2global[:3]
                # debug
                extri_opencv = np.linalg.inv(cam2global)
                extri_opencv = extri_opencv[:3]
                intri_opencv = K
                images.append(RGB_image)
                images_path.append(img_path)
                extris.append(extri_opencv)
                intris.append(intri_opencv)
                depths.append(depth_map)
                valid_masks.append(valid_mask)
                        
        return [images_path,extris,intris,depths,valid_masks]

def process_sequence(model, nusc_data, use_ba, device, dtype,seq_name,stat,depth_step):
    """
    Process a single sequence and compute pose errors.

    Args:
        model: VGGT model
        seq_name: Sequence name
        seq_data: Sequence data
        scene: scene name
        co3d_dir: CO3D dataset directory
        min_num_images: Minimum number of images required
        num_frames: Number of frames to sample
        use_ba: Whether to use bundle adjustment
        device: Device to run on
        dtype: Data type for model inference

    Returns:
        rError: Rotation errors
        tError: Translation errors
    """

    # 1. get nusc img path + ex
    
    nusc_data,image_names = nusc_data
    

    ids = len(image_names)
    print("Image ids", ids)

    # #2. images process no needed
    # images = load_and_preprocess_images_pos(image_names).to(device)

    if use_ba:
        pass

    else:
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                images = nusc_data["images"]
                
                start = time.time()          # CPU记录开始时间
                predictions = model(nusc_data["images"],others = nusc_data)
                torch.cuda.synchronize()
                end = time.time()            # CPU记录结束时间
                t_model = (end-start)*1000
                print("inference_times_is:  ", t_model,'ms')
                
        if ("mv_enc_list" in predictions.keys()) is False:
            with torch.cuda.amp.autocast(dtype=torch.float64):
                extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        else:
            S,_,H,W = nusc_data["images"].size()
            # nusc_input norm
            N_cam = 6
            N_seq = S//N_cam
            frame_extrinsic, _ = pose_encoding_to_extri_intri(predictions["seq_enc_list"][-1], nusc_data["images"].shape[-2:]) # 1 9 3 4
            relative_extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["mv_enc_list"][-1], nusc_data["images"].shape[-2:])# 1 6 3 4
            a = torch.zeros([1,S,1,3]).to('cuda')
            b = torch.ones([1,S,1,1]).to('cuda')
            c = torch.cat([a,b],dim=-1) # 1 S 1 4
            relative_pose = relative_extrinsic.unsqueeze(2).expand(1,N_cam,N_seq,3,4).reshape(1,-1,3,4)
            relative_pose = torch.cat([relative_pose,c],dim=-2)
            intrinsic = intrinsic.unsqueeze(2).expand(1,N_cam,N_seq,3,3).reshape(1,-1,3,3)
            frame_pose = frame_extrinsic.unsqueeze(1).expand(1,N_cam,N_seq,3,4).reshape(1,-1,3,4)
            frame_pose = torch.cat([frame_pose,c],dim=-2)

            extrinsic = relative_pose.matmul(frame_pose)[...,:3,:]
            print("predict",extrinsic.size())

        with torch.cuda.amp.autocast(dtype=torch.float64):
            pred_extrinsic = extrinsic[0]

    with torch.cuda.amp.autocast(dtype=torch.float64):
        gt_extrinsic = nusc_data["extrinsics"][0].to(device)
        add_row = torch.tensor([0, 0, 0, 1], device=device).expand(pred_extrinsic.size(0), 1, 4)
        pred_se3 = torch.cat((pred_extrinsic, add_row), dim=1)
        gt_se3 = torch.cat((gt_extrinsic, add_row), dim=1)

        rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(pred_se3, gt_se3, num_frames=ids)


        Racc_5 = (rel_rangle_deg < 5).float().mean().item()
        Tacc_5 = (rel_tangle_deg < 5).float().mean().item()
        
        B,_,H,W = nusc_data["depths"].size()
        depths_gt = nusc_data["depths"].view(B,H,W).cpu().numpy()
        valid_masks = nusc_data["valid_masks"].view(B,H,W).cpu().numpy()
        depths_predict = predictions["depth"].view(B,H,W).cpu().numpy()
        
        i = 0
        abs_rel = 0
        mae = 0
        sq_rel = 0
        delta = 0
        for i in range(depths_predict.shape[0]):
            _, _, aligned_depth_predict = align_pred_to_gt(depths_predict[i],depths_gt[i],valid_masks[i])
            depth_error = aligned_depth_predict[valid_masks[i]]-depths_gt[i][valid_masks[i]]
            norm_error = depth_error/depths_gt[i][valid_masks[i]]
            
            abs_rel += np.mean(np.abs(norm_error))
            mae += np.mean(np.abs(depth_error))
            sq_rel += np.mean(norm_error*norm_error*depths_gt[i][valid_masks[i]])
            
            depth_compare1 = aligned_depth_predict[valid_masks[i]]/depths_gt[i][valid_masks[i]]
            depth_compare2 = depths_gt[i][valid_masks[i]]/aligned_depth_predict[valid_masks[i]]
            depth_compare = np.where(depth_compare1>depth_compare2,depth_compare1,depth_compare2)
            
            delta += depth_compare[depth_compare<1.25*1.25*1.25].size/depth_compare.size
        
        abs_rel /= depths_predict.shape[0]
        mae /= depths_predict.shape[0]
        sq_rel /= depths_predict.shape[0]
        delta /= depths_predict.shape[0]
            
        print(f"{seq_name} sequence {stat} R_ACC@5: {Racc_5:.4f}")
        print(f"{seq_name} sequence {stat} T_ACC@5: {Tacc_5:.4f}")
        
        return rel_rangle_deg.cpu().numpy(), rel_tangle_deg.cpu().numpy(),t_model,[abs_rel,mae,sq_rel,delta]

def calculate_auc_np(r_error, t_error, max_threshold=30):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays using NumPy.

    Args:
        r_error: numpy array representing R error values (Degree)
        t_error: numpy array representing T error values (Degree)
        max_threshold: Maximum threshold value for binning the histogram

    Returns:
        AUC value and the normalized histogram
    """
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs
    return np.mean(np.cumsum(normalized_histogram)), normalized_histogram

def main():
    """Main function to evaluate VGGT on CO3D dataset."""
    # Parse command-line arguments
    args = setup_args()

    # Setup device and data type
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Load model
    # args.model_path = "/inspire/hdd/global_user/chenxinyan-240108120066/liuyanhao/vggt/training/logs/exp001/ckpts/checkpoint_50.pt"
    if args.model_path == None:
        # args.model_path = "/inspire/hdd/global_user/chenxinyan-240108120066/liuyanhao/vggt/training/logs/exp001/ckpts/checkpoint_30.pt"
        args.model_path = "/inspire/hdd/global_user/chenxinyan-240108120066/liuyanhao/vggt/ckpt/model.pt"
        # args.model_path = "/inspire/hdd/global_user/chenxinyan-240108120066/liuyanhao/vggt/ckpt/vggt_ft_c_d/checkpoint_15.pt"
        # args.model_path = "/inspire/hdd/global_user/chenxinyan-240108120066/liuyanhao/vggt/ckpt/model.pt"
    model = load_model(device, model_path=args.model_path,debug=False,raw_vggt=True,vggt_raw_para=False,name=args.model_name)

    # Set random seeds
    set_random_seeds(args.seed)

    # Categories to evaluate

    nusc = NuscenesDatasetEval()
    per_category_results = {}

    image_stat = 0
    seq_step = [25]
    t_scene_all = 0
    abs_rel_all =0
    mae_all =0
    sq_rel_all =0
    delta_all = 0
    for scene in val:
        rError = []
        tError = []

        for step in seq_step:
            image_names,extris,intris,depths,valid_masks = nusc.get_data(scene_name=scene,image_stat=image_stat,seq_step=step)
            nusc_data = load_and_preprocess_images_pos(image_names,extris,intris,depths,valid_masks)      
            S,_,H,W = nusc_data["images"].size()
            # nusc_input norm
            real_world_extrinics = nusc_data["extrinsics"] # B S 3 4
            N_cam = 6
            N_seq = S//N_cam
            extrinsics_homog = torch.cat(
                    [
                        real_world_extrinics,
                        torch.zeros((1, S, 1, 4), device=device),
                    ],
                    dim=-2,
                )
            extrinsics_homog[:, :, -1, -1] = 1.0
            first_cam_extrinsic_inv = closed_form_inverse_se3(extrinsics_homog[:, 0])
            # new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv)
            new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv.unsqueeze(1))[...,:3,:]  # (B,N,3,4)
            nusc_data["extrinsics"] = new_extrinsics
           
  
            seq_rError, seq_tError,t_model,[abs_rel,mae,sq_rel,delta] = process_sequence(
                model, [nusc_data,image_names], args.use_ba, device, dtype, seq_name=scene,stat=image_stat,depth_step=step
            )

            print("-" * 50)

            if seq_rError is not None and seq_tError is not None:
                rError.extend(seq_rError)
                tError.extend(seq_tError)

        rError = np.array(rError)
        tError = np.array(tError)

        Auc_30, _ = calculate_auc_np(rError, tError, max_threshold=30)
        Auc_15, _ = calculate_auc_np(rError, tError, max_threshold=15)
        Auc_5, _ = calculate_auc_np(rError, tError, max_threshold=5)
        Auc_3, _ = calculate_auc_np(rError, tError, max_threshold=3)

        per_category_results[scene] = {
            "rError": rError,
            "tError": tError,
            "Auc_30": Auc_30,
            "Auc_15": Auc_15,
            "Auc_5": Auc_5,
            "Auc_3": Auc_3
        }

        print("="*80)
        # Print results with colors
        GREEN = "\033[92m"
        RED = "\033[91m"
        BLUE = "\033[94m"
        BOLD = "\033[1m"
        RESET = "\033[0m"

        print(f"{BOLD}{BLUE}AUC of {scene} test set:{RESET} {GREEN}{Auc_30:.4f} (AUC@30), {Auc_15:.4f} (AUC@15), {Auc_5:.4f} (AUC@5), {Auc_3:.4f} (AUC@3){RESET}")
        mean_AUC_30_by_now = np.mean([per_category_results[scene]["Auc_30"] for scene in per_category_results])
        mean_AUC_15_by_now = np.mean([per_category_results[scene]["Auc_15"] for scene in per_category_results])
        mean_AUC_5_by_now = np.mean([per_category_results[scene]["Auc_5"] for scene in per_category_results])
        mean_AUC_3_by_now = np.mean([per_category_results[scene]["Auc_3"] for scene in per_category_results])
        print(f"{BOLD}{BLUE}Mean AUC of categories by now:{RESET} {RED}{mean_AUC_30_by_now:.4f} (AUC@30), {mean_AUC_15_by_now:.4f} (AUC@15), {mean_AUC_5_by_now:.4f} (AUC@5), {mean_AUC_3_by_now:.4f} (AUC@3){RESET}")
        t_scene_all += t_model
        abs_rel_all += abs_rel
        mae_all += mae
        sq_rel_all += sq_rel
        delta_all += delta
        print(f"{BOLD}{BLUE}Inference time:{RESET} {GREEN},{t_model:.4f},(ms){RESET}")
        print(f"{BOLD}{BLUE}Abs_rel:{RESET} {GREEN},{abs_rel:.4f},{RESET}")
        print(f"{BOLD}{BLUE}Mae:{RESET} {GREEN},{mae:.4f},{RESET}")
        print(f"{BOLD}{BLUE}Sq_rel:{RESET} {GREEN},{sq_rel:.4f},{RESET}")
        print(f"{BOLD}{BLUE}delta:{RESET} {GREEN},{delta:.4f},{RESET}")
        print("="*80)

    # Print summary results
    print("\nSummary of AUC results:")
    print("-"*50)
    for scene in sorted(per_category_results.keys()):
        print(f"{scene:<15}: {per_category_results[scene]['Auc_30']:.4f} (AUC@30), {per_category_results[scene]['Auc_15']:.4f} (AUC@15), {per_category_results[scene]['Auc_5']:.4f} (AUC@5), {per_category_results[scene]['Auc_3']:.4f} (AUC@3)")

    if per_category_results:
        mean_AUC_30 = np.mean([per_category_results[scene]["Auc_30"] for scene in per_category_results])
        mean_AUC_15 = np.mean([per_category_results[scene]["Auc_15"] for scene in per_category_results])
        mean_AUC_5 = np.mean([per_category_results[scene]["Auc_5"] for scene in per_category_results])
        mean_AUC_3 = np.mean([per_category_results[scene]["Auc_3"] for scene in per_category_results])
        print("-"*50)
        print(f"Mean AUC: {mean_AUC_30:.4f} (AUC@30), {mean_AUC_15:.4f} (AUC@15), {mean_AUC_5:.4f} (AUC@5), {mean_AUC_3:.4f} (AUC@3)")
        t_scene_all = t_scene_all/150
        abs_rel_all = abs_rel_all/150
        mae_all = mae_all/150
        sq_rel_all = sq_rel_all/150
        delta_all = delta_all/150
        print(f"{BOLD}{BLUE}Inference Time: {t_scene_all:.4f},(ms)")
        print(f"{BOLD}{BLUE}abs_rel_all: {abs_rel_all:.4f},(ms)")
        print(f"{BOLD}{BLUE}mae_all: {mae_all:.4f},(ms)")
        print(f"{BOLD}{BLUE}sq_rel_all: {sq_rel_all:.4f},(ms)")
        print(f"{BOLD}{BLUE}delta_all: {delta_all:.4f},(ms)")

    print(args.model_path)

if __name__ == "__main__":
    main()