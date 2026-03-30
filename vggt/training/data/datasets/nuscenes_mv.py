# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
import logging
import random
import glob

import cv2
import numpy as np
import json
from data.dataset_util import *
from data.base_dataset import BaseDataset
from PIL import Image
import PIL
try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC
from scipy.ndimage import maximum_filter
import sys 

train = \
    ['scene-0001', 'scene-0002', 'scene-0041', 'scene-0042', 'scene-0043', 'scene-0044', 'scene-0045', 'scene-0046',
     'scene-0047', 'scene-0048', 'scene-0049', 'scene-0050', 'scene-0051', 'scene-0052', 'scene-0053', 'scene-0054',
     'scene-0055', 'scene-0056', 'scene-0057', 'scene-0058', 'scene-0059', 'scene-0060', 'scene-0061', 'scene-0062',
     'scene-0063', 'scene-0064', 'scene-0065', 'scene-0066', 'scene-0067', 'scene-0068', 'scene-0069', 'scene-0070',
     'scene-0071', 'scene-0072', 'scene-0073', 'scene-0074', 'scene-0075', 'scene-0076', 'scene-0161', 'scene-0162',
     'scene-0163', 'scene-0164', 'scene-0165', 'scene-0166', 'scene-0167', 'scene-0168', 'scene-0170', 'scene-0171',
     'scene-0172', 'scene-0173', 'scene-0174', 'scene-0175', 'scene-0176', 'scene-0190', 'scene-0191', 'scene-0192',
     'scene-0193', 'scene-0194', 'scene-0195', 'scene-0196', 'scene-0199', 'scene-0200', 'scene-0202', 'scene-0203',
     'scene-0204', 'scene-0206', 'scene-0207', 'scene-0208', 'scene-0209', 'scene-0210', 'scene-0211', 'scene-0212',
     'scene-0213', 'scene-0214', 'scene-0254', 'scene-0255', 'scene-0256', 'scene-0257', 'scene-0258', 'scene-0259',
     'scene-0260', 'scene-0261', 'scene-0262', 'scene-0263', 'scene-0264', 'scene-0283', 'scene-0284', 'scene-0285',
     'scene-0286', 'scene-0287', 'scene-0288', 'scene-0289', 'scene-0290', 'scene-0291', 'scene-0292', 'scene-0293',
     'scene-0294', 'scene-0295', 'scene-0296', 'scene-0297', 'scene-0298', 'scene-0299', 'scene-0300', 'scene-0301',
     'scene-0302', 'scene-0303', 'scene-0304', 'scene-0305', 'scene-0306', 'scene-0315', 'scene-0316', 'scene-0317',
     'scene-0318', 'scene-0321', 'scene-0323', 'scene-0324', 'scene-0347', 'scene-0348', 'scene-0349', 'scene-0350',
     'scene-0351', 'scene-0352', 'scene-0353', 'scene-0354', 'scene-0355', 'scene-0356', 'scene-0357', 'scene-0358',
     'scene-0359', 'scene-0360', 'scene-0361', 'scene-0362', 'scene-0363', 'scene-0364', 'scene-0365', 'scene-0366',
     'scene-0367', 'scene-0368', 'scene-0369', 'scene-0370', 'scene-0371', 'scene-0372', 'scene-0373', 'scene-0374',
     'scene-0375', 'scene-0382', 'scene-0420', 'scene-0421', 'scene-0422', 'scene-0423', 'scene-0424', 'scene-0425',
     'scene-0426', 'scene-0427', 'scene-0428', 'scene-0429', 'scene-0430', 'scene-0431', 'scene-0432', 'scene-0433',
     'scene-0434', 'scene-0435', 'scene-0436', 'scene-0437', 'scene-0438', 'scene-0439', 'scene-0457', 'scene-0458',
     'scene-0459', 'scene-0461', 'scene-0462', 'scene-0463', 'scene-0464', 'scene-0465', 'scene-0467', 'scene-0468',
     'scene-0469', 'scene-0471', 'scene-0472', 'scene-0474', 'scene-0475', 'scene-0476', 'scene-0477', 'scene-0478',
     'scene-0479', 'scene-0480', 'scene-0566', 'scene-0568', 'scene-0570', 'scene-0571', 'scene-0572', 'scene-0573',
     'scene-0574', 'scene-0575', 'scene-0576', 'scene-0577', 'scene-0578', 'scene-0580', 'scene-0582', 'scene-0583',
     'scene-0665', 'scene-0666', 'scene-0667', 'scene-0668', 'scene-0669', 'scene-0670', 'scene-0671', 'scene-0672',
     'scene-0673', 'scene-0674', 'scene-0675', 'scene-0676', 'scene-0677', 'scene-0678', 'scene-0679', 'scene-0681',
     'scene-0683', 'scene-0684', 'scene-0685', 'scene-0686', 'scene-0687', 'scene-0688', 'scene-0689', 'scene-0739',
     'scene-0740', 'scene-0741', 'scene-0744', 'scene-0746', 'scene-0747', 'scene-0749', 'scene-0750', 'scene-0751',
     'scene-0752', 'scene-0757', 'scene-0758', 'scene-0759', 'scene-0760', 'scene-0761', 'scene-0762', 'scene-0763',
     'scene-0764', 'scene-0765', 'scene-0767', 'scene-0768', 'scene-0769', 'scene-0868', 'scene-0869', 'scene-0870',
     'scene-0871', 'scene-0872', 'scene-0873', 'scene-0875', 'scene-0876', 'scene-0877', 'scene-0878', 'scene-0880',
     'scene-0882', 'scene-0883', 'scene-0884', 'scene-0885', 'scene-0886', 'scene-0887', 'scene-0888', 'scene-0889',
     'scene-0890', 'scene-0891', 'scene-0892', 'scene-0893', 'scene-0894', 'scene-0895', 'scene-0896', 'scene-0897',
     'scene-0898', 'scene-0899', 'scene-0900', 'scene-0901', 'scene-0902', 'scene-0903', 'scene-0945', 'scene-0947',
     'scene-0949', 'scene-0952', 'scene-0953', 'scene-0955', 'scene-0956', 'scene-0957', 'scene-0958', 'scene-0959',
     'scene-0960', 'scene-0961', 'scene-0975', 'scene-0976', 'scene-0977', 'scene-0978', 'scene-0979', 'scene-0980',
     'scene-0981', 'scene-0982', 'scene-0983', 'scene-0984', 'scene-0988', 'scene-0989', 'scene-0990', 'scene-0991',
     'scene-1011', 'scene-1012', 'scene-1013', 'scene-1014', 'scene-1015', 'scene-1016', 'scene-1017', 'scene-1018',
     'scene-1019', 'scene-1020', 'scene-1021', 'scene-1022', 'scene-1023', 'scene-1024', 'scene-1025', 'scene-1074',
     'scene-1075', 'scene-1076', 'scene-1077', 'scene-1078', 'scene-1079', 'scene-1080', 'scene-1081', 'scene-1082',
     'scene-1083', 'scene-1084', 'scene-1085', 'scene-1086', 'scene-1087', 'scene-1088', 'scene-1089', 'scene-1090',
     'scene-1091', 'scene-1092', 'scene-1093', 'scene-1094', 'scene-1095', 'scene-1096', 'scene-1097', 'scene-1098',
     'scene-1099', 'scene-1100', 'scene-1101', 'scene-1102', 'scene-1104', 'scene-1105', 'scene-0004', 'scene-0005', 
     'scene-0006', 'scene-0007', 'scene-0008', 'scene-0009', 'scene-0010', 'scene-0011',
     'scene-0019', 'scene-0020', 'scene-0021', 'scene-0022', 'scene-0023', 'scene-0024', 'scene-0025', 'scene-0026',
     'scene-0027', 'scene-0028', 'scene-0029', 'scene-0030', 'scene-0031', 'scene-0032', 'scene-0033', 'scene-0034',
     'scene-0120', 'scene-0121', 'scene-0122', 'scene-0123', 'scene-0124', 'scene-0125', 'scene-0126', 'scene-0127',
     'scene-0128', 'scene-0129', 'scene-0130', 'scene-0131', 'scene-0132', 'scene-0133', 'scene-0134', 'scene-0135',
     'scene-0138', 'scene-0139', 'scene-0149', 'scene-0150', 'scene-0151', 'scene-0152', 'scene-0154', 'scene-0155',
     'scene-0157', 'scene-0158', 'scene-0159', 'scene-0160', 'scene-0177', 'scene-0178', 'scene-0179', 'scene-0180',
     'scene-0181', 'scene-0182', 'scene-0183', 'scene-0184', 'scene-0185', 'scene-0187', 'scene-0188', 'scene-0218',
     'scene-0219', 'scene-0220', 'scene-0222', 'scene-0224', 'scene-0225', 'scene-0226', 'scene-0227', 'scene-0228',
     'scene-0229', 'scene-0230', 'scene-0231', 'scene-0232', 'scene-0233', 'scene-0234', 'scene-0235', 'scene-0236',
     'scene-0237', 'scene-0238', 'scene-0239', 'scene-0240', 'scene-0241', 'scene-0242', 'scene-0243', 'scene-0244',
     'scene-0245', 'scene-0246', 'scene-0247', 'scene-0248', 'scene-0249', 'scene-0250', 'scene-0251', 'scene-0252',
     'scene-0253', 'scene-0328', 'scene-0376', 'scene-0377', 'scene-0378', 'scene-0379', 'scene-0380', 'scene-0381',
     'scene-0383', 'scene-0384', 'scene-0385', 'scene-0386', 'scene-0388', 'scene-0389', 'scene-0390', 'scene-0391',
     'scene-0392', 'scene-0393', 'scene-0394', 'scene-0395', 'scene-0396', 'scene-0397', 'scene-0398', 'scene-0399',
     'scene-0400', 'scene-0401', 'scene-0402', 'scene-0403', 'scene-0405', 'scene-0406', 'scene-0407', 'scene-0408',
     'scene-0410', 'scene-0411', 'scene-0412', 'scene-0413', 'scene-0414', 'scene-0415', 'scene-0416', 'scene-0417',
     'scene-0418', 'scene-0419', 'scene-0440', 'scene-0441', 'scene-0442', 'scene-0443', 'scene-0444', 'scene-0445',
     'scene-0446', 'scene-0447', 'scene-0448', 'scene-0449', 'scene-0450', 'scene-0451', 'scene-0452', 'scene-0453',
     'scene-0454', 'scene-0455', 'scene-0456', 'scene-0499', 'scene-0500', 'scene-0501', 'scene-0502', 'scene-0504',
     'scene-0505', 'scene-0506', 'scene-0507', 'scene-0508', 'scene-0509', 'scene-0510', 'scene-0511', 'scene-0512',
     'scene-0513', 'scene-0514', 'scene-0515', 'scene-0517', 'scene-0518', 'scene-0525', 'scene-0526', 'scene-0527',
     'scene-0528', 'scene-0529', 'scene-0530', 'scene-0531', 'scene-0532', 'scene-0533', 'scene-0534', 'scene-0535',
     'scene-0536', 'scene-0537', 'scene-0538', 'scene-0539', 'scene-0541', 'scene-0542', 'scene-0543', 'scene-0544',
     'scene-0545', 'scene-0546', 'scene-0584', 'scene-0585', 'scene-0586', 'scene-0587', 'scene-0588', 'scene-0589',
     'scene-0590', 'scene-0591', 'scene-0592', 'scene-0593', 'scene-0594', 'scene-0595', 'scene-0596', 'scene-0597',
     'scene-0598', 'scene-0599', 'scene-0600', 'scene-0639', 'scene-0640', 'scene-0641', 'scene-0642', 'scene-0643',
     'scene-0644', 'scene-0645', 'scene-0646', 'scene-0647', 'scene-0648', 'scene-0649', 'scene-0650', 'scene-0651',
     'scene-0652', 'scene-0653', 'scene-0654', 'scene-0655', 'scene-0656', 'scene-0657', 'scene-0658', 'scene-0659',
     'scene-0660', 'scene-0661', 'scene-0662', 'scene-0663', 'scene-0664', 'scene-0695', 'scene-0696', 'scene-0697',
     'scene-0698', 'scene-0700', 'scene-0701', 'scene-0703', 'scene-0704', 'scene-0705', 'scene-0706', 'scene-0707',
     'scene-0708', 'scene-0709', 'scene-0710', 'scene-0711', 'scene-0712', 'scene-0713', 'scene-0714', 'scene-0715',
     'scene-0716', 'scene-0717', 'scene-0718', 'scene-0719', 'scene-0726', 'scene-0727', 'scene-0728', 'scene-0730',
     'scene-0731', 'scene-0733', 'scene-0734', 'scene-0735', 'scene-0736', 'scene-0737', 'scene-0738', 'scene-0786',
     'scene-0787', 'scene-0789', 'scene-0790', 'scene-0791', 'scene-0792', 'scene-0803', 'scene-0804', 'scene-0805',
     'scene-0806', 'scene-0808', 'scene-0809', 'scene-0810', 'scene-0811', 'scene-0812', 'scene-0813', 'scene-0815',
     'scene-0816', 'scene-0817', 'scene-0819', 'scene-0820', 'scene-0821', 'scene-0822', 'scene-0847', 'scene-0848',
     'scene-0849', 'scene-0850', 'scene-0851', 'scene-0852', 'scene-0853', 'scene-0854', 'scene-0855', 'scene-0856',
     'scene-0858', 'scene-0860', 'scene-0861', 'scene-0862', 'scene-0863', 'scene-0864', 'scene-0865', 'scene-0866',
     'scene-0992', 'scene-0994', 'scene-0995', 'scene-0996', 'scene-0997', 'scene-0998', 'scene-0999', 'scene-1000',
     'scene-1001', 'scene-1002', 'scene-1003', 'scene-1004', 'scene-1005', 'scene-1006', 'scene-1007', 'scene-1008',
     'scene-1009', 'scene-1010', 'scene-1044', 'scene-1045', 'scene-1046', 'scene-1047', 'scene-1048', 'scene-1049',
     'scene-1050', 'scene-1051', 'scene-1052', 'scene-1053', 'scene-1054', 'scene-1055', 'scene-1056', 'scene-1057',
     'scene-1058', 'scene-1106', 'scene-1107', 'scene-1108', 'scene-1109', 'scene-1110']

class NuscenesDataset_MultiView(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        nusc_info: str = "/inspire/hdd/global_user/chenxinyan-240108120066/liuyanhao/data_nuscenes/nuscene_json/",
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
        
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
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        
        self.expand_ratio = expand_ratio
        self.min_num_images = min_num_images
        self.cam_list = ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']
        self.seq_step = random.randint(3, 10)
        # self.seq_step = 2
        
        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")
        self.nusc_info = nusc_info
        
        self.sequence_list = train
        # self.sequence_list.sort()
        self.sequence_list_len = len(self.sequence_list)
        self.depth_max = 80
        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: Nusc Data size: {self.sequence_list_len}")
        logging.info(f"{status}: Nusc Data dataset length: {len(self)}")
        
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
                (depth <= depth_max) & (depth > 0)
        coor, depth = coor[kept1], depth[kept1]
        
        coor = coor.astype(np.int32)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        eps = 10e-6
        valid_mask = np.where(depth_map > eps,True,False)
        
        # # # visualization
        # # for u in range(width):
        # #     for v in range(height):
        # #         if depth_map[v,u]>depth_max:
        # #             depth_map[v,u]=depth_max
        # downsample
        # patch = 2
        # depth_map = depth_map.reshape((height//patch,patch,width//patch,patch)).transpose(0,2,1,3).reshape((-1,patch*patch))
        # depth_downsample = np.zeros([depth_map.shape[0],1])
        # eps = 10e-6
        # for index in range(depth_map.shape[0]):
        #     depth_valid = depth_map[index,:]
        #     depth_valid_index = np.where(depth_valid>eps)
            
        #     depth_valid = depth_valid[depth_valid_index]
        #     if depth_valid.size == 0:
        #         pass
        #     else:
        #         depth_min = np.min(depth_valid)ite
        #         depth_downsample[index] = depth_min
        
        # depth_downsample = depth_downsample.reshape((height//patch,width//patch,1))
        # valid_mask = valid_mask.reshape((height//patch,patch,width//patch,patch)).transpose(0,2,1,3).reshape((-1,patch*patch))
        # depth_map = cv2.resize(depth_downsample,(width,height)).astype(np.uint8)
        
        
        norm_depth = np.power(depth_map/depth_max*255,1).astype(np.uint8)
        depth_vis = cv2.applyColorMap(norm_depth, cv2.COLORMAP_JET)
        # depth_vis = np.where(depth_vis==0,255,depth_vis)
        cv2.imwrite('./test_depth_raw.jpg',depth_vis)
        
        sys.path.append("/inspire/hdd/global_user/chenxinyan-240108120066/liuyanhao/data_nuscenes/ip_basic")
        from ip_basic import depth_map_utils
        extrapolate = False
        # blur_type = 'gaussian'
        blur_type='bilateral'
        depth_map = depth_map_utils.fill_in_fast(
            depth_map,max_depth=80.0, extrapolate=extrapolate, blur_type=blur_type)
        
        norm_depth = np.power(depth_map/depth_max*255,1).astype(np.uint8)
        depth_vis = cv2.applyColorMap(norm_depth, cv2.COLORMAP_JET)
        # depth_vis = np.where(depth_vis==[0,0,0],255,depth_vis)
        cv2.imwrite('./test_depth.jpg',depth_vis)
        assert 1==0
        
        return depth_map

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
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
        
        
        if self.inside_random and self.training:
            seq_index = random.randint(0, (self.sequence_list_len - 1))
            
        
        if seq_name is None:
            seq_name = self.sequence_list[seq_index]
            seq_name = seq_name.split('.')[0]

        json_path = os.path.join(self.nusc_info, seq_name+".json")
        with open(json_path, "r") as f:
            scene1 = json.load(f)
        aspect_ratio = 0.5406
        target_image_shape = self.get_target_shape(aspect_ratio)# no
        scene_name = seq_name
        
        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        original_sizes = []
        images_path = []
        image_len = random.randint(3, 10)
        image_stat = random.randint(0, len(scene1[scene_name]) - image_len - 1)
        camera_list = range(image_stat,image_stat+image_len)
        # for time_step in range(len(scene1[scene_name])):
        
        f_seq,fl_seq,fr_seq,b_seq,bl_seq,br_seq =  [[] for x in range(len(self.cam_list))]
        
        for cam_pos in self.cam_list:
            for time_step in camera_list:               
                lidar_path = scene1[scene_name][time_step]['lidar']['lidar_path']
                print(lidar_path)
                img_path = scene1[scene_name][time_step]['camera'][cam_pos]['image_path']
                K = np.array(scene1[scene_name][time_step]['camera'][cam_pos]['intrinsics'])
                cam2global = np.array(scene1[scene_name][time_step]['camera'][cam_pos]['cam2global'])
                lidar2cam = np.linalg.inv(cam2global)  # lidar -> cam As lidar is in the axis of global
                cam2img = np.eye(4)
                cam2img[:3,:3] = K
                lidar2img = cam2img @ lidar2cam 
                pts_lidar = np.load(lidar_path).reshape(-1,3)
                RGB_image = cv2.imread(img_path)
                original_size = np.array(RGB_image.shape[:2])# no
                extri_opencv = cam2global[:3]
                # debug
                extri_opencv = np.linalg.inv(cam2global)
                
                extri_opencv = extri_opencv[:3]
                intri_opencv = K
                (
                    image,
                    depth_map,
                    extri_opencv,
                    intri_opencv,
                    world_coords_points,
                    cam_coords_points,
                    point_mask,
                    _,
                ) = self.process_one_image_nusc(
                    RGB_image,
                    pts_lidar,
                    extri_opencv,
                    intri_opencv,
                    original_size,
                    target_image_shape,
                    filepath=img_path,
                )
                images.append(image)
                depths.append(depth_map)
                extrinsics.append(extri_opencv)
                intrinsics.append(intri_opencv)
                cam_points.append(cam_coords_points)
                world_points.append(world_coords_points)
                point_masks.append(point_mask)
                original_sizes.append(original_size)
                # # debug
                # images_path.append(img_path)
        num_images = image_len*len(self.cam_list)
        ids = np.arange(num_images)

        #debug
        # print(images_path)
        
        # images_array = np.array(images).transpose(0,3,1,2)
        # # .transpose(0,3,1,2) # (476, 518, 3)
        # depths = np.array(depths)[:,:,:,np.newaxis]
        # extrinsics = np.array(extrinsics)
        # intrinsics = np.array(intrinsics)
        # world_points = np.array(world_points)
        # world_points_conf = np.ones_like(world_points)[:,:,:,0]
        # depths_conf = np.ones_like(depths)[:,:,:,0]
        # extrinsics[:,:,3] = extrinsics[:,:,3]
        # print(images_array.shape)
        # print(depths.shape)
        # print(extrinsics)
        # print(intrinsics.shape)
        # print(world_points.shape)
        # print(world_points_conf.shape)
        # print(depths_conf.shape)
        
        # pred_dict = \
        # {
        #     "images": images_array ,
        #     "world_points": world_points,
        #     "world_points_conf": world_points_conf,
        #     "depth": depths,
        #     "depth_conf": depths_conf,
        #     "extrinsic": extrinsics,
        #     "intrinsic": intrinsics,
        # }
        # import sys
        # sys.path.append("/inspire/hdd/global_user/chenxinyan-240108120066/liuyanhao/vggt")
        # from demo_viser import viser_wrapper
        # viser_wrapper(pred_dict,use_point_map=True)
        # assert 1==0
        
        set_name = "nuscenes"
        batch = {
            "seq_name": set_name + "_" + seq_name,
            "ids": ids,
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }
        return batch

    def process_one_image_nusc_old(
        self,
        image,
        pts_lidar,
        extri_opencv,
        intri_opencv,
        original_size,
        target_image_shape,
        track=None,
        filepath=None,
        safe_bound=4,
    ):
        """
        Process a single image and its associated data.

        This method handles image transformations, depth processing, and coordinate conversions.

        Args:
            image (numpy.ndarray): Input image array
            depth_map (numpy.ndarray): Depth map array
            extri_opencv (numpy.ndarray): Extrinsic camera matrix (OpenCV convention)
            intri_opencv (numpy.ndarray): Intrinsic camera matrix (OpenCV convention)
            original_size (numpy.ndarray): Original image size [height, width]
            target_image_shape (numpy.ndarray): Target image shape after processing
            track (numpy.ndarray, optional): Optional tracking information. Defaults to None.
            filepath (str, optional): Optional file path for debugging. Defaults to None.
            safe_bound (int, optional): Safety margin for cropping operations. Defaults to 4.

        Returns:
            tuple: (
                image (numpy.ndarray): Processed image,
                depth_map (numpy.ndarray): Processed depth map,
                extri_opencv (numpy.ndarray): Updated extrinsic matrix,
                intri_opencv (numpy.ndarray): Updated intrinsic matrix,
                world_coords_points (numpy.ndarray): 3D points in world coordinates,
                cam_coords_points (numpy.ndarray): 3D points in camera coordinates,
                point_mask (numpy.ndarray): Boolean mask of valid points,
                track (numpy.ndarray, optional): Updated tracking information
            )
        """
        # Make copies to avoid in-place operations affecting original data
        image = np.copy(image)
        extri_opencv = np.copy(extri_opencv)
        intri_opencv = np.copy(intri_opencv)
        if track is not None:
            track = np.copy(track)

        # Apply random scale augmentation during training if enabled
        if self.training and self.aug_scale:
            random_h_scale, random_w_scale = np.random.uniform(
                self.aug_scale[0], self.aug_scale[1], 2
            )
            # Avoid random padding by capping at 1.0
            random_h_scale = min(random_h_scale, 1.0)
            random_w_scale = min(random_w_scale, 1.0)
            aug_size = original_size * np.array([random_h_scale, random_w_scale])
            aug_size = aug_size.astype(np.int32)
        else:
            aug_size = original_size

        # Move principal point to the image center and crop if necessary
        # image, depth_map, intri_opencv, track = crop_image_depth_and_intrinsic_by_pp(
        #     image, depth_map, intri_opencv, aug_size, track=track, filepath=filepath,strict=True,
        # )

        original_size = np.array(image.shape[:2])  # update original_size
        target_shape = target_image_shape

        # Handle landscape vs. portrait orientation
        rotate_to_portrait = False
        if self.landscape_check:
            # Switch between landscape and portrait if necessary
            if original_size[0] > 1.25 * original_size[1]:
                if (target_image_shape[0] != target_image_shape[1]) and (np.random.rand() > 0.5):
                    target_shape = np.array([target_image_shape[1], target_image_shape[0]])
                    rotate_to_portrait = True

        # print('raw_shape',image.shape)
        # Resize images and update intrinsics
        if self.rescale:
            # print('target_shape',target_shape)
            image_input, _, track = resize_image_and_intrinsic(
                image, intri_opencv, target_shape, original_size, track=track,
                # safe_bound=safe_bound,
                safe_bound=0,
                rescale_aug=False,               
            )
            minimal_size = np.array([110,258])
            image, intri_opencv, track = resize_image_and_intrinsic(
                image, intri_opencv, minimal_size, original_size, track=track,
                # safe_bound=safe_bound,
                safe_bound=0,
                rescale_aug=False,               
            )
            lidar2img = intri_opencv @ extri_opencv 
            depth_map = self.lidar2depthmap(pts_lidar,image,lidar2img)
            
            image, intri_opencv, track = resize_image_and_intrinsic(
                image, intri_opencv, target_shape, minimal_size, track=track,
                # safe_bound=safe_bound,
                safe_bound=0,
                rescale_aug=False,
                
            )
            # print('depth_shape_raw',depth_map.shape)
            resize_scales = target_shape / minimal_size
            max_resize_scale = np.max(resize_scales)
            depth_map =  cv2.resize(depth_map,
                                    [image.shape[1],image.shape[0]],
                                    interpolation=cv2.INTER_NEAREST
                                    )
            image = image_input
            # print('depth_shape',max_resize_scale,depth_map.shape,image.shape,target_shape)
            
            
        else:
            print("Not rescaling the images")

        # target_shape = (224,518)
        # Ensure final crop to target shape
        image, depth_map, intri_opencv, track = crop_image_depth_and_intrinsic_by_pp(
            image, depth_map, intri_opencv, target_shape, track=track, filepath=filepath, strict=True,
        )
        # debug
        # norm_depth = np.power(depth_map/80*255,2).astype(np.uint8)
        # depth_vis = cv2.applyColorMap(norm_depth, cv2.COLORMAP_AUTUMN)
        # cv2.imwrite('./test_depth.jpg',(0.3*depth_vis+0.7*image).astype(np.uint8))
            
        # Apply 90-degree rotation if needed
        if rotate_to_portrait:
            assert self.landscape_check
            clockwise = np.random.rand() > 0.5
            image, depth_map, extri_opencv, intri_opencv, track = rotate_90_degrees(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                clockwise=clockwise,
                track=track,
            )

        # Convert depth to world and camera coordinates
        world_coords_points, cam_coords_points, point_mask = (
            depth_to_world_coords_points(depth_map, extri_opencv, intri_opencv)
        )
        # print('newsize',image.shape)
        return (
            image,
            depth_map,
            extri_opencv,
            intri_opencv,
            world_coords_points,
            cam_coords_points,
            point_mask,
            track,
        )

    def process_one_image_nusc(
        self,
        image,
        pts_lidar,
        extri_opencv,
        intri_opencv,
        original_size,
        target_image_shape,
        track=None,
        filepath=None,
        safe_bound=4,
    ):
        # Make copies to avoid in-place operations affecting original data
        image = np.copy(image)
        extri_opencv = np.copy(extri_opencv)
        intri_opencv = np.copy(intri_opencv)
        if track is not None:
            track = np.copy(track)

        # Apply random scale augmentation during training if enabled
        if self.training and self.aug_scale:
            random_h_scale, random_w_scale = np.random.uniform(
                self.aug_scale[0], self.aug_scale[1], 2
            )
            # Avoid random padding by capping at 1.0
            random_h_scale = min(random_h_scale, 1.0)
            random_w_scale = min(random_w_scale, 1.0)
            aug_size = original_size * np.array([random_h_scale, random_w_scale])
            aug_size = aug_size.astype(np.int32)
        else:
            aug_size = original_size
        original_size = np.array(image.shape[:2])  # update original_size
        target_shape = target_image_shape
        
        lidar2img = intri_opencv @ extri_opencv 
        depth_map = self.lidar2depthmap(pts_lidar,image,lidar2img)
        
        # Handle landscape vs. portrait orientation
        rotate_to_portrait = False
        if self.landscape_check:
            # Switch between landscape and portrait if necessary
            if original_size[0] > 1.25 * original_size[1]:
                if (target_image_shape[0] != target_image_shape[1]) and (np.random.rand() > 0.5):
                    target_shape = np.array([target_image_shape[1], target_image_shape[0]])
                    rotate_to_portrait = True

        if self.rescale:
            # print('target_shape',target_shape)
            image, intri_opencv, track = resize_image_and_intrinsic(
                image, intri_opencv, target_shape, original_size, track=track,
                # safe_bound=safe_bound,
                safe_bound=0,
                rescale_aug=False,               
            )

            depth_map =  cv2.resize(depth_map,
                                    [image.shape[1],image.shape[0]],
                                    interpolation=cv2.INTER_NEAREST)
            
            # print('after resize',depth_map.shape,image.shape)
            # print('depth_shape',max_resize_scale,depth_map.shape,image.shape,target_shape)
            
        else:
            print("Not rescaling the images")

        # target_shape = (224,518)
        # Ensure final crop to target shape
        image, depth_map, intri_opencv, track = crop_image_depth_and_intrinsic_by_pp(
            image, depth_map, intri_opencv, target_shape, track=track, filepath=filepath, strict=True,
        )
        # debug
        # norm_depth = np.power(depth_map/80*255,2).astype(np.uint8)
        # depth_vis = cv2.applyColorMap(norm_depth, cv2.COLORMAP_AUTUMN)
        # cv2.imwrite('./test_depth.jpg',(0.3*depth_vis+0.7*image).astype(np.uint8))
            
        # Apply 90-degree rotation if needed
        if rotate_to_portrait:
            assert self.landscape_check
            clockwise = np.random.rand() > 0.5
            image, depth_map, extri_opencv, intri_opencv, track = rotate_90_degrees(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                clockwise=clockwise,
                track=track,
            )

        # Convert depth to world and camera coordinates
        world_coords_points, cam_coords_points, point_mask = (
            depth_to_world_coords_points(depth_map, extri_opencv, intri_opencv)
        )
        # print('newsize',image.shape)
        return (
            image,
            depth_map,
            extri_opencv,
            intri_opencv,
            world_coords_points,
            cam_coords_points,
            point_mask,
            track,
        )

def resize_image_and_intrinsic(
    image,
    intrinsic,
    target_shape,
    original_size,
    track=None,
    pixel_center=True,
    safe_bound=4,
    rescale_aug=True,
):
    """
    Resizes the given image and depth map (if provided) to slightly larger than `target_shape`,
    updating the intrinsic matrix (and track array if present). Optionally uses random rescaling
    to create some additional margin (based on `rescale_aug`).

    Steps:
      1. Compute a scaling factor so that the resized result is at least `target_shape + safe_bound`.
      2. Apply an optional triangular random factor if `rescale_aug=True`.
      3. Resize the image with LANCZOS if downscaling, BICUBIC if upscaling.
      4. Resize the depth map with nearest-neighbor.
      5. Update the camera intrinsic and track coordinates (if any).

    Args:
        image (np.ndarray):
            Input image array (H, W, 3).
        depth_map (np.ndarray or None):
            Depth map array (H, W), or None if unavailable.
        intrinsic (np.ndarray):
            Camera intrinsic matrix (3x3).
        target_shape (np.ndarray or tuple[int, int]):
            Desired final shape (height, width).
        original_size (np.ndarray or tuple[int, int]):
            Original size of the image in (height, width).
        track (np.ndarray or None):
            Optional (N, 2) array of pixel coordinates. Will be scaled.
        pixel_center (bool):
            If True, accounts for 0.5 pixel center shift during resizing.
        safe_bound (int or float):
            Additional margin (in pixels) to add to target_shape before resizing.
        rescale_aug (bool):
            If True, randomly increase the `safe_bound` within a certain range to simulate augmentation.

    Returns:
        tuple:
            (resized_image, resized_depth_map, updated_intrinsic, updated_track)

            - resized_image (np.ndarray): The resized image.
            - resized_depth_map (np.ndarray or None): The resized depth map.
            - updated_intrinsic (np.ndarray): Camera intrinsic updated for new resolution.
            - updated_track (np.ndarray or None): Track array updated or None if not provided.

    Raises:
        AssertionError:
            If the shapes of the resized image and depth map do not match.
    """
    if rescale_aug:
        random_boundary = np.random.triangular(0, 0, 0.3)
        safe_bound = safe_bound + random_boundary * target_shape.max()
        
    resize_scales = (target_shape + safe_bound) / original_size
    # debug: to avoid extremelly crop, using np.min instead of np.max
    # max_resize_scale = np.max(resize_scales)
    max_resize_scale = np.max(resize_scales)
    intrinsic = np.copy(intrinsic)

    # Convert image to PIL for resizing
    image = Image.fromarray(image)
    input_resolution = np.array(image.size)
    output_resolution = np.floor(input_resolution * max_resize_scale).astype(int)
    image = image.resize(tuple(output_resolution), resample=lanczos if max_resize_scale < 1 else bicubic)
    image = np.array(image)

    actual_size = np.array(image.shape[:2])
    actual_resize_scale = np.max(actual_size / original_size)

    if pixel_center:
        intrinsic[0, 2] = intrinsic[0, 2] + 0.5
        intrinsic[1, 2] = intrinsic[1, 2] + 0.5

    intrinsic[:2, :] = intrinsic[:2, :] * actual_resize_scale

    if track is not None:
        track = track * actual_resize_scale

    if pixel_center:
        intrinsic[0, 2] = intrinsic[0, 2] - 0.5
        intrinsic[1, 2] = intrinsic[1, 2] - 0.5

    return image, intrinsic, track