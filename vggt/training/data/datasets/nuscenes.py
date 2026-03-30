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

class NuscenesDataset(BaseDataset):
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
        self.seq_step = 5
        
        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")
        self.nusc_info = nusc_info
        
        self.sequence_list = os.listdir(nusc_info)
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
        
        # depth_map = maximum_filter(depth_map,size=(6,6))
        # norm_depth = np.power(depth_map/depth_max*255,1.5).astype(np.uint8)
        # depth_vis = cv2.applyColorMap(norm_depth, cv2.COLORMAP_AUTUMN)

        # cv2.imwrite('./test_depth.jpg',depth_vis)
        # norm_depth = np.power(depth_map/depth_max*255,1).astype(np.uint8)
        # depth_vis = cv2.applyColorMap(norm_depth, cv2.COLORMAP_HOT)
        # cv2.imwrite('./test_depth_raw.jpg',depth_vis)
        
        sys.path.append("/inspire/hdd/global_user/chenxinyan-240108120066/liuyanhao/data_nuscenes/ip_basic")
        from ip_basic import depth_map_utils
        extrapolate = False
        # blur_type = 'gaussian'
        blur_type='bilateral'
        depth_map = depth_map_utils.fill_in_fast(
            depth_map,max_depth=80.0, extrapolate=extrapolate, blur_type=blur_type)
        
        # norm_depth = np.power(depth_map/depth_max*255,1).astype(np.uint8)
        # depth_vis = cv2.applyColorMap(norm_depth, cv2.COLORMAP_HOT)
        # cv2.imwrite('./test_depth.jpg',depth_vis)
        # assert 1==0
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
        image_len = self.seq_step
        image_stat = random.randint(0, len(scene1[scene_name]) - image_len - 1)
        camera_list = range(image_stat,image_stat+image_len)
        # for time_step in range(len(scene1[scene_name])):
        
        f_seq,fl_seq,fr_seq,b_seq,bl_seq,br_seq =  [[] for x in range(len(self.cam_list))]
        
        for time_step in camera_list:
            lidar_path = scene1[scene_name][time_step]['lidar']['lidar_path']
            # print(lidar_path)
            # lidar2global = np.array(scene1[scene_name][time_step]['lidar']['lidar2global'])
            
            for cam_pos in self.cam_list:
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

        # print('raw_shape',image.shape)
        # Resize images and update intrinsics

        if self.rescale:
            # print('target_shape',target_shape)
            image, intri_opencv, track = resize_image_and_intrinsic(
                image, intri_opencv, target_shape, original_size, track=track,
                # safe_bound=safe_bound,
                safe_bound=0,
                rescale_aug=False,               
            )
            # minimal_size = np.array([330,518])
            # image, intri_opencv, track = resize_image_and_intrinsic(
            #     image, intri_opencv, minimal_size, original_size, track=track,
            #     # safe_bound=safe_bound,
            #     safe_bound=0,
            #     rescale_aug=False,               
            # )
            
            # image, intri_opencv, track = resize_image_and_intrinsic(
            #     image, intri_opencv, target_shape, minimal_size, track=track,
            #     # safe_bound=safe_bound,
            #     safe_bound=0,
            #     rescale_aug=False,
            # )
            # print('depth_shape_raw',depth_map.shape)
            # resize_scales = target_shape / minimal_size
            # max_resize_scale = np.max(resize_scales)
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