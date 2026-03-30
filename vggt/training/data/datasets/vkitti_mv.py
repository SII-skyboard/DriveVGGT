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

from data.dataset_util import *
from data.base_dataset import BaseDataset


class VKittiDatasetMV(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        VKitti_DIR: str = "/inspire/hdd/global_user/chenxinyan-240108120066/liuyanhao/vggt/datasets/vkitti",
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
        min_frames: int = 3,
        max_frames: int = 18,
    ):
        """
        Initialize the VKittiDatasetMV (Multi-View).

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            VKitti_DIR (str): Directory path to VKitti data.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
            expand_ratio (int): Range for expanding nearby image selection.
            min_frames (int): Minimum number of sequential frames to load.
            max_frames (int): Maximum number of sequential frames to load.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        self.expand_ratio = expand_ratio
        self.VKitti_DIR = VKitti_DIR
        self.min_num_images = min_num_images
        self.min_frames = min_frames
        self.max_frames = max_frames

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        logging.info(f"VKitti_DIR is {self.VKitti_DIR}")

        # Load or generate sequence list (only camera0 sequences as base)
        txt_path = osp.join(self.VKitti_DIR, "sequence_list_camera0.txt")
        if osp.exists(txt_path):
            with open(txt_path, 'r') as f:
                sequence_list = [line.strip() for line in f.readlines()]
        else:
            # Generate sequence list for camera0 only
            sequence_list = glob.glob(osp.join(self.VKitti_DIR, "*/*/*/rgb/Camera_0"))
            sequence_list = [file_path.split(self.VKitti_DIR)[-1].lstrip('/') for file_path in sequence_list]
            # Filter to keep only camera0 sequences
            sequence_list = [seq for seq in sequence_list if 'Camera_0' in seq]
            sequence_list = sorted(sequence_list)

            # Save to txt file
            with open(txt_path, 'w') as f:
                f.write('\n'.join(sequence_list))

        self.sequence_list = sequence_list
        self.sequence_list_len = len(self.sequence_list)

        self.depth_max = 80

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: VKitti Multi-View Data size: {self.sequence_list_len}")
        logging.info(f"{status}: VKitti Multi-View dataset length: {len(self)}")

    def process_one_image_vkitti(
        self,
        image,
        depth_map,
        extri_opencv,
        intri_opencv,
        original_size,
        target_width=518,
        target_height=154,
        track=None,
        filepath=None,
    ):
        """
        Process VKitti image with aspect ratio preservation.
        VKitti images are 1242x375, we resize to 518x154 to maintain aspect ratio.

        Args:
            image: Input image
            depth_map: Depth map
            extri_opencv: Extrinsic matrix
            intri_opencv: Intrinsic matrix
            original_size: Original image size [H, W]
            target_width: Target width (default 518)
            target_height: Target height (default 154)
            track: Optional tracking info
            filepath: Optional file path for debugging

        Returns:
            Processed image, depth, extrinsics, intrinsics, world points, cam points, mask, track
        """
        from data.dataset_util import resize_image_depth_and_intrinsic, depth_to_world_coords_points

        # Make copies
        image = np.copy(image)
        depth_map = np.copy(depth_map)
        extri_opencv = np.copy(extri_opencv)
        intri_opencv = np.copy(intri_opencv)
        if track is not None:
            track = np.copy(track)

        # Target shape for VKitti: maintain aspect ratio
        target_shape = np.array([target_height, target_width])

        # Calculate scale factors
        original_h, original_w = original_size
        scale_h = target_height / original_h
        scale_w = target_width / original_w

        # Resize image and depth map
        image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        depth_map = cv2.resize(depth_map, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

        # Update intrinsics with scale factors
        intri_opencv[0, 0] *= scale_w  # fx
        intri_opencv[1, 1] *= scale_h  # fy
        intri_opencv[0, 2] *= scale_w  # cx
        intri_opencv[1, 2] *= scale_h  # cy

        # Update track if provided
        if track is not None:
            track = track * np.array([scale_w, scale_h])

        # Convert depth to world and camera coordinates
        world_coords_points, cam_coords_points, point_mask = (
            depth_to_world_coords_points(depth_map, extri_opencv, intri_opencv)
        )

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

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve multi-view data for sequential frames.

        Args:
            seq_index (int): Index of the sequence to retrieve.
            img_per_seq (int): Number of images per sequence (not used in MV version).
            seq_name (str): Name of the sequence.
            ids (list): Specific IDs to retrieve (not used in MV version).
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of multi-view data including images from both cameras.
        """
        if self.inside_random and self.training:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        # Get the base sequence path (camera0)
        seq_base = seq_name.replace('Camera_0', '').replace('/rgb/', '')

        # Load camera parameters for both cameras
        camera_params = {}
        camera_intrinsics = {}

        for cam_id in [0, 1]:
            try:
                camera_parameters = np.loadtxt(
                    osp.join(self.VKitti_DIR, "/".join(seq_name.split("/")[:2]), "extrinsic.txt"),
                    delimiter=" ",
                    skiprows=1
                )
                camera_parameters = camera_parameters[camera_parameters[:, 1] == cam_id]

                camera_intrinsic = np.loadtxt(
                    osp.join(self.VKitti_DIR, "/".join(seq_name.split("/")[:2]), "intrinsic.txt"),
                    delimiter=" ",
                    skiprows=1
                )
                camera_intrinsic = camera_intrinsic[camera_intrinsic[:, 1] == cam_id]

                camera_params[cam_id] = camera_parameters
                camera_intrinsics[cam_id] = camera_intrinsic
            except Exception as e:
                logging.error(f"Error loading camera parameters for camera {cam_id}: {e}")
                raise

        num_images = len(camera_params[0])

        # Randomly select number of sequential frames (3-6)
        num_frames = random.randint(self.min_frames, self.max_frames)

        # Randomly select starting frame, ensuring we have enough sequential frames
        max_start_idx = max(0, num_images - num_frames)
        start_idx = random.randint(0, max_start_idx)

        # Generate sequential frame IDs
        frame_ids = list(range(start_idx, start_idx + num_frames))

        target_image_shape = self.get_target_shape(aspect_ratio)

        # Data containers for multi-view
        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        original_sizes = []
        camera_ids = []

        # Load data for each frame and each camera
        for cam_id in [0, 1]:
            for frame_idx in frame_ids:
                # Construct paths for current camera
                cam_seq_name = seq_name.replace('Camera_0', f'Camera_{cam_id}')

                image_filepath = osp.join(self.VKitti_DIR, cam_seq_name, f"rgb_{frame_idx:05d}.jpg")
                depth_filepath = osp.join(self.VKitti_DIR, cam_seq_name, f"depth_{frame_idx:05d}.png").replace("/rgb", "/depth")

                # if self.debug:
                #     print(f"Loading: {image_filepath}")

                image = read_image_cv2(image_filepath)
                depth_map = cv2.imread(depth_filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                depth_map = depth_map / 100
                depth_map = threshold_depth_map(depth_map, max_percentile=-1, min_percentile=-1, max_depth=self.depth_max)
                assert image.shape[:2] == depth_map.shape, f"Image and depth shape mismatch: {image.shape[:2]} vs {depth_map.shape}"

                original_size = np.array(image.shape[:2])

                # Process camera matrices
                extri_opencv = camera_params[cam_id][frame_idx][2:].reshape(4, 4)
                extri_opencv = extri_opencv[:3]

                intri_opencv = np.eye(3)
                intri_opencv[0, 0] = camera_intrinsics[cam_id][frame_idx][-4]
                intri_opencv[1, 1] = camera_intrinsics[cam_id][frame_idx][-3]
                intri_opencv[0, 2] = camera_intrinsics[cam_id][frame_idx][-2]
                intri_opencv[1, 2] = camera_intrinsics[cam_id][frame_idx][-1]

                # Use VKitti-specific image processing
                (
                    image,
                    depth_map,
                    extri_opencv,
                    intri_opencv,
                    world_coords_points,
                    cam_coords_points,
                    point_mask,
                    _,
                ) = self.process_one_image_vkitti(
                    image,
                    depth_map,
                    extri_opencv,
                    intri_opencv,
                    original_size,
                    target_width=518,
                    target_height=154,
                    filepath=image_filepath,
                )

                if (image.shape[:2] != np.array([154, 518])).any():
                    logging.error(f"Wrong shape for {cam_seq_name}: expected {target_image_shape}, got {image.shape[:2]}")
                    continue

                images.append(image)
                depths.append(depth_map)
                extrinsics.append(extri_opencv)
                intrinsics.append(intri_opencv)
                cam_points.append(cam_coords_points)
                world_points.append(world_coords_points)
                point_masks.append(point_mask)
                original_sizes.append(original_size)
                camera_ids.append(cam_id)

        set_name = "vkitti_mv"
        batch = {
            "seq_name": set_name + "_" + seq_name,
            "ids": np.array(len(frame_ids)*2),
            "frame_ids": frame_ids,
            "camera_ids": camera_ids,
            "num_frames": num_frames,
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
