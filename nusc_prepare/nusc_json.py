import torch
from nuscenes.nuscenes import NuScenes
import os
from vggt.utils.load_fn import load_and_preprocess_images
import numpy as np
from pyquaternion import Quaternion
import cv2 
from nuscenes.utils.data_classes import Box
import json

sensor_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT']

def to_matrix4x4_2(rotation, translation, inverse=True):
    output = np.eye(4)
    output[:3, :3] = rotation
    output[:3, 3]  = translation
    
    if inverse:
        output = np.linalg.inv(output)
    return output

def to_matrix4x4(m):
    output = np.eye(4)
    output[:3, :3] = m
    return output

def get_R_T(quat, trans):
    """
        Args:
            quat: 四元素,eg.[w,x,y,z]
            trans: 偏移量，eg.[x',y',z']
        Return:
            RT: 转移矩阵
    """
    RT = np.eye(4)
    RT[:3,:3] = Quaternion(quat).rotation_matrix
    RT[:3,3] = np.array(trans)
    return RT

class NuScenesScenesLoader():
    def __init__(
        self,
        scene_num = 0,
        data_path =  "/inspire/hdd/global_user/chenxinyan-240108120066/liuyanhao/data_nuscenes/nuscenes"
        ):
        super().__init__()
        self.scene_num = scene_num
        self.data_path = data_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def scene_img_preview(self):
        nusc = NuScenes(version='v1.0-trainval',dataroot=self.data_path,verbose=True)
        scnene_num = len(nusc.scene)
        nusc_scene_dict = {}
        for index in range(scnene_num):
            scene = nusc.scene[index]
            
            scene_name = scene['name']
            first_sample_token = scene["first_sample_token"]
            last_sample_token = scene["last_sample_token"]

            first_token = first_sample_token
            current_token = first_sample_token
            last_token = last_sample_token

            img_list = []
            scene_info_list = []

            while current_token:
                # 1. sample
                cur_sample = nusc.get('sample', current_token)
                timestamp = cur_sample['timestamp']
                
                # 2. lidar info
                lidar_path, gt_boxes, _ = nusc.get_sample_data(cur_sample['data']['LIDAR_TOP']) 
                sd_rec = nusc.get('sample_data', cur_sample['data']['LIDAR_TOP'])
                cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token']) 
                
                lidar2ego_translation = cs_record['translation']
                lidar2ego_quat =  cs_record['rotation']
                
                #3. ego pose
                pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
                ego2global_translation = pose_record['translation']
                ego2global_rotation =  pose_record['rotation']
                # lidar and ego pose
                lidar2ego_RT = get_R_T(lidar2ego_quat, lidar2ego_translation) # lidar -> ego
                ego2global_RT = get_R_T(ego2global_rotation, ego2global_translation) # lidarego -> global
                lidar2global = ego2global_RT @ lidar2ego_RT # lidar -> global
                # 2. camera info
                camera_sample = {}
                for sensor in sensor_list:    
                    # cam path and pose    
                    camera_types = sensor
                    cam_token = cur_sample['data'][camera_types]
                    cam_path, _, K = nusc.get_sample_data(cam_token)
                    cam_rec = nusc.get('sample_data', cam_token)
                    cam_record = nusc.get('calibrated_sensor', cam_rec['calibrated_sensor_token'])
                    cam2ego_translation = cam_record['translation']
                    cam2ego_rotation =  cam_record['rotation']
                    cam_pose_record = nusc.get('ego_pose', cam_rec['ego_pose_token'])
                    camego2global_translation = cam_pose_record['translation']
                    camego2global_rotation = cam_pose_record['rotation']
                    cam2ego_RT = get_R_T(cam2ego_rotation, cam2ego_translation) # cam -> ego
                    camego2global_RT = get_R_T(camego2global_rotation, camego2global_translation)

                    # lidar2cam

                    cam2global = camego2global_RT @ cam2ego_RT # cam -> global
                    lidar2cam = np.linalg.inv(cam2global) @ lidar2global # lidar -> cam        
                    
                    # lidar2img and intrinsics
                    cam2img = np.eye(4)
                    cam2img[:3,:3] = K              
                    lidar2img = cam2img @ lidar2cam # lidar -> img
                    
                    # output dict
                    cam_dict = {sensor:{'image_path':cam_path,'intrinsics':K.tolist(),'cam2global':cam2global.tolist(),'cam2ego':cam2ego_RT.tolist()}}
                    camera_sample.update(cam_dict)
                
                relative_lidar_path = lidar_path.split('/')[-1]
                relative_lidar_path = relative_lidar_path.split('.')[0]
                write_path = "/inspire/hdd/project/wuliqifa/public/liuyanhao/aggregate_lidar/"+relative_lidar_path+".npy"
                sample_dict = {'timestamp':timestamp,'camera':camera_sample,'lidar':{'lidar_path':write_path,'lidar2global':lidar2global.tolist(),'lidar2img':lidar2img.tolist()}}
                scene_info_list.append(sample_dict)
                
                current_token = cur_sample['next']
                if current_token == '':
                    break

            nusc_scene_dict = {scene_name: scene_info_list}
            
            json_name = scene_name+'.json'
            json_path = "/inspire/hdd/global_user/chenxinyan-240108120066/liuyanhao/data_nuscenes/nuscene_json"
            name = os.path.join(json_path,json_name)
            with open(name, "w") as f:
                json.dump(nusc_scene_dict, f, indent=4, ensure_ascii=False)

        return img_list

    
    def nuscenes_dataloader(self):
        img_list, mask_list = self.scene_img_preview()
        images = load_and_preprocess_images(img_list).to(self.device)
        masks = load_and_preprocess_images(mask_list).to(self.device)
        return images, masks


nusc = NuScenesScenesLoader()
nusc.scene_img_preview()