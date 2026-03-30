import torch
from nuscenes.nuscenes import NuScenes
import os
from vggt.utils.load_fn import load_and_preprocess_images
import numpy as np
from pyquaternion import Quaternion
import cv2 
from nuscenes.utils.data_classes import Box

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

class NuScenesScenesLoader():
    def __init__(
        self,
        scene_num = 0,
        write_path_sample = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/yjc/liuyanhao/data_nuscenes/sample_mask_img",
        data_path =  "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/yjc/liuyanhao/data_nuscenes/nuscenes"
        ):
        super().__init__()
        self.scene_num = scene_num
        self.data_path = data_path
        self.write_path_sample = write_path_sample
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def scene_img_names(self):
        nusc = NuScenes(version='v1.0-trainval',dataroot=self.data_path,verbose=True)
        
        scene = nusc.scene[self.scene_num]

        first_sample_token = scene["first_sample_token"]
        last_sample_token = scene["first_sample_token"]

        first_token = first_sample_token
        current_token = first_sample_token
        last_token = last_sample_token

        img_list = []
        mask_list = []
        i = 0
        while current_token:
            # 1. sample
            cur_sample = nusc.get('sample', current_token)
            # 2. camera images
            cur_cam_front_data = nusc.get('sample_data', cur_sample['data']["CAM_FRONT"])
            cur_cam_front_data_path = cur_cam_front_data['filename']
            direct_cur_cam_front_data_path = os.path.join(self.data_path, cur_cam_front_data_path)
            img_list.append(direct_cur_cam_front_data_path)
            # 3. camera position
            ego_pose           = nusc.get('ego_pose', cur_cam_front_data['ego_pose_token'])
            global_to_ego      = to_matrix4x4_2(Quaternion(ego_pose['rotation']).rotation_matrix, np.array(ego_pose['translation']))
            camera_sensor    = nusc.get('calibrated_sensor', cur_cam_front_data['calibrated_sensor_token']) 
            camera_intrinsic = to_matrix4x4(camera_sensor['camera_intrinsic'])
            ego_to_camera    = to_matrix4x4_2(Quaternion(camera_sensor['rotation']).rotation_matrix, np.array(camera_sensor['translation']))
            global_to_image  = camera_intrinsic @ ego_to_camera @ global_to_ego 
            image = cv2.imread(direct_cur_cam_front_data_path)
            # image = np.zeros_like(image)
            H,W = image.shape[0],image.shape[1]
            # 4. annotation
            for ann_token in cur_sample['anns']:
                anno_cur = nusc.get('sample_annotation', ann_token)
                trans, rot, size = anno_cur['translation'], anno_cur['rotation'], anno_cur['size']
                box = Box(trans, size, Quaternion(rot))
                corners = np.ones((4, 8))
                corners[:3, :] = box.corners() 
                corners = (global_to_image @ corners)[:3]
                
                corners[:2] /= corners[[2]]
                corners = corners.T
                norm_depth = np.where(corners[:,2]<0,-1,1).astype(int) 
                corners[:,0] = (corners[:,0] * norm_depth)
                corners = corners.astype(int) 
                
                x_min, x_max = 0, W
                y_min, y_max = 0, H
                z_min = 0
                
                out_range = (
                    (corners[:, 0] <= x_min) | (corners[:, 0] >= x_max) |
                    (corners[:, 1] <= y_min) | (corners[:, 1] >= y_max) |
                    (corners[:, 2] <= z_min) 
                )               
                
                if out_range.all() == True:
                    continue
                else:
                    
                    w_min = min(corners[:,0])
                    w_max = max(corners[:,0])
                    h_min = min(corners[:,1])
                    h_max = max(corners[:,1])
                    if (w_min>=W)|(w_max<=0)|(h_min>=H)|(h_max<=0):
                        continue
                    else:
                        
                        w_min = max(min(corners[:,0]),0)
                        w_max = min(max(corners[:,0]),W)
                        h_min = max(min(corners[:,1]),0)
                        h_max = min(max(corners[:,1]),H)
                    
                    if (w_min==w_max)|(h_min==h_max):
                        continue
                    else:
                        pass
                        # image[h_min:h_max,w_min:w_max] = 255
            sensor_sample = cur_cam_front_data_path.split('/')[-1]              
            anno_img_path = os.path.join(self.write_path_sample , sensor_sample)
            cv2.imwrite(anno_img_path,image)

            mask_list.append(anno_img_path)
            
            current_token = cur_sample['next']
            i = i + 1

            if current_token == last_token:
                break
        return img_list, mask_list
    
    def scene_img_preview(self):
        nusc = NuScenes(version='v1.0-trainval',dataroot=self.data_path,verbose=True)
        for index in range(20):
            scene = nusc.scene[index]

            first_sample_token = scene["first_sample_token"]
            last_sample_token = scene["first_sample_token"]

            first_token = first_sample_token
            current_token = first_sample_token
            last_token = last_sample_token

            img_list = []
            mask_list = []
            i = 0
            sample_path =  os.path.join(self.write_path_sample, str(index))
            os.makedirs(sample_path)
            while current_token:
                # 1. sample
                cur_sample = nusc.get('sample', current_token)
                # 2. camera images
                for sensor in sensor_list:
                    cur_cam_front_data = nusc.get('sample_data', cur_sample['data'][sensor])
                    
                    cur_cam_front_data_path = cur_cam_front_data['filename']
                    direct_cur_cam_front_data_path = os.path.join(self.data_path, cur_cam_front_data_path)
                    img_list.append(direct_cur_cam_front_data_path)
                    # 3. camera position
                    ego_pose           = nusc.get('ego_pose', cur_cam_front_data['ego_pose_token'])
                    global_to_ego      = to_matrix4x4_2(Quaternion(ego_pose['rotation']).rotation_matrix, np.array(ego_pose['translation']))
                    camera_sensor    = nusc.get('calibrated_sensor', cur_cam_front_data['calibrated_sensor_token']) 
                    camera_intrinsic = to_matrix4x4(camera_sensor['camera_intrinsic'])
                    ego_to_camera    = to_matrix4x4_2(Quaternion(camera_sensor['rotation']).rotation_matrix, np.array(camera_sensor['translation']))
                    global_to_image  = camera_intrinsic @ ego_to_camera @ global_to_ego 
                    image = cv2.imread(direct_cur_cam_front_data_path)
                    sensor_sample = cur_cam_front_data_path.split('/')[-1]       
                    anno_img_path = os.path.join(sample_path , sensor_sample)
                    cv2.imwrite(anno_img_path,image)
                    print(anno_img_path)
                    mask_list.append(anno_img_path)
                    
                    current_token = cur_sample['next']
                    i = i + 1

                    if current_token == last_token:
                        break
        return img_list, mask_list

    
    def nuscenes_dataloader(self):
        img_list, mask_list = self.scene_img_preview()
        images = load_and_preprocess_images(img_list).to(self.device)
        masks = load_and_preprocess_images(mask_list).to(self.device)
        return images, masks
