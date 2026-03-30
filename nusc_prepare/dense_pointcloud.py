from nuscenes.nuscenes import NuScenes
import os
from pyquaternion import Quaternion
import numpy as np
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

def get_R_T(quat, trans):
    RT = np.eye(4)
    RT[:3,:3] = Quaternion(quat).rotation_matrix
    RT[:3,3] = np.array(trans)
    return RT


# scenes info
def dense_pc_gen(scene):
    scene = nusc.scene[scene]
    sample_nums = scene['nbr_samples']
    first_sample_token = scene["first_sample_token"]
    last_sample_token = scene["first_sample_token"]
    first_token = first_sample_token
    current_token = first_sample_token
    last_token = last_sample_token
    img_list = []
    instances_list = []

    i = 0
    aggregate_frames = sample_nums

    #1. get aggregate frames of all frames
    aggregate_samples_list = []
    aggregate_sample_token = current_token
    for frame in range(aggregate_frames):
        aggregate_samples_list.append(aggregate_sample_token)
        aggregate_sample_token = nusc.get('sample', aggregate_sample_token)['next']
    aggregate_grounds_list = []
    aggregate_targets_list = []
    aggregate_boxes_list = []
    for frame in aggregate_samples_list:
        sample = nusc.get('sample', frame)
        lidar_path, gt_boxes, _ = nusc.get_sample_data(sample['data']['LIDAR_TOP'])
        # lidar2global
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token']) 
        lidar2ego_translation = cs_record['translation']
        lidar2ego_quat =  cs_record['rotation']
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        ego2global_translation = pose_record['translation']
        ego2global_rotation =  pose_record['rotation']
        lidar2ego_RT = get_R_T(lidar2ego_quat, lidar2ego_translation) # lidar -> ego
        ego2global_RT = get_R_T(ego2global_rotation, ego2global_translation) # lidarego -> global
        lidar2global = ego2global_RT @ lidar2ego_RT # lidar -> global
        
        pts_lidar = np.fromfile(lidar_path, dtype=np.float32).reshape(-1,5)


        # ego_vehicle remove
        # point_ego = (np.linalg.inv(ego2global_RT[:3,:3]) @ (point_global[:,:3] - ego2global_RT[:3,3]).T).T
        # size = np.array([[-1,2],[-1,1],[-10,10]])
        # point_mask = ((point_ego[:,0]>=size[0][1])|(point_ego[:,0]<=-size[0][0]))| \
        #                 ((point_ego[:,1]>=size[1][1])|(point_ego[:,1]<=-size[1][0]))| \
        #                     ((point_ego[:,2]>=size[2][1])|(point_ego[:,2]<=-size[2][0]))
        # point_global = point_global[point_mask]

        size = np.array([[2,2],[2,2],[8,8]])
        point_mask = ((pts_lidar[:,0]>=size[0][1])|(pts_lidar[:,0]<=-size[0][0]))| \
                    ((pts_lidar[:,1]>=size[1][1])|(pts_lidar[:,1]<=-size[1][0]))| \
                        ((pts_lidar[:,2]>=size[2][1])|(pts_lidar[:,2]<=-size[2][0]))
        pts_lidar = pts_lidar[point_mask]

        
        point_global = (lidar2global[:3,:3] @ pts_lidar[:,:3].T).T + lidar2global[:3,3]   
        # anno target remove
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            trans, rot, size = ann['translation'], ann['rotation'], ann['size']
            ann_RT = get_R_T(rot,trans)
            w,l,h = size
            ratio = 1.15
            w = ratio*w
            h = 1*h
            l = ratio*l
            box = Box(trans, size, Quaternion(rot))
            point_ann = (np.linalg.inv(ann_RT[:3,:3]) @ (point_global[:,:3] - ann_RT[:3,3]).T).T
            point_mask = ((point_ann[:,0]>=l/2)|(point_ann[:,0]<=-l/2))| \
                            ((point_ann[:,1]>=w/2)|(point_ann[:,1]<=-w/2))| \
                                ((point_ann[:,2]>=h/2)|(point_ann[:,2]<=-h/2))
                                
            target_mask = ((point_ann[:,0]<=l/2)&(point_ann[:,0]>=-l/2))& \
                            ((point_ann[:,1]<=w/2)&(point_ann[:,1]>=-w/2))& \
                                ((point_ann[:,2]<=h/2)&(point_ann[:,2]>=-h/2))
            target_point = point_global[target_mask]
            aggregate_targets_list.append(target_point)
            point_global = point_global[point_mask]
            #visualize 3D bbox

            
            corners = np.ones((1, 3, 8))
            corners[0, :3, :] = box.corners()
            aggregate_boxes_list.append(corners.transpose(0,2,1)) # 8*3
            
        aggregate_grounds_list.append(point_global)

    # aggregate_grounds = np.concatenate(aggregate_grounds_list, axis=0)
    # write_path = "/inspire/hdd/global_user/chenxinyan-240108120066/liuyanhao/data_nuscenes/aggerate_lidar/g.npy"
    # np.save(write_path, aggregate_grounds)

    # aggregate_box = np.concatenate(aggregate_boxes_list, axis=0)
    # print(aggregate_box.shape)
    # write_path_b = "/inspire/hdd/global_user/chenxinyan-240108120066/liuyanhao/data_nuscenes/aggerate_lidar/b.npy"
    # np.save(write_path_b, aggregate_box)

    # aggregate_t = np.concatenate(aggregate_targets_list, axis=0)
    # write_path_t = "/inspire/hdd/global_user/chenxinyan-240108120066/liuyanhao/data_nuscenes/aggerate_lidar/t.npy"
    # np.save(write_path_t, aggregate_t)
    # assert 1==0


    while current_token:
        # 2. get insatnces of cur and grounds of all
        cur_sample = nusc.get('sample',current_token)
        for ann_token in cur_sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            instance_token = ann['instance_token']
            if instance_token not in instances_list :
                instances_list.append(instance_token)
        # dense instance pc
        
        current_token = cur_sample['next']
        i = i+1
        if current_token == '':
            break
    # print(instances_list)

    aggreagte_instances_pc_dict = {}

    for instance_token in instances_list:
        instance_list = []
        instance = nusc.get('instance', instance_token)
        first_annotation_token = instance['first_annotation_token']
        last_annotation_token = instance['last_annotation_token']
        cur_anno_token = first_annotation_token
        i = 0
        while cur_anno_token:
            cur_anno = nusc.get('sample_annotation', cur_anno_token)
            trans, rot, size = cur_anno['translation'], cur_anno['rotation'], cur_anno['size']
            ann_RT = get_R_T(rot,trans)
            w,l,h = size
            ratio = 1.1
            w = ratio*w
            h = 1*h
            l = ratio*l
            sample = nusc.get('sample', cur_anno['sample_token'])
            lidar_path, gt_boxes, _ = nusc.get_sample_data(sample['data']['LIDAR_TOP'])
            # lidar2global
            sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token']) 
            lidar2ego_translation = cs_record['translation']
            lidar2ego_quat =  cs_record['rotation']
            pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
            ego2global_translation = pose_record['translation']
            ego2global_rotation =  pose_record['rotation']
            lidar2ego_RT = get_R_T(lidar2ego_quat, lidar2ego_translation) # lidar -> ego
            ego2global_RT = get_R_T(ego2global_rotation, ego2global_translation) # lidarego -> global
            lidar2global = ego2global_RT @ lidar2ego_RT # lidar -> global
            
            pts_lidar = np.fromfile(lidar_path, dtype=np.float32).reshape(-1,5)
            point_global = (lidar2global[:3,:3] @ pts_lidar[:,:3].T).T + lidar2global[:3,3]
            # anno target aggregate
            point_ann = (np.linalg.inv(ann_RT[:3,:3]) @ (point_global[:,:3] - ann_RT[:3,3]).T).T
            target_mask = ((point_ann[:,0]<=l/2)&(point_ann[:,0]>=-l/2))& \
                            ((point_ann[:,1]<=w/2)&(point_ann[:,1]>=-w/2))& \
                                ((point_ann[:,2]<=h/2)&(point_ann[:,2]>=-h/2))
            target_point = point_ann[target_mask]
            instance_list.append(target_point)
            i = i+1
            cur_anno_token = cur_anno['next']
            if cur_anno_token == '':
                break
        aggregate_insatnce_pc = np.concatenate(instance_list, axis=0)
        aggreagte_instances_pc_dict.update({instance_token:aggregate_insatnce_pc})

    aggregate_grounds = np.concatenate(aggregate_grounds_list, axis=0)

    current_token = first_token
    i = 0
    while current_token:
        # 2. get insatnces of cur and grounds of all
        cur_aggregate_pc_list = []
        cur_aggregate_pc_list.append(aggregate_grounds)
        cur_sample = nusc.get('sample',current_token)
        for ann_token in cur_sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            trans, rot, size = ann['translation'], ann['rotation'], ann['size']
            ann_RT = get_R_T(rot,trans)
            instance_token = ann['instance_token']
            ann_aggreagte_pc = aggreagte_instances_pc_dict[instance_token]
            ann_aggreagte_pc = (ann_RT[:3,:3] @ ann_aggreagte_pc[:,:3].T).T + ann_RT[:3,3] # global ann pc
            cur_aggregate_pc_list.append(ann_aggreagte_pc)

        aggregate_cur_pc = np.concatenate(cur_aggregate_pc_list, axis=0)
        # remove pc out of scope
        scale_size = [160,160,40]
        w = scale_size[0]
        l = scale_size[1]
        r = 100
        h = scale_size[2]
        sd_rec = nusc.get('sample_data', cur_sample['data']['LIDAR_TOP'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        ego2global_translation = np.array(pose_record['translation'])
        ego2global_rotation =  np.array(pose_record['rotation'])
        ego_RT = get_R_T(ego2global_rotation,ego2global_translation)
        point_ego = (np.linalg.inv(ego_RT[:3,:3]) @ (aggregate_cur_pc[:,:3] - ego_RT[:3,3]).T).T
        # target_mask = ((point_ego[:,0]<=l/2)&(point_ego[:,0]>=-l/2))& \
        #             ((point_ego[:,1]<=w/2)&(point_ego[:,1]>=-w/2))& \
        #                 ((point_ego[:,2]<=h/2)&(point_ego[:,2]>=-h/2))
        target_mask = ((point_ego[:,0]*point_ego[:,0]+point_ego[:,1]*point_ego[:,1])<=r*r)& \
                    ((point_ego[:,2]<=h/2)&(point_ego[:,2]>=-h/2))
        aggregate_cur_pc = aggregate_cur_pc[target_mask]
        
        lidar_path, gt_boxes, _ = nusc.get_sample_data(cur_sample['data']['LIDAR_TOP'])
        relative_lidar_path = lidar_path.split('/')[-1]
        relative_lidar_path = relative_lidar_path.split('.')[0]
        write_path = "/inspire/hdd/project/wuliqifa/public/liuyanhao/aggregate_lidar/"+relative_lidar_path+".npy"
        np.save(write_path, aggregate_cur_pc)
        i = i+1
        
        current_token = cur_sample['next']
        if current_token == '':
            break

data_path = "/inspire/hdd/global_user/chenxinyan-240108120066/liuyanhao/data_nuscenes/nuscenes"
nusc = NuScenes(version='v1.0-trainval',dataroot=data_path,verbose=True)
scnene_num = len(nusc.scene)

print(scnene_num)
set_num=200
# 0-199 200-399
for index in range(200,399):
    print('process the %.0f scene' %index)
    dense_pc_gen(index)
    




