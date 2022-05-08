'''
load hand point data
'''
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
from os.path import join as pjoin
from copy import deepcopy
from manopth.manolayer import ManoLayer

import sys 
BASEPATH = os.path.dirname(__file__)
sys.path.append(pjoin(BASEPATH, '../../../network/models'))
from pointnet_utils import farthest_point_sample as farthest_point_sample_cuda

def farthest_point_sample(xyz, npoint, device):
    """
    Input:
        xyz: pointcloud data, [N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint]
    """
    if torch.cuda.is_available():
        if len(xyz) > 5 * npoint:
            idx = np.random.permutation(len(xyz))[:5 * npoint]
            torch_xyz = torch.tensor(xyz[idx]).float().to(device).reshape(1, -1, 3)
            torch_idx = farthest_point_sample_cuda(torch_xyz, npoint)
            torch_idx = torch_idx.cpu().numpy().reshape(-1)
            idx = idx[torch_idx]
        else:
            torch_xyz = torch.tensor(xyz).float().to(device).reshape(1, -1, 3)
            torch_idx = farthest_point_sample_cuda(torch_xyz, npoint)
            idx = torch_idx.reshape(-1).cpu().numpy()
        return idx
    else:
        print('FPS on CPU: use random sampling instead')
        idx = np.random.permutation(len(xyz))[:npoint]
        return idx

def OBB(x):
    '''
    transform x to the Oriented Bounding Box frame
    x: [N, 3]
    obb_x: [N, 3]
    '''
    x = deepcopy(x)
    #pca X
    n = x.shape[0]
    trans = x.mean(axis=0)
    x -= trans  
    C = np.dot(x.T, x) / (n - 1)
    # Eigen decomposition
    eigen_vals, eigen_vecs = np.linalg.eig(C)
    R = np.eye(3)
    max_ind = np.argmax(eigen_vals)        
    min_ind = np.argmin(eigen_vals)        
    R[:, 0] = eigen_vecs[:, max_ind]
    R[:, 2] = eigen_vecs[:, min_ind]

    R[:, 1] = np.cross(R[:, 2], R[:, 0])
    R[:, 1] = R[:, 1] / np.linalg.norm(R[:, 1])
    rotated_x = x @ R

    scale = 1     
    bbox_len = scale * (rotated_x.max(axis=0) - rotated_x.min(axis=0))
    #only use x-axis length
    normalized_x = rotated_x / bbox_len[0]

    T = normalized_x.mean(axis=0)
    obb_x = normalized_x - T
    record = {'rotation':R, 'translation': trans[:,None]+((R@T[:,None])* bbox_len[0]), 'scale': bbox_len[0]}  #R^(-1)(X-T)/s = obb_x
    return obb_x, record

def generate_sim_data(path):
    cloud_dict = np.load(path, allow_pickle=True)['all_dict'].item()
    mano_pose = np.array(cloud_dict['hand_pose']['mano_pose'])
    mano_trans = np.array(cloud_dict['hand_pose']['mano_trans'])
    mano_beta = np.array(cloud_dict['hand_pose']['mano_beta'])
    mano_layer_right = ManoLayer(
            mano_root='/home/jiayichen/manopth/mano/models', side='right', use_pca=False, ncomps=45,
            flat_hand_mean=True)
    _, hand_kp = mano_layer_right.forward(th_pose_coeffs=torch.FloatTensor(np.array(mano_pose).reshape(1, -1)),
                            th_trans=torch.FloatTensor(mano_trans).reshape(1, -1),
                            th_betas=torch.FloatTensor(mano_beta).reshape(1, -1))
    hand_kp = hand_kp.cpu().data.numpy()[0]/1000      #[21,3]

    cam = cloud_dict['points']
    label = cloud_dict['labels']
    if len(cam) == 0:
        return None, None 

    #shuffle
    n = cam.shape[0]
    perm = np.random.permutation(n)
    cam = cam[perm]
    label = label[perm]

    hand_idx = np.where(label == 1)[0]
    if len(hand_idx) == 0:
        return None, None 
    else:
        hand_pcd = cam[hand_idx]
        sample_idx = farthest_point_sample(hand_pcd, 1024, "cuda:0")
        hand_pcd = hand_pcd[sample_idx]
    obb_pc, record = OBB(hand_pcd)
    gt = np.matmul(hand_kp - record['translation'].transpose(-1,-2), record['rotation']) / record['scale'] 
    if record['scale'] < 0.001:
        return None, None 
    # os.makedirs('vis', exist_ok=True)
    # id = path[-15:-4]
    # np.savetxt(f'vis/{id}_pc.txt', obb_pc)
    # np.savetxt(f'vis/{id}_anotherpc.txt', another)
    # np.savetxt(f'vis/{id}_gt.txt', gt)
    return torch.from_numpy(obb_pc).float(), record['scale'], torch.from_numpy(gt).float()



class SimDataset(data.Dataset):
    def __init__(self, train=True):
        mode = 'train' if train else 'test'
        self.all_file_list = []
        obj_category_list = ['bottle_sim', 'bowl_sim', 'car_sim']
        for category in obj_category_list:
            read_folder = pjoin('/data/h2o_data/new_sim_dataset/render', 'preproc', category, 'seq')
            splits_path = pjoin('/data/h2o_data/new_sim_dataset/render', "splits", category, 'seq')
            with open(pjoin(read_folder, splits_path, mode+'.txt'), 'r') as f:
                lines = f.readlines()
                file_list = [pjoin(read_folder, i.strip()) for i in lines]
                self.all_file_list.extend(file_list)
        self.invalid_dict = {}
        self.len = len(self.all_file_list)
        print('data number: ', self.len)

    def __getitem__(self, index):
        pose_pth = self.all_file_list[index]
        if index not in self.invalid_dict:
            data = generate_sim_data(pose_pth)
            if data[0] is None:
                self.invalid_dict[index] = True
        if index in self.invalid_dict:
            return self.__getitem__((index + 1) % self.len)
        return data

    def __len__(self):
        return self.len
