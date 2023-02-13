import torch
from torch.utils.data import Dataset
import numpy as np


class LidarLsegDataset:
    def __init__(self, map_path, device='cpu') -> None:
        self.pcd_feat_map = np.load(map_path + 'pcd_feat_map.npy')
        self.pcd_color_map = np.load(map_path + 'pcd_color_map.npy')
        self.pcd_map = np.load(map_path + 'pcd_map.npy')
        self.device = device
        
        assert self.pcd_feat_map.shape[0] == self.pcd_color_map.shape[0] == self.pcd_map.shape[0], 'Incorrect Dataset!'


    def __len__(self):
        return self.pcd_feat_map.shape[0]

    def __getitem__(self, idx):
        point = torch.tensor(self.pcd_map[idx][:3], device=self.device, dtype=torch.float32)
        feat = torch.tensor(self.pcd_feat_map[idx], device=self.device, dtype=torch.float32)
        return point, feat
        