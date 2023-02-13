from util.o3d_util import visualize_multiple_pcd, pick_points, create_o3d_pcd
from util.kitti_util import KittiUtil
from matplotlib import pyplot as plt
import numpy as np
from lseg import LSegNet
from pathlib import Path
from typing import Union
import tyro
from dataclasses import dataclass
import torch
import random
import cv2
import clip
import rospy
from util.rosutil import RosCom
import time
import fast_pytorch_kmeans as fpk
import open3d as o3d
from typing import List, Literal, Union
import os
import json

torch.cuda.empty_cache()

def get_new_pallete(num_colors: int) -> torch.Tensor:
    """Create a color pallete given the number of distinct colors to generate.
    Args:
        num_colors (int): Number of colors to include in the pallete
    Returns:
        torch.Tensor: Generated color pallete of shape (num_colors, 3)
    """
    pallete = []
    for j in range(num_colors):
        lab = j
        r, g, b = 0, 0, 0
        i = 0
        while lab > 0:
            r |= ((lab >> 0) & 1) << (7 - i)
            g |= ((lab >> 1) & 1) << (7 - i)
            b |= ((lab >> 2) & 1) << (7 - i)
            i = i + 1
            lab >>= 3
        pallete.append([r, g, b])
    return torch.Tensor(pallete).float() / 255.0



@dataclass
class ProgramArgs:
    checkpoint_path: str =  ".." + "/lseg-minimal" + "/examples" +"/checkpoints" + "/lseg_minimal_e200.ckpt"
    backbone: str = "clip_vitl16_384"
    num_features: int = 256
    arch_option: int = 0
    block_depth: int = 0
    activation: str = "lrelu"
    sequence: str = "00"
    crop_size: int = 480
    # query_image: Union[str, Path] = (
    #     Path(__file__).parent.parent / "images" / "teddybear.jpg"
    # )
    # prompt: str = "teddy"
    # Clustering parameters
    num_clusters: int = 8
    distance_type: Literal["euclidean", "cosine"] = "cosine"



def trajectory_tracking(pcd, filename, res=100, delay=0.01):
    '''
    Args: 
        pcd: o3d.geometry.pointcloud
        filename: str
        res: int
        delay: float
    filename -> name of the json file which is an array of copy pasted waypoints from o3d visualizer
    res -> interpolation resolution
    delay -> delay for visualization
    '''
    with open(filename, 'r') as f:
        data_arr = json.load(f)

    # Interpolate between two waypoints
    full_traj = []
    for i in range(len(data_arr)-1):
        wp1_raw  = data_arr[i]['trajectory'][0]
        wp2_raw  = data_arr[i+1]['trajectory'][0]
        wp1 = np.array([
            wp1_raw['front'],
            wp1_raw['lookat'],
            wp1_raw['up'],
            [wp1_raw['zoom'],wp1_raw['zoom'],wp1_raw['zoom']]
        ])
        wp2 = np.array([
            wp2_raw['front'],
            wp2_raw['lookat'],
            wp2_raw['up'],
            [wp2_raw['zoom'],wp2_raw['zoom'],wp2_raw['zoom']]
        ])
        wps = np.linspace(wp1, wp2, num=100)
        full_traj.append(wps)
    full_traj = np.vstack(full_traj)

    count = -1
    # Callback Function
    def track_callback(vis):
        nonlocal count
        count+=1
        ctr = vis.get_view_control()
        data = full_traj[count%len(full_traj)]
        
        
        ctr.set_front(data[0])
        ctr.set_lookat(np.array(data[1]))
        ctr.set_up(np.array(data[2]))
        ctr.set_zoom(np.array(data[3][0]))
        time.sleep(delay)
        return False

    o3d.visualization.draw_geometries_with_animation_callback([pcd], track_callback)

def main():
    # Lseg
    args = tyro.cli(ProgramArgs)
    sequence = args.sequence
    pcd_color_map = np.load('data/'+sequence+'_pcd_color_map.npy')
    pcd_map = np.load('data/'+sequence+'_pcd_map.npy')
    pcd = create_o3d_pcd(pcd_map[:, :3], pcd_color_map)

    o3d.visualization.draw_geometries([pcd])
    trajectory_tracking(pcd, 'data/kmean_viz_traj.json')


    pcd2 = o3d.io.read_point_cloud('data/kmeans.pcd')
    o3d.visualization.draw_geometries([pcd2])
    trajectory_tracking(pcd2, 'data/kmean_viz_traj.json')



if __name__=='__main__':
    main()