from util.o3d_util import visualize_multiple_pcd, pick_points
from util.kitti_util import KittiUtil
from matplotlib import pyplot as plt
import numpy as np
from lseg import LSegNet
from pathlib import Path
from typing import Union
import tyro
from dataclasses import dataclass
import torch
import cv2
import clip
import rospy
from util.rosutil import RosCom
import time

from util.transforms import build_se3_transform, transform_numpy_pcd

torch.cuda.empty_cache()

import rospy


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


def get_img_feat(img, net):
    '''
    img -> RGB (0, 255)
    '''
    # Load the input image pcd_map[similarity>thresh][:,:3]
    with torch.no_grad():
        print(f"Original image shape: {img.shape}")
        img = cv2.resize(img, (640, 480))
        img = torch.from_numpy(img).float() / 255.0
        img = img[..., :3]  # drop alpha channel, if present
        img = img.cuda()
        img = img.permute(2, 0, 1)  # C, H, W
        img = img.unsqueeze(0)  # 1, C, H, W
        print(f"Image shape: {img.shape}")

        # Extract per-pixel CLIP features (1, 512, H // 2, W // 2)
        img_feat = net.forward(img)
        # Normalize features (per-pixel unit vectors)
        img_feat_norm = torch.nn.functional.normalize(img_feat, dim=1)
        print(f"Extracted CLIP image feat: {img_feat_norm.shape}")
    return img_feat_norm




def main():
    # Lseg
    args = tyro.cli(ProgramArgs)
    net = LSegNet(
        backbone=args.backbone,
        features=args.num_features,
        crop_size=args.crop_size,
        arch_option=args.arch_option,
        block_depth=args.block_depth,
        activation=args.activation,
    )

    net.load_state_dict(torch.load(str(args.checkpoint_path)))
    net.eval()
    net.cuda()

    cosine_similarity = torch.nn.CosineSimilarity(dim=1)

    sequence = args.sequence
    # pcd_feat_map = torch.tensor(np.load('data/'+sequence+'_pcd_feat_map.npy'))
    # pcd_color_map = np.load('data/'+sequence+'_pcd_color_map.npy')
    # pcd_map = np.load('data/'+sequence+'_pcd_map.npy')
    
    map_path = 'data/lego_loam_map1/'
    pcd_feat_map = torch.tensor(np.load(map_path + 'pcd_feat_map.npy'))
    pcd_color_map = np.load(map_path + 'pcd_color_map.npy')
    pcd_map = np.load(map_path + 'pcd_map.npy')
    
    thresh = 0.77  

    roscom  = RosCom()

    while True:
        # Text feats 
        prompt = input('Enter Text Prompt: ')
        if prompt=='exit':
            break
        clip_text_encoder = net.clip_pretrained.encode_text
        prompt = clip.tokenize(prompt)
        prompt = prompt.cuda()
        text_feat = clip_text_encoder(prompt)  # 1, 512
        text_feat_norm = torch.nn.functional.normalize(text_feat, dim=1).detach().cpu()
        # Distance in PCD Space
        similarity = cosine_similarity(pcd_feat_map, text_feat_norm).cpu().numpy()
        # similarity = similarity * mask

        # visualize_multiple_pcd([pcd_map[:,:3], pcd_map[similarity>thresh][:,:3]], [None, None])

        road = pcd_map[similarity>thresh]
        not_road = pcd_map[similarity<thresh]
        
        map_vel_tf = build_se3_transform([-48, 0, -2,  0.106, -1.25, 0.037])
        
        # road_vel = transform_numpy_pcd(road[:,:3], map_vel_tf)
        # not_road_vel = transform_numpy_pcd(not_road[:,:3], map_vel_tf)
        
        road_vel = road
        not_road_vel = not_road

        # x_mask = abs(road_vel[:, 0]) < 100
        # y_mask = abs(road_vel[:, 1]) < 25
        z_mask = road_vel[:, 2] < 0
        # mask = x_mask & y_mask & z_mask
        road_vel = road_vel[z_mask]

        # x_mask = abs(not_road_vel[:, 0]) < 100
        # y_mask = abs(not_road_vel[:, 1]) < 25
        z_mask = not_road_vel[:, 2] < 0
        # mask = x_mask & y_mask & z_mask
        not_road_vel = not_road_vel[z_mask]
        
        roscom.publish_road(road_vel[:, :3])
        roscom.publish_not_road(not_road_vel[:, :3])

        visualize_multiple_pcd([not_road_vel[:, :3], road_vel[:, :3]])

        breakpoint()


if __name__=='__main__':
    main()