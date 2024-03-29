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


torch.cuda.empty_cache()



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
    # Load the input image
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
    pcd_feat_map = torch.tensor(np.load('data/'+sequence+'_pcd_feat_map.npy'))
    pcd_color_map = np.load('data/'+sequence+'_pcd_color_map.npy')
    pcd_map = np.load('data/'+sequence+'_pcd_map.npy')
    while True:
        pts = pick_points(pcd_map[:, :3], pcd_color_map)
        query_feat = pcd_feat_map[pts]


        # Distance in PCD Space
        similarity = cosine_similarity(pcd_feat_map, query_feat).cpu().numpy()
        # similarity = similarity * mask

        visualize_multiple_pcd([pcd_map[:,:3], pcd_map[similarity>0.7][:,:3]], color_list=[pcd_color_map, None])



if __name__=='__main__':
    main()