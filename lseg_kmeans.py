from util.o3d_util import visualize_multiple_pcd, pick_points, create_o3d_pcd
from util.kitti_util import KittiUtil
from matplotlib import pyplot as plt
import numpy as np
np.random.seed(1)
from lseg import LSegNet
from pathlib import Path
from typing import Union
import tyro
from dataclasses import dataclass
import torch
torch.manual_seed(2)
import random
random.seed(0)
import cv2
import clip
import rospy
from util.rosutil import RosCom
import time
import fast_pytorch_kmeans as fpk
import open3d as o3d
from typing import List, Literal, Union

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
    
    
    # Initialize a KMeans clustering module
    kmeans = fpk.KMeans(
        n_clusters=args.num_clusters, mode=args.distance_type, verbose=1, 
    )

    # Cluster the map embeddings
    clusters = kmeans.fit_predict(pcd_feat_map)

    # Visualize
    # Visualize
    pallete = get_new_pallete(args.num_clusters + 1)
    pallete = pallete[1:].cuda()  # drop the first color (black) -- hard to visualize
    
    while True: 
        perm = torch.randperm(pallete.size()[0])
        # perm = [1, 0, 4, 2, 3]
        print(perm)
        pallete = pallete[perm]
        pcd = create_o3d_pcd(pcd_map[:, :3], pcd_color_map)
        map_colors = np.asarray(pcd.colors)
        cluster_colors = torch.from_numpy(map_colors.copy()).float().cuda()
        cluster_colors = pallete[clusters]
        cluster_colors = cluster_colors.detach().cpu().numpy()
        map_colors = 0.5 * map_colors + 0.5 * cluster_colors

        # Assign colors and display GUI
        pcd.colors = o3d.utility.Vector3dVector(map_colors)
        o3d.io.write_point_cloud('data/kmeans.pcd', pcd)
        o3d.visualization.draw_geometries([pcd])


if __name__=='__main__':
    main()