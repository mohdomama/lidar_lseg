from util.o3d_util import visualize_multiple_pcd
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
import yaml
import time
import os
from tqdm import tqdm


torch.cuda.empty_cache()


@dataclass
class ProgramArgs:
    checkpoint_path: str = "data/lseg_minimal_e200.ckpt"
    kitti_dataset: str = "data/SEMANTIC-KITTI-DATASET/"
    sequence: str = "08"
    semantic_kitti_api_config: str = 'packages/semantic-kitti-api/config/semantic-kitti.yaml'
    semantic_preds_dir: str = 'data/SemanticSeg/SemanticKITTI/lseg_text/'

    # Needed for LSEG
    backbone: str = "clip_vitl16_384"
    num_features: int = 256
    arch_option: int = 0
    block_depth: int = 0
    activation: str = "lrelu"
    crop_size: int = 480


def get_img_feat(img, net):
    '''
    img -> RGB (0, 255)
    '''
    # Load the input image
    with torch.no_grad():
        img = cv2.resize(img, (640, 480))
        img = torch.from_numpy(img).float() / 255.0
        img = img[..., :3]  # drop alpha channel, if present
        img = img.cuda()
        img = img.permute(2, 0, 1)  # C, H, W
        img = img.unsqueeze(0)  # 1, C, H, W

        # Extract per-pixel CLIP features (1, 512, H // 2, W // 2)
        img_feat = net.forward(img)
        # Normalize features (per-pixel unit vectors)
        img_feat_norm = torch.nn.functional.normalize(img_feat, dim=1)
    return img_feat_norm


Classes_Prompts = [
    "other", "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist", "motorcyclist", "road", "parking", "sidewalk", "other-ground", "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign"

]


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

    cosine_similarity = torch.nn.CosineSimilarity(dim=2)

    kitti_config_data = yaml.safe_load(open(args.semantic_kitti_api_config, 'r'))
    # kitti_config_data['color_map'][kitti_config_data['learning_map_inv'][pred]]

    color_map = []
    for class_idx in sorted(kitti_config_data['learning_map_inv'].keys()):
        print(class_idx)
        color_map.append(kitti_config_data['color_map'][kitti_config_data['learning_map_inv'][class_idx]])
    color_map = np.array(color_map)


    sequence = args.sequence
    kitti_path = args.kitti_dataset + 'sequences/' + sequence + '/'
    out_path = args.semantic_preds_dir + 'sequences/' + sequence + '/'
    # Custom utils
    kitti_util = KittiUtil(kitti_path+'calib.txt')

    # image
    pcd_map = []
    pcd_feat_map = []
    pcd_color_map = []
    clip_text_encoder = net.clip_pretrained.encode_text

    for frame in tqdm(range(len(os.listdir(kitti_path+'velodyne/')))):
        img = kitti_util.load_img(
            kitti_path + 'image_2/' + str(frame).zfill(6) + '.png')  # HxWxC

        pcd = kitti_util.load_pcd(
            kitti_path + 'velodyne/' + str(frame).zfill(6) + '.bin')

        mask = np.ones(pcd.shape[0], dtype=np.float32)
        preds_all = -1 * np.ones(pcd.shape[0], dtype=np.float32) 

        # Only take lidar points that are on positive side of camera plane
        mask[np.where(pcd[:, 0] < 1)[0]] = 0

        # x = Pi * T * X  | Lidar to camera projection
        pts_cam = kitti_util.velo_to_cam(pcd, 2)
        pts_cam = np.array(pts_cam, dtype=np.int32)  # 0th column is img width

        #  Filter pts_cam to get only the point in image limits
        # There should be a one liner to do this.
        mask[np.where(pts_cam[:, 0] >= img.shape[1])[0]] = 0
        mask[np.where(pts_cam[:, 0] < 0)[0]] = 0
        mask[np.where(pts_cam[:, 1] >= img.shape[0])[0]] = 0
        mask[np.where(pts_cam[:, 1] < 0)[0]] = 0

        # mask_idx are indexes we are considering, where mask is 1
        # Somehow this returns a tuple of len 2
        mask_idx = np.where([mask > 0])[1]

        # Project lidar points on camera plane
        # img[pts_cam[mask_idx, 1], pts_cam[mask_idx, 0], :] = (255, 0, 0)
        # plt.imshow(img)
        # plt.show()

        # Getting image features
        img_feat = get_img_feat(img, net)

        # Features in PCD Space
        pcd_feat = np.zeros((pcd.shape[0], 512), dtype=np.float32)
        img_feat_np = img_feat.detach().cpu().numpy()[0]
        img_feat_np = np.transpose(img_feat_np, (1, 2, 0))
        img_feat_np = cv2.resize(img_feat_np, (img.shape[1], img.shape[0]))
        pcd_feat[mask_idx] = img_feat_np[pts_cam[mask_idx, 1],
                                         pts_cam[mask_idx, 0], :]
        pcd_feat = torch.tensor(pcd_feat)

        pcd = torch.tensor(pcd[mask_idx])
        pcd_feat = torch.tensor(pcd_feat[mask_idx])

        text_feats_all = []
        for prompt in Classes_Prompts:
            prompt = clip.tokenize(prompt)
            prompt = prompt.cuda()
            text_feat = clip_text_encoder(prompt)  # 1, 512
            text_feat_norm = torch.nn.functional.normalize(
                text_feat, dim=1)
            text_feats_all.append(text_feat_norm[0])
        text_feats_all = torch.vstack(text_feats_all)

        similarity = cosine_similarity(pcd_feat.unsqueeze(
            0), text_feats_all.unsqueeze(1).cpu())
        
        pred = similarity.argmax(axis=0)

        preds_all[mask_idx] = pred
        
        np.save(out_path+'predictions/' + str(frame).zfill(6) + '.npy', preds_all)
        np.save(out_path+'masks/' + str(frame).zfill(6) + '.npy', mask)

        

if __name__ == '__main__':
    main()
