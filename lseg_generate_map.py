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
import rospy
from util.rosutil import RosCom
import time


rospy.init_node('lidar_lseg')
roscom = RosCom()

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
    kitti_path = '../../KITTI/dataset/sequences/' + sequence + '/'
    # Custom utils
    kitti_util = KittiUtil(kitti_path+'calib.txt')



    # image
    pcd_map = []
    pcd_feat_map = []
    pcd_color_map = []
    for frame in range(350):
        img = kitti_util.load_img(kitti_path + 'image_2/' + str(frame).zfill(6) + '.png')  # HxWxC


        pcd = kitti_util.load_pcd(kitti_path + 'velodyne/' + str(frame).zfill(6) + '.bin')

        # Publish to loam
        roscom.publish_points(pcd[:, :3])

        mask = np.ones(pcd.shape[0])

        # Only take lidar points that are on positive side of camera plane
        mask[np.where(pcd[:,0]<1)[0]] = 0

        # x = Pi * T * X  | Lidar to camera projection
        pts_cam = kitti_util.velo_to_cam(pcd, 2)
        pts_cam = np.array(pts_cam, dtype=np.int32)  # 0th column is img width


        #  Filter pts_cam to get only the point in image limits
        # There should be a one liner to do this.
        mask[np.where(pts_cam[:,0] >=img.shape[1])[0]] = 0
        mask[np.where(pts_cam[:,0] <0)[0]] = 0
        mask[np.where(pts_cam[:,1] >=img.shape[0])[0]] = 0
        mask[np.where(pts_cam[:,1] <0)[0]] = 0

        # mask_idx are indexes we are considering, where mask is 1
        mask_idx = np.where([mask>0])[1]  # Somehow this returns a tuple of len 2

        # Project lidar points on camera plane
        # img[pts_cam[mask_idx, 1], pts_cam[mask_idx, 0], :] = (255, 0, 0)
        # plt.imshow(img)
        # plt.show()

        # Getting image features
        img_feat = get_img_feat(img, net)
        

        # Features in PCD Space
        pcd_feat = np.zeros((pcd.shape[0], 512), dtype=np.float32)
        img_feat_np = img_feat.detach().cpu().numpy()[0]
        img_feat_np = np.transpose(img_feat_np, (1,2,0))
        img_feat_np = cv2.resize(img_feat_np, (img.shape[1], img.shape[0]))
        pcd_feat[mask_idx] =  img_feat_np[pts_cam[mask_idx, 1], pts_cam[mask_idx, 0], :]
        pcd_feat = torch.tensor(pcd_feat)

        # Transfrom wrt odom
        if frame%5==0:
            pcd_map.append((roscom.odom @ pcd[mask_idx].T).T)
            pcd_feat_map.append(pcd_feat[mask_idx])
            pcd_color_map.append(img[pts_cam[mask_idx, 1], pts_cam[mask_idx, 0], :])
            
    pcd_feat_map = torch.vstack(tuple(pcd_feat_map)) 
    pcd_color_map = np.vstack(tuple(pcd_color_map)) 
    pcd_map = np.vstack(pcd_map)

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

        visualize_multiple_pcd([pcd_map[:,:3], pcd_map[similarity>0.85][:,:3]], [pcd_color_map, None])

    breakpoint()
    np.save('data/'+ sequence + '_pcd_feat_map.npy', pcd_feat_map)
    np.save('data/'+ sequence + '_pcd_color_map.npy', pcd_color_map)
    np.save('data/'+ sequence + '_pcd_map.npy', pcd_map)

if __name__=='__main__':
    main()