from util.o3d_util import visualize_multiple_pcd
from util.kitti_util import KittiUtil
from matplotlib import pyplot as plt
import numpy as np
from lseg import LSegNet
from pathlib import Path
from typing import Union
import tyro
from dataclasses import dataclass, field
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
    one_shot_seqence: str = "00"
    semantic_kitti_api_config: str = 'packages/semantic-kitti-api/config/semantic-kitti.yaml'
    semantic_preds_dir: str = 'data/SemanticSeg/SemanticKITTI/lseg_text/'
    crop: bool = field(default=False)
    width_min: int = 293
    width_max: int = 933

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


    
    kitti_path = args.kitti_dataset + 'sequences/' + args.one_shot_seqence + '/'
    # Custom utils
    kitti_util = KittiUtil(kitti_path+'calib.txt')


    # +100 hack making lut bigger just in case there are unknown labels
    # make lookup table for mapping
    class_remap = kitti_config_data["learning_map"]
    maxkey = max(class_remap.keys())
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(class_remap.keys())] = list(class_remap.values())
    one_shot_vectors = torch.zeros((20,512), dtype=torch.float32).cuda()
    one_shot_vectors_counts = torch.zeros(20, dtype=torch.float32).cuda()

    print('Generating One Shot Vectors for Classes')
    frame_count = len(os.listdir(kitti_path+'velodyne/'))
    for frame in tqdm(range(frame_count)):
        tic = time.time()
        img = kitti_util.load_img(
            kitti_path + 'image_2/' + str(frame).zfill(6) + '.png')  # HxWxC

        pcd = kitti_util.load_pcd(
            kitti_path + 'velodyne/' + str(frame).zfill(6) + '.bin')

        if args.crop:
            img = img[:, args.width_min:args.width_max, :]

        label_file = kitti_path + 'labels/' + str(frame).zfill(6) + '.label'
        labels = np.fromfile(label_file, dtype=np.int32)
        labels = labels.reshape((-1))  # reshape to vector
        labels = labels & 0xFFFF       # get lower half for semantics
        labels = remap_lut[labels]       # remap to xentropy format

        toc = time.time()
        # print('Data Load Time: ', toc - tic)

        
        tic = time.time()
        mask = np.ones(pcd.shape[0], dtype=np.float32)
        preds_all = -1 * np.ones(pcd.shape[0], dtype=np.float32) 

        # Only take lidar points that are on positive side of camera plane
        mask[np.where(pcd[:, 0] < 1)[0]] = 0

        # x = Pi * T * X  | Lidar to camera projection
        pts_cam = kitti_util.velo_to_cam(pcd, 2)
        pts_cam = np.array(pts_cam, dtype=np.int32)  # 0th column is img width

        #  Filter pts_cam to get only the point in image limits
        # There should be a one liner to do this.
        if args.crop:
            mask[np.where(pts_cam[:, 0] >= args.width_max)[0]] = 0
            mask[np.where(pts_cam[:, 0] < args.width_min)[0]] = 0
            mask[np.where(pts_cam[:, 1] >= img.shape[0])[0]] = 0
            mask[np.where(pts_cam[:, 1] < 0)[0]] = 0
        else:
            mask[np.where(pts_cam[:, 0] >= img.shape[1])[0]] = 0
            mask[np.where(pts_cam[:, 0] < 0)[0]] = 0
            mask[np.where(pts_cam[:, 1] >= img.shape[0])[0]] = 0
            mask[np.where(pts_cam[:, 1] < 0)[0]] = 0

        # mask_idx are indexes we are considering, where mask is 1
        # Somehow this returns a tuple of len 2
        mask_idx = np.where([mask > 0])[1]
        toc = time.time()
        # print('Projection and Masking Time: ', toc - tic)
        # Project lidar points on camera plane
        # img[pts_cam[mask_idx, 1], pts_cam[mask_idx, 0], :] = (255, 0, 0)
        # plt.imshow(img)
        # plt.show()

        # Getting image features
        tic = time.time()
        img_feat = get_img_feat(img, net)
        toc = time.time()
        # print('Inference Time: ', toc-tic)
        
        # Features in PCD Space
        tic = time.time()
        img_feat = torch.nn.functional.interpolate(img_feat, [img.shape[0], img.shape[1]], mode="bilinear", align_corners=True)
        img_feat = torch.permute(img_feat[0], (1,2,0))
        # img_feat_np = cv2.resize(img_feat_np, (img.shape[1], img.shape[0]))
        if args.crop:
            pcd_feat = img_feat[pts_cam[mask_idx, 1], pts_cam[mask_idx, 0]-args.width_min, :]
        else:
            pcd_feat = img_feat[pts_cam[mask_idx, 1], pts_cam[mask_idx, 0], :]


        pcd = pcd[mask_idx]
        labels = labels[mask_idx]
        toc = time.time(); # print('Feature PCD Space Time: ', toc-tic)
        
        tic = time.time()
        labels = torch.tensor(labels)
        for label in range(20):
            idxs = torch.where(labels==label)[0]
            count_curr = idxs.shape[0]
            u_curr = torch.mean(pcd_feat[idxs], axis=0)
            count = one_shot_vectors_counts[label]
            if count_curr!=0:
                one_shot_vectors[label] = (one_shot_vectors[label]* count + u_curr*count_curr) / (count+count_curr)
            one_shot_vectors_counts[label] += count_curr
        
            
        #for feat_idx in range(pcd_feat.shape[0]):
        if one_shot_vectors.isnan().any():
            print('Got Nan!')
            breakpoint()
        #    label = labels[feat_idx]
        #    one_shot_vectors_counts[label]+=1
        #    count = one_shot_vectors_counts[label]
        #    one_shot_vectors[label] = one_shot_vectors[label] + (pcd_feat[feat_idx] - one_shot_vectors[label]) / count
        toc = time.time()
        # print('Label Agg. Time: ', toc - tic)

    # one_shot_vectors = torch.tensor(one_shot_vectors)
    
    # image
    pcd_map = []
    pcd_feat_map = []
    pcd_color_map = []
    clip_text_encoder = net.clip_pretrained.encode_text

    kitti_path = args.kitti_dataset + 'sequences/' + args.sequence + '/'
    # Custom utils
    kitti_util = KittiUtil(kitti_path+'calib.txt')

    out_path = args.semantic_preds_dir + 'sequences/' + args.sequence + '/'
    os.makedirs(out_path+'predictions/', exist_ok=True)
    os.makedirs(out_path+'masks/', exist_ok=True)
    print('Running Inference!')
    for frame in tqdm(range(len(os.listdir(kitti_path+'velodyne/')))):
        img = kitti_util.load_img(
            kitti_path + 'image_2/' + str(frame).zfill(6) + '.png')  # HxWxC

        pcd = kitti_util.load_pcd(
            kitti_path + 'velodyne/' + str(frame).zfill(6) + '.bin')
        
        if args.crop:
            img = img[:, args.width_min:args.width_max, :]


        mask = np.ones(pcd.shape[0], dtype=np.float32)
        preds_all = -1 * np.ones(pcd.shape[0], dtype=np.float32) 

        # Only take lidar points that are on positive side of camera plane
        mask[np.where(pcd[:, 0] < 1)[0]] = 0

        # x = Pi * T * X  | Lidar to camera projection
        pts_cam = kitti_util.velo_to_cam(pcd, 2)
        pts_cam = np.array(pts_cam, dtype=np.int32)  # 0th column is img width

        #  Filter pts_cam to get only the point in image limits
        # There should be a one liner to do this.
        if args.crop:
            mask[np.where(pts_cam[:, 0] >= args.width_max)[0]] = 0
            mask[np.where(pts_cam[:, 0] < args.width_min)[0]] = 0
            mask[np.where(pts_cam[:, 1] >= img.shape[0])[0]] = 0
            mask[np.where(pts_cam[:, 1] < 0)[0]] = 0
        else:
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

        ## Features in PCD Space
        #pcd_feat = np.zeros((pcd.shape[0], 512), dtype=np.float32)
        #img_feat_np = img_feat.detach().cpu().numpy()[0]
        #img_feat_np = np.transpose(img_feat_np, (1, 2, 0))
        #img_feat_np = cv2.resize(img_feat_np, (img.shape[1], img.shape[0]))
        #pcd_feat[mask_idx] = img_feat_np[pts_cam[mask_idx, 1],
        #                                 pts_cam[mask_idx, 0], :]
        #pcd_feat = torch.tensor(pcd_feat)

        #pcd = torch.tensor(pcd[mask_idx])
        #pcd_feat = torch.tensor(pcd_feat[mask_idx])

        img_feat = torch.nn.functional.interpolate(img_feat, [img.shape[0], img.shape[1]], mode="bilinear", align_corners=True)
        img_feat = torch.permute(img_feat[0], (1,2,0))
        # img_feat_np = cv2.resize(img_feat_np, (img.shape[1], img.shape[0]))
        if args.crop:
            pcd_feat = img_feat[pts_cam[mask_idx, 1], pts_cam[mask_idx, 0]-args.width_min, :]
        else:
            pcd_feat = img_feat[pts_cam[mask_idx, 1], pts_cam[mask_idx, 0], :]
        
        similarity = cosine_similarity(pcd_feat.unsqueeze(
            0), one_shot_vectors.unsqueeze(1))
        
        pred = similarity.argmax(axis=0)
        debug = True
        

        preds_all[mask_idx] = pred.detach().cpu().numpy()
        np.save(out_path+'predictions/' + str(frame).zfill(6) + '.npy', preds_all)
        np.save(out_path+'masks/' + str(frame).zfill(6) + '.npy', mask)
        

        

if __name__ == '__main__':
    main()
