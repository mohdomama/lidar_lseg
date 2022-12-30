import rospy
import sensor_msgs
from std_msgs.msg import String
from sensor_msgs.msg import Image
# import ros_numpy
import numpy as np
import atexit
import open3d as o3d
from o3d_util import visualize_multiple_pcd
from matplotlib import pyplot as plt
from transforms import build_se3_transform
from lseg import LSegNet
from pathlib import Path
from typing import Union
import tyro
from dataclasses import dataclass
import torch
import cv2
import clip
torch.cuda.empty_cache()


class LegoLoamKeyFrame:
    def __init__(self) -> None:
        rospy.Subscriber('/key_frame_alert', String, self.key_frame_alert)
        rospy.Subscriber('/zed/zed_node/left/image_rect_color', Image, self.image_handler)
        self.images = []
        self.image = None

    def key_frame_alert(self, msg):
        print('Received Key Frame Alert')
        print(type(self.image))
        self.images.append(self.image)

    def image_handler(self, msg):
        msg.__class__ = sensor_msgs.msg.Image
        img = ros_numpy.numpify(msg)
        img = img[:, :, [2,1,0]]
        self.image = img # Latest image
    
    def save(self):
        print('Saving Images')
        for i in range(len(self.images)):
            np.save('data/lego_loam_images/' + str(i).zfill(6), self.images[i])


def save_images():
    rospy.init_node('key_frame_listener')
    keyFrame = LegoLoamKeyFrame()
    atexit.register(keyFrame.save)
    rospy.spin()

#########################################################


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

def read_lego_loam_data_file(map_path, keyframe_id):
    filename = map_path + 'dump/' + keyframe_id + '/data'
    with open(filename, 'r') as f:
        lines = f.readlines()

    odom = []
    for i in range(7, 11):
        row = [float(x) for x in lines[i].strip().split()]
        odom.append(row)

    return np.array(odom)


def velo_to_cam(pcd, P, T):
        out =  (P @ T @ pcd.T).T
        out[:, 0] = out[:, 0] / out[:, 2]
        out[:, 1] = out[:, 1] / out[:, 2]
        out[:, 2] = out[:, 2] / out[:, 2]
        return out


def get_projection_matrices():
    # From rostopic echo /zed/zed_node/left/camera_info
    K = [336.3924560546875, 0.0, 307.7497863769531, 0.0, 336.3924560546875, 165.84719848632812, 0.0, 0.0, 1.0]
    K = np.array(K).reshape((3,3))
    K = np.hstack([K, np.array([0,0,0]).reshape(3,1)])

    T_camo_c = build_se3_transform([0.13, -0.35, -0.7, 0, 0, 0])
    P = K @ T_camo_c

    Tr = [0, -1,  0, 0, 
        0,  0, -1, 0, 
        1,  0,  0, 0]
    Tr = np.array(Tr).reshape(3,4)
    Tr_vel_camo = np.vstack([Tr, [0,0,0,1]])
    return P, Tr_vel_camo


def create_map():
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


    map_path = 'data/lego_loam_map1/'
    P, Tr_vel_camo = get_projection_matrices()

    T_static = build_se3_transform([0,0,0,0,0,1.57]) 

    pcd_map = []
    pcd_feat_map = []
    pcd_color_map = []
    for idx in range(797):
        keyframe_id = str(idx).zfill(6)

        odom = read_lego_loam_data_file(map_path, keyframe_id)
        odom = T_static @ odom
        
        pcd = o3d.io.read_point_cloud(map_path+'dump/' +  keyframe_id + '/' + 'cloud.pcd')
        pcd = np.asarray(pcd.points)
        print(pcd.shape)

        
        img = np.load(map_path+'lego_loam_images/' + keyframe_id + '.npy')

        ##################
        # Begin Projection
        
        pcd =  np.hstack([pcd, np.ones((pcd.shape[0], 1))])
        mask = np.ones(pcd.shape[0])
        mask[np.where(pcd[:,0]<1)[0]] = 0
        
        pts_cam = velo_to_cam(pcd, P, Tr_vel_camo)
        pts_cam = np.array(pts_cam, dtype=np.int32)  # 0th column is img height (different from kitti)

        # #  Filter pts_cam to get only the point in image limits
        # # There should be a one liner to do this.
        mask[np.where(pts_cam[:,0] >=img.shape[1])[0]] = 0
        mask[np.where(pts_cam[:,0] <0)[0]] = 0
        mask[np.where(pts_cam[:,1] >=img.shape[0])[0]] = 0
        mask[np.where(pts_cam[:,1] <0)[0]] = 0

        # mask_idx are indexes we are considering, where mask is 1
        mask_idx = np.where([mask>0])[1]  # Somehow this returns a tuple of len 2

        # # Project lidar points on camera plane
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


        pcd_map.append((odom @ pcd[mask_idx].T).T)
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
    np.save(map_path +  'pcd_feat_map.npy', pcd_feat_map)
    np.save(map_path +  'pcd_color_map.npy', pcd_color_map)
    np.save(map_path +  'pcd_map.npy', pcd_map)





    

    

if __name__=='__main__':
    # save_images()
    create_map()
    
    