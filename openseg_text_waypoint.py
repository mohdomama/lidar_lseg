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
import tqdm
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header



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

def build_text_embedding(categories, clip_model):

    run_on_gpu = torch.cuda.is_available()

    with torch.no_grad():
        all_text_embeddings = []
        print("Building text embeddings...")
        for category in tqdm(categories):
            texts = clip.tokenize(category)  # tokenize
            if run_on_gpu:
                texts = texts.cuda()
            text_embeddings = clip_model.encode_text(texts)  # embed with text encoder
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            all_text_embeddings.append(text_embedding)
        all_text_embeddings = torch.stack(all_text_embeddings, dim=1)

        if run_on_gpu:
            all_text_embeddings = all_text_embeddings.cuda()
    return all_text_embeddings.cpu().numpy().T


def main(pub):
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
    
    map_path = 'data/lego_loam_map2/'
    pcd_feat_map = torch.tensor(np.load(map_path + 'pcd_feat_map.npy'))
    pcd_color_map = np.load(map_path + 'pcd_color_map.npy')
    pcd_map = np.load(map_path + 'pcd_map.npy')
    
    thresh = 0.1
    clip_model, preprocess = clip.load("ViT-L/14@336px")
    while True:
        # Text feats 
        prompt = input('Enter Text Prompt: ')
        if prompt=='exit':
            break
        # clip_text_encoder = net.clip_pretrained.encode_text
        # prompt = clip.tokenize(prompt)
        # prompt = prompt.cuda()
        # text_feat = clip_text_encoder(prompt)  # 1, 512
        prompt = clip.tokenize(prompt)
        prompt = prompt.cuda()
        text_feat = clip_model.encode_text(prompt)
        text_feat = torch.nn.functional.normalize(text_feat, dim=0).detach().cpu()
        # Distance in PCD Space
        similarity = cosine_similarity(pcd_feat_map, text_feat).cpu().numpy()
        # similarity = similarity * mask

        pcd_highlight_map = np.copy(pcd_color_map)
        pcd_highlight_map[:, ] = np.array([255, 0, 0])
        visualize_multiple_pcd([pcd_map[:,:3], pcd_map[similarity>thresh][:,:3]], [pcd_color_map, pcd_highlight_map[similarity>thresh]])


        
        clusters = {}
        for point in pcd_map[similarity>thresh][:,:3]:
            clusters[tuple(point)] = [point]
        
        change = True
        while change:
            print(len(clusters.keys()))
            new_clusters = {}
            change = False
            for point in clusters.keys():
                assigned = False
                for new_point in new_clusters.keys():
                    if np.linalg.norm(np.array(point)-np.array(new_point)) < 20:
                        change = True
                        assigned = True
                        new_clusters[new_point].extend(clusters[point])
                        break
                if not assigned:
                    new_clusters[point] = clusters[point]
                
            
            
            clusters = new_clusters

            # Change cluster center value
            new_clusters = {}
            for point in clusters.keys():
                new_clusters[tuple(np.mean(clusters[point], axis=0 ))] = clusters[point]
            

            clusters = new_clusters

        max_len = 0
        selected_point = None
        for point in clusters.keys(): 
            if len(clusters[point]) > max_len:
                selected_point = point
                max_len = len(clusters[point])

        waypoint = selected_point


        # clusters_centers = []
        # best_matches = pcd_map[similarity.argsort()[::-1]][:100]
        # for pt in best_matches:
        #     if len(clusters_centers) == 0:
        #         clusters_centers.append(pt)
        #     else:
        #         if np.linalg.norm(clusters_centers[0]-pt)>10:
        #             clusters_centers.append(pt)
        #             break
        
        # cluster0, cluster1 = [], []
        # best_matches = pcd_map[similarity.argsort()[::-1]][:500]
        # for pt in best_matches:
        #     if np.linalg.norm(clusters_centers[0]-pt)<10:
        #         cluster0.append(pt)
        #         clusters_centers[0] = np.mean(cluster0, axis=0)
        #     if np.linalg.norm(clusters_centers[1]-pt)<10:
        #         cluster1.append(pt)
        #         clusters_centers[1] = np.mean(cluster1, axis=0)


        # if len(cluster0) > len(cluster1):
        #     waypoint = clusters_centers[0]
        # else:
        #     waypoint = clusters_centers[1]



        ######## Publishing Waypoints ###########
        # print(waypoint)

        wpt_msg = PoseStamped()
			
        wpt_msg.header = Header()
        wpt_msg.header.stamp = rospy.Time.now()
        wpt_msg.header.frame_id = 'map'


        wpt_msg.pose.position.x = waypoint[0]
        wpt_msg.pose.position.y = waypoint[1]
        wpt_msg.pose.position.z = 0

        # quat = quaternion_from_euler(0, 0, wpt[2])
        wpt_msg.pose.orientation.x = 0
        wpt_msg.pose.orientation.y = 0
        wpt_msg.pose.orientation.z = 0
        wpt_msg.pose.orientation.w = 1
        pub.publish(wpt_msg)

        breakpoint()


if __name__=='__main__':
    rospy.init_node('lseg')
    pub = rospy.Publisher("/goal", PoseStamped, queue_size=1, latch=True)
    main(pub)