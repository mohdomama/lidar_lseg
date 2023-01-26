from util.kitti_util import KittiUtil
from matplotlib import pyplot as plt
import numpy as np
from lseg import LSegNet
import tyro
from dataclasses import dataclass, field
import torch
import cv2
import clip
import yaml
import os
from tqdm import tqdm
from sklearn.metrics import jaccard_score, confusion_matrix, accuracy_score, classification_report
from test_clipseg import upsample_feat_vec
from clipseg import CLIPDensePredT
from torchvision import transforms


torch.cuda.empty_cache()


@dataclass
class ProgramArgs:
    checkpoint_path: str = "data/clipseg/rd64-uni.pth"
    kitti_dataset: str = "data/SEMANTIC-KITTI-DATASET/"
    sequence: str = "08"
    semantic_kitti_api_config: str = 'packages/semantic-kitti-api/config/semantic-kitti.yaml'
    semantic_preds_dir: str = 'data/SemanticSeg/SemanticKITTI/lseg_text/'

    crop: bool = field(default=False)
    width_min: int = 293
    width_max: int = 933

    filter_distance: bool = field(default=False)
    distance_limit: int = 10
    

    # # Needed for LSEG
    # backbone: str = "clip_vitl16_384"
    # num_features: int = 256
    # arch_option: int = 0
    # block_depth: int = 0
    # activation: str = "lrelu"
    # crop_size: int = 480


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
    "other", "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist", 
    "motorcyclist", "road", "parking", "sidewalk", "other-ground", "building", "fence", 
    "vegetation", "trunk", "terrain", "pole", "traffic-sign"

]

Lseg_Prompts = [
    "other",            # 0            
    "car",              # 1    
    "bicycle",          # 2        
    "motorcycle",       # 3            
    "person",           # 4        
    "road",             # 5        
    "sidewalk",         # 6            
    "building",         # 7           
    "vegetation",       # 8           
    "pole",             # 9        
    ]

LABEL_MAP_LSEG_DICT = {
  0: 0,       
  1: 1,       
  2: 2,       
  3: 3,       
  4: 1,       
  5: 1,       
  6: 4,       
  7: 2,       
  8: 3,       
  9: 5,       
  10: 5,       
  11: 6,       
  12: 8,      
  13: 7,      
  14: 7,      
  15: 8,      
  16: 8,      
  17: 8,      
  18: 9,      
  19: 9,      
}

LABEL_MAP_LSEG = np.zeros(20, dtype=np.int16)
for key in LABEL_MAP_LSEG_DICT.keys():
    LABEL_MAP_LSEG[key] = LABEL_MAP_LSEG_DICT[key]


def main():
    # ClipSeg
    args = tyro.cli(ProgramArgs)
    model = CLIPDensePredT(version="ViT-B/16", reduce_dim=64)
    model.eval()

    model.load_state_dict(
        torch.load(str(args.checkpoint_path), map_location=torch.device("cpu")),
        strict=False,
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((352, 352)),
        ]
    )


    kitti_config_data = yaml.safe_load(open(args.semantic_kitti_api_config, 'r'))
    # kitti_config_data['color_map'][kitti_config_data['learning_map_inv'][pred]]

    color_map = []
    for class_idx in sorted(kitti_config_data['learning_map_inv'].keys()):
        color_map.append(kitti_config_data['color_map'][kitti_config_data['learning_map_inv'][class_idx]])
    color_map = np.array(color_map)


    sequence = args.sequence
    kitti_path = args.kitti_dataset + 'sequences/' + sequence + '/'
    out_path = args.semantic_preds_dir + 'sequences/' + sequence + '/'
    os.makedirs(out_path+'predictions/', exist_ok=True)
    os.makedirs(out_path+'masks/', exist_ok=True)
    # Custom utils
    kitti_util = KittiUtil(kitti_path+'calib.txt')

    
    # For lables
    # make lookup table for mapping
    class_remap = kitti_config_data["learning_map"]
    maxkey = max(class_remap.keys())
    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(class_remap.keys())] = list(class_remap.values())

    labels_seq = []
    preds_seq = []
    unqlabel_seq = []
    
    for frame in tqdm(range(len(os.listdir(kitti_path+'velodyne/')))):
        img = kitti_util.load_img(
            kitti_path + 'image_2/' + str(frame).zfill(6) + '.png')  # HxWxC

        pcd = kitti_util.load_pcd(
            kitti_path + 'velodyne/' + str(frame).zfill(6) + '.bin')

        if args.crop:
            img = img[:, args.width_min:args.width_max, :]

        label_file = kitti_path + 'labels/' + str(frame).zfill(6) + '.label'

        # Reading label file
        label = np.fromfile(label_file, dtype=np.int32)
        label = label.reshape((-1))  # reshape to vector
        label = label & 0xFFFF       # get lower half for semantics
        label = remap_lut[label]       # remap to xentropy format

        mask = np.ones(pcd.shape[0], dtype=np.float32)
        preds_all = -1 * np.ones(pcd.shape[0], dtype=np.float32) 

        # Only take lidar points that are on positive side of camera plane
        mask[np.where(pcd[:, 0] < 1)[0]] = 0
        dis = np.linalg.norm(pcd[:, :3], axis=1)

        if args.filter_distance:
            mask[np.where(dis>args.distance_limit)[0]] = 0


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
        # Clipseg code gives similarity scored directly. 
        preds = model(transform(img).unsqueeze(0).repeat(len(Lseg_Prompts), 1, 1, 1), Lseg_Prompts)[0]

        # Features in PCD Space
        preds = torch.nn.functional.interpolate(preds, (img.shape[0], img.shape[1]), mode="bilinear", align_corners=True)
        preds = preds.argmax(axis=0)[0]


        if args.crop:
            pcd_preds = preds[pts_cam[mask_idx, 1], pts_cam[mask_idx, 0]-args.width_min, :]
        else:
            pcd_preds = preds[pts_cam[mask_idx, 1], pts_cam[mask_idx, 0], :]

        pcd = torch.tensor(pcd[mask_idx])

        # text_feats_all = []
        # for prompt in Lseg_Prompts:
        #     prompt = clip.tokenize(prompt)
        #     prompt = prompt.cuda()
        #     text_feat = clip_text_encoder(prompt)  # 1, 512
        #     text_feat_norm = torch.nn.functional.normalize(
        #         text_feat, dim=1)
        #     text_feats_all.append(text_feat_norm[0])
        # text_feats_all = torch.vstack(text_feats_all)

        # similarity = cosine_similarity(pcd_feat.unsqueeze(
        #     0), text_feats_all.unsqueeze(1))
        
        pred = pcd_preds.detach().cpu().numpy()

        preds_all[mask_idx] = pred
        label = label[mask_idx]
        label = LABEL_MAP_LSEG[label]

        labels_seq.extend(label.tolist())
        preds_seq.extend(pred.tolist())

        unqlabel = np.unique(label)
        unqlabel = unqlabel[unqlabel!=0]

        unqlabel_seq.extend(unqlabel.tolist())
        unqlabel_seq = np.unique(unqlabel_seq).tolist()
        
        np.save(out_path+'predictions/' + str(frame).zfill(6) + '.npy', preds_all)
        np.save(out_path+'masks/' + str(frame).zfill(6) + '.npy', mask)

    unqlabel_seq = np.sort(unqlabel_seq).tolist()
    labels_seq = np.array(labels_seq)
    preds_seq = np.array(preds_seq)
    miou = jaccard_score(y_true=labels_seq, y_pred=preds_seq, average='macro', labels=unqlabel_seq)
    wmiou = jaccard_score(y_true=labels_seq, y_pred=preds_seq, average='weighted', labels=unqlabel_seq)
    iou_all = jaccard_score(y_true=labels_seq, y_pred=preds_seq, average=None, labels=unqlabel_seq)
    
    accuracy = accuracy_score(y_true=labels_seq, y_pred=preds_seq, )
    accuracy = accuracy_score(y_true=labels_seq[labels_seq!=0], y_pred=preds_seq[labels_seq!=0], )
    report = classification_report(y_true=labels_seq, y_pred=preds_seq, labels=unqlabel_seq)

    print('\n')
    for i in range(len(iou_all)):
        print(Lseg_Prompts[unqlabel_seq[i]], '\t', iou_all[i])

    print()
    print('MIOU: ', miou)
    print('WMIOU: ', wmiou)
    print('Accuracy: ', accuracy)

    print()
    print(report)


    # cm = confusion_matrix(y_true=labels_seq, y_pred=preds_seq)
    cm_pred = confusion_matrix(y_true=labels_seq, y_pred=preds_seq, normalize='pred')
    cm_true = confusion_matrix(y_true=labels_seq, y_pred=preds_seq, normalize='true')
    cm = confusion_matrix(y_true=labels_seq, y_pred=preds_seq)
    plt.matshow(cm_pred)
    plt.savefig(fname='data/'+sequence+'cm_pred.png', dpi=500)
    plt.matshow(cm_true)
    plt.savefig(fname='data/'+sequence+'cm_true.png', dpi=500)

if __name__ == '__main__':
    main()
