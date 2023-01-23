from util.kitti_util import KittiUtil
from matplotlib import pyplot as plt
import numpy as np
import tyro
from dataclasses import dataclass
import torch
import cv2
import clip
import yaml
import time
import os
from tqdm import tqdm

import tensorflow.compat.v1 as tf
import tensorflow as tf2

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

torch.cuda.empty_cache()


@dataclass
class ProgramArgs:
    checkpoint_path: str = "data/openseg/exported_model/"
    kitti_dataset: str = "data/SEMANTIC-KITTI-DATASET/"
    sequence: str = "08"
    one_shot_seqence: str = "00"
    semantic_kitti_api_config: str = 'packages/semantic-kitti-api/config/semantic-kitti.yaml'
    semantic_preds_dir: str = 'data/SemanticSeg/SemanticKITTI/lseg_text/'

    # Needed for LSEG
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
    "other", "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist", "motorcyclist", "road", "parking", "sidewalk", "other-ground", "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign"

]

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

def upsample_feat_vec(feat, target_shape):
    return torch.nn.functional.interpolate(
        feat, target_shape, mode="bilinear", align_corners=True
    )


def main():
    args = tyro.cli(ProgramArgs)

    # Load a CLIP model
    print("Loading CLIP model...")
    clip_model, preprocess = clip.load("ViT-L/14@336px")

    # Build a dummy text embedding (needed for the TF OpenSeg model)
    print("Building a dummy text embedding (needed by the TF OpenSeg model...")
    text_prompts = ["sofa", "pillow", "other"]
    text_embedding = build_text_embedding(text_prompts, clip_model)
    num_words_per_category = 1
    text_embedding = tf.reshape(
        text_embedding, [-1, num_words_per_category, text_embedding.shape[-1]]
    )
    text_embedding = tf.cast(text_embedding, tf.float32)

    print('Creating Our Text Feat: ')
    text_feats = []
    for prompt in ['Road', 'car']:
        prompt_token = clip.tokenize(prompt)
        text_feat = clip_model.encode_text(prompt_token)
        text_feat = torch.nn.functional.normalize(text_feat, dim=1).detach().cpu().numpy()
        text_feats.append(text_feat)
    text_feats = tf.convert_to_tensor(text_feats)[:, 0,:]

    print("Loading OpenSeg model...")
    
    openseg_model = tf2.saved_model.load(
        args.checkpoint_path, tags=[tf.saved_model.tag_constants.SERVING]
    )

    # cosine_similarity = torch.nn.CosineSimilarity(dim=2)
    cosine_similarity = tf2.keras.losses.CosineSimilarity(axis=2, reduction=tf2.keras.losses.Reduction.NONE)
    cosine_similarity_a3 = tf2.keras.losses.CosineSimilarity(axis=3, reduction=tf2.keras.losses.Reduction.NONE)


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
    one_shot_vectors = torch.zeros((20,768), dtype=torch.float32)
    one_shot_vectors_counts = torch.zeros(20, dtype=torch.float32)


    # one_shot_vectors = torch.tensor(one_shot_vectors)
    
    # image
    one_shot_vectors = tf2.convert_to_tensor(one_shot_vectors.unsqueeze(1).numpy())
    text_feats = tf2.expand_dims(text_feats, 1)

    kitti_path = args.kitti_dataset + 'sequences/' + args.sequence + '/'
    # Custom utils
    kitti_util = KittiUtil(kitti_path+'calib.txt')

    out_path = args.semantic_preds_dir + 'sequences/' + args.sequence + '/'
    os.makedirs(out_path+'predictions/', exist_ok=True)
    os.makedirs(out_path+'masks/', exist_ok=True)

    print('Running Inference!')
    for frame in tqdm(range(len(os.listdir(kitti_path+'velodyne/')))):
        imgfile = kitti_path + 'image_2/' + str(frame).zfill(6) + '.png'
        img = kitti_util.load_img(imgfile)  # HxWxC

        with tf.gfile.GFile(imgfile, "rb") as f:
            np_image_string = np.array([f.read()])

        pcd = kitti_util.load_pcd(
            kitti_path + 'velodyne/' + str(frame).zfill(6) + '.bin')

        label_file = kitti_path + 'labels/' + str(frame).zfill(6) + '.label'
        labels = np.fromfile(label_file, dtype=np.int32)
        labels = labels.reshape((-1))  # reshape to vector
        labels = labels & 0xFFFF       # get lower half for semantics
        labels = remap_lut[labels]       # remap to xentropy format



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


                # Getting image features
        tic = time.time()
        output = openseg_model.signatures["serving_default"](
            inp_image_bytes=tf.convert_to_tensor(np_image_string[0]),
            inp_text_emb=text_embedding,
        )
        toc = time.time()
        # print('Inference Time: ', toc-tic)
        

        tic = time.time()
        # img_feat = output["ppixel_ave_feat"] # 1xHxWxC # This has zero norms at many places
        img_feat = output["image_embedding_feat"] # 1xHxWxC
        image_height, image_width, _ = img.shape

        # TODO: 190 is for kitt. Verify whi first approach is not working
        # img_feat = img_feat[:, :int(image_height*(640/image_width)), :, :]
        img_feat = img_feat[:, :190, :, :] # 190 was for per pixel avg

        # Default is bilinear
        # breakpoint()
        tic = time.time()
        img_feat = tf2.raw_ops.ResizeBilinear(images=img_feat, size=[image_height, image_width], align_corners=True)
        toc = time.time()
        # print('Upscaling Time: ',toc-tic)


        # Normalize embeddings to unit vectors
        # image_embedding = torch.nn.functional.normalize(image_embedding, dim=1)
        tic = time.time()
        img_feat = tf2.linalg.normalize(img_feat, ord='euclidean', axis=3,)[0][0]
        # img_feat = image_embedding[0]
        toc = time.time()
        # print('Normalize Time: ', toc -tic)


        
        # Features in PCD Space
        tic = time.time()
        # We want img_feat to be HxWxC
        # img_feat = torch.permute(image_embedding[0], (1,2,0))
        # img_feat_np = cv2.resize(img_feat_np, (img.shape[1], img.shape[0]))
        indices = np.vstack([pts_cam[mask_idx,1], pts_cam[mask_idx,0]]).T
        pcd_feat = tf.gather_nd(img_feat, indices)

        # pcd_feat = torch.tensor(pcd_feat.numpy())     
        pcd_feat = tf2.expand_dims(pcd_feat, axis=0)
        similarity = cosine_similarity(pcd_feat, text_feats)
        pred = tf2.math.argmax(similarity, axis=0)
        labels = labels[mask_idx]
        breakpoint()
        debug = True
        

        preds_all[mask_idx] = pred.numpy()
        np.save(out_path+'predictions/' + str(frame).zfill(6) + '.npy', preds_all)
        np.save(out_path+'masks/' + str(frame).zfill(6) + '.npy', mask)
        
        del output, pcd_feat, img_feat, similarity, pred

        

if __name__ == '__main__':
    main()


'''
# if count_curr!=0 and label!=0:
            #     feat = torch.tensor(pcd_feat.numpy(), dtype=torch.float32)
            #     selected_vector = torch.tensor(tf.gather(pcd_feat, idxs)[0].numpy(), dtype=torch.float32)

            #     sim = torch.nn.functional.cosine_similarity(feat, u_curr.unsqueeze(0), dim=1)
            #     pred = torch.where(sim>0.7)[0]
            #     intersection=pred[(pred.view(1, -1) == idxs.view(-1, 1)).any(dim=0)]
            #     union = torch.cat((pred, idxs)).unique()
            #     print('IoU with Mean Vector: ', intersection.shape[0]/union.shape[0])

            #     sim = torch.nn.functional.cosine_similarity(feat, selected_vector.unsqueeze(0), dim=1)
            #     pred = torch.where(sim>0.7)[0]
            #     intersection=pred[(pred.view(1, -1) == idxs.view(-1, 1)).any(dim=0)]
            #     union = torch.cat((pred, idxs)).unique()
            #     print('IoU with One Vector: ', intersection.shape[0]/union.shape[0])

            #     breakpoint()
'''
