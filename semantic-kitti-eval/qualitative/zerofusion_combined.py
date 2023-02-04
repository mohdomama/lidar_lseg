"""
Extract Mask2Former predictions for an input image
"""
import sys
sys.path.insert(1, "/scratch/padfoot7/Mask2Former")
from predictor import VisualizationDemo
from mask2former import add_maskformer2_config
from detectron2.utils.logger import setup_logger
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from detectron2.config import get_cfg
from natsort import natsorted
import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import warnings
import time
import tempfile
import argparse
import glob
import multiprocessing as mp
import os
import pickle
from extract_clip_features import get_image_features_sg
import open_clip
import clip



OPENCLIP_MODEL = "ViT-H-14"
OPENCLIP_PRETRAINED_DATASET = "laion2b_s32b_b79k"
OUT_IMG_HEIGHT = 370
OUT_IMG_WIDTH = 1226

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    return cfg


def get_parser():
    parser = argparse.ArgumentParser(
        description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="/scratch/padfoot7/Mask2Former/configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--kitti_dataset",
        default="data/SEMANTIC-KITTI-DATASET/",
        metavar="FILE",
        help="path to semantic kitti dataset",
    )
    return parser


def main():

    # ZeroFusion Model
    # ZeroFusion
    torch.autograd.set_grad_enabled(False)
    model, _, preprocess = open_clip.create_model_and_transforms(
        OPENCLIP_MODEL, OPENCLIP_PRETRAINED_DATASET
    )
    model.cuda()
    model.eval()
    tokenizer = open_clip.get_tokenizer(OPENCLIP_MODEL)
    cosine_similarity = torch.nn.CosineSimilarity(dim=2)
    
    # Maskformer stuf
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    # Begin
    sequence, frame = '00', 2736

    kitti_path = args.kitti_dataset + 'sequences/' + sequence + '/'
    frame = int(frame)
    imgfile = kitti_path + 'image_2/' + str(frame).zfill(6) + '.png'
    maskfile = 'data/zerofusion/qualitative/mask.pt'

    stem = os.path.splitext(os.path.basename(imgfile))[0]

    img = read_image(imgfile, format="BGR")
    predictions, visualized_output = demo.run_on_image(img)
    start_time = time.time()

    out_masks = torch.nn.functional.interpolate(
        predictions["instances"].pred_masks.unsqueeze(0), [OUT_IMG_HEIGHT, OUT_IMG_WIDTH], mode="nearest"
    )
    torch.save(out_masks[0].detach().cpu(), maskfile)
    

    # ZeroFusion Clip Features
    img_feat, global_feat = get_image_features_sg(imgfile, maskfile, model, preprocess)

    while True:
        prompt = input('Enter Prompt: ')
        if prompt=='exit':
            break
        prompt = clip.tokenize(prompt)
        prompt = prompt.cuda()
        text_feat = model.encode_text(prompt)  # 1, 512
        text_feat = torch.nn.functional.normalize(
            text_feat, dim=1).cpu()
        sim = cosine_similarity(img_feat, text_feat.unsqueeze(0)).detach().numpy()
        sim = sim / sim.max()
        sim[sim<0.9] = 0.1
        plt.imshow(sim)
        plt.imsave('data/zerofusion/qualitative/zf_result.png', sim)

        plt.imshow(img)
        plt.imsave('data/zerofusion/qualitative/zf_img.png', img)

        img2 = np.dstack([img/255.0, sim])
        plt.imshow(img2)
        plt.imsave('data/zerofusion/qualitative/zf_img2.png', img2)


        breakpoint()


        
    


if __name__ == "__main__":
    main()