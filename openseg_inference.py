"""
Script to extract LSeg features over an ICL sequence and save them in the
ICL directory (for later use with gradslam).
"""

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import clip
import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
# from lseg import LSegNet
from natsort import natsorted
# from tqdm import tqdm, trange

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

def get_new_pallete(num_colors: int) -> torch.Tensor:
    """Create a color pallete given the number of distinct colors to generate.

    Args:
        num_colors (int): Number of colors to include in the pallete

    Returns:
        torch.Tensor: Generated color pallete of shape (num_colors, 3)
    """
    pallete = []
    # The first color is always black, so generate one additional color
    # (we will drop the black color out)
    for j in range(num_colors + 1):
        lab = j
        r, g, b = 0, 0, 0
        i = 0
        while lab > 0:
            r |= ((lab >> 0) & 1) << (7 - i)
            g |= ((lab >> 1) & 1) << (7 - i)
            b |= ((lab >> 2) & 1) << (7 - i)
            i = i + 1
            lab >>= 3
        if j > 0:
            pallete.append([r, g, b])
    return torch.Tensor(pallete).float() / 255.0


@dataclass
class ProgramArgs:
    """Commandline args for this script"""

    # LSeg model params
    checkpoint_path: Union[str, Path] = (
        Path(__file__).parent / "data" / "openseg" / "exported_model"
    )
    backbone: str = "clip_vitl16_384"
    num_features: int = 256
    arch_option: int = 0
    block_depth: int = 0
    activation: str = "lrelu"
    crop_size: int = 480

    # Hardware accelerator
    device: str = "cuda"

    # Dataset args
    in_dir: Union[str, Path] = Path("./data/lego_loam_map/lego_loam_images_png").expanduser()
    emb_dir: Union[str, Path] = Path(
        "./data/lego_loam_map/feat_openseg_640_640"
    ).expanduser()

    # Height to reshape feature maps to
    image_height: int = 360
    # Width to reshape feature maps to
    image_width: int = 640


if __name__ == "__main__":

    # Process command-line args (if any)
    args = tyro.cli(ProgramArgs)

    # Load a CLIP model
    print("Loading CLIP model...")
    clip_model, preprocess = clip.load("ViT-L/14@336px")

    imgfiles = natsorted(glob.glob(os.path.join(args.in_dir, "*.png")))

    prompt_text = [
        "roundabout",
        "road",
        # "car",
        # "person",
        # "building",
        "tree",
        "playground",
        # "football",
        # "field",
        "unknown",
    ]

    # outdir = os.path.join(args.in_dir, "..", f"openseg_debug_img_{prompt_text}")

    imgfiles = natsorted(glob.glob(os.path.join(args.in_dir, "*.png")))

    for imgfile in imgfiles:

        # imgidx = str(i)
        # imgfile = os.path.join(args.in_dir, imgidx + ".png")
        # embfile = os.path.join(args.emb_dir, imgidx + ".pt")

        # filestem = Path(imgfile).stem
        # embfile = os.path.join(args.emb_dir, filestem + ".pt")

        imgfile = './data/lego_loam_map/lego_loam_images_png/000645.png'
        embfile = './data/lego_loam_map/feat_openseg_640_640/000645.pt'

        # print("Reading image...")
        img = cv2.imread(imgfile)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # (360, 640, 3) original image size
        img = cv2.resize(img, (args.image_width, args.image_height))

        # print("Reading image embedding ...")
        image_embedding = torch.load(embfile)
        # image_embedding = image_embedding.cuda()
        image_embedding = image_embedding


        # print("Computing similarities ...")

        cosine_similarity = torch.nn.CosineSimilarity(dim=1)

        similarities = []
        for prompt in prompt_text:
            prompt = clip.tokenize(prompt)
            # prompt = prompt.cuda()
            prompt = prompt
            text_feat = clip_model.encode_text(prompt)
            text_feat = torch.nn.functional.normalize(text_feat, dim=0)

            similarity = cosine_similarity(
                image_embedding, text_feat.unsqueeze(-1).unsqueeze(-1)
            )
            similarities.append(similarity)

        similarities = torch.cat(similarities, dim=0)  # num_classes, H, W
        similarities = similarities.unsqueeze(0)  # 1, num_classes, H // 2, W // 2
        class_scores = torch.max(similarities, 1)[1]  # 1, H // 2, W // 2
        class_scores = class_scores[0].detach()

        pallete = get_new_pallete(len(prompt_text))
        # img size // 2 for height and width dims
        disp_img = torch.zeros(img.shape[0], img.shape[1], 3)
        for _i in range(len(prompt_text)):
            disp_img[class_scores == _i] = pallete[_i]
        disp_img = 0.5 * disp_img + 0.5 * (torch.from_numpy(img).float() / 255)

        plt.imshow(disp_img.detach().cpu().numpy())
        plt.legend(
            handles=[
                mpatches.Patch(
                    color=(
                        pallete[i][0].item(),
                        pallete[i][1].item(),
                        pallete[i][2].item(),
                    ),
                    label=prompt_text[i],
                )
                for i in range(len(prompt_text))
            ]
        )
        plt.savefig('openseg_inf.png', dpi=200)
        breakpoint()
