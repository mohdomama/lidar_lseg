"""
Script to extract LSeg features over an ICL sequence and save them in the
ICL directory (for later use with gradslam).
"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

import tensorflow.compat.v1 as tf
import torch
import tyro
from natsort import natsorted

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import clip
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf2

from tqdm import tqdm, trange


def upsample_feat_vec(feat, target_shape):
    return torch.nn.functional.interpolate(
        feat, target_shape, mode="bilinear", align_corners=True
    )


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
    in_dir: Union[str, Path] = Path(
        "./data/lego_loam_map/lego_loam_images_png"
    ).expanduser()
    out_dir: Union[str, Path] = Path(
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

    # Build a dummy text embedding (needed for the TF OpenSeg model)
    print("Building a dummy text embedding (needed by the TF OpenSeg model...")
    text_prompts = ["sofa", "pillow", "other"]
    text_embedding = build_text_embedding(text_prompts, clip_model)
    num_words_per_category = 1
    text_embedding = tf.reshape(
        text_embedding, [-1, num_words_per_category, text_embedding.shape[-1]]
    )
    text_embedding = tf.cast(text_embedding, tf.float32)

    print("Loading OpenSeg model...")
    openseg_model = tf2.saved_model.load(
        args.checkpoint_path, tags=[tf.saved_model.tag_constants.SERVING]
    )

    imgfiles = natsorted(glob.glob(os.path.join(args.in_dir, "*.png")))

    os.makedirs(args.out_dir, exist_ok=True)

    upsample_feature_maps = torch.nn.Upsample(
        size=(args.image_height, args.image_width), mode="bilinear"
    )

    for imgfile in tqdm(imgfiles):
        file_stem = str(Path(imgfile).stem)
        outfile = os.path.join(args.out_dir, file_stem + ".pt")

        with tf.gfile.GFile(imgfile, "rb") as f:
            np_image_string = np.array([f.read()])

        output = openseg_model.signatures["serving_default"](
            inp_image_bytes=tf.convert_to_tensor(np_image_string[0]),
            inp_text_emb=text_embedding,
        )

        image_embedding = torch.from_numpy(output["ppixel_ave_feat"].numpy()).float()
        # (1, 640, 640, 768) --> (1, 768, 640, 640)
        image_embedding = image_embedding.permute(0, 3, 1, 2)

        # Drop the rows from rows 480 and beyond (a quirk to handle OpenSeg's zero-padding scheme)
        
        # TODO: Verify if this is correct
        image_embedding = image_embedding[:, :, :int(args.image_height*(640/args.image_width)), :]

        image_embedding = upsample_feat_vec(
            image_embedding,
            [args.image_height, args.image_width],
        )

        # Normalize embeddings to unit vectors
        image_embedding = torch.nn.functional.normalize(image_embedding, dim=1)

        torch.save(image_embedding, outfile)