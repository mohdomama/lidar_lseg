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



# constants
WINDOW_NAME = "mask2former demo"


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
        "--indir",
        type=str,
        required=True,
        help="Directory containing input image files",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Directory to save instance masks to",
    )
    parser.add_argument(
        "--outdir_viz",
        type=str,
        help="(optional) Directory to save visualized output images to"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=25,
        help="Number of images to skip between successive frames"
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
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    # OUT_IMG_HEIGHT = 960
    # OUT_IMG_WIDTH = 1280

    OUT_IMG_HEIGHT = 370
    OUT_IMG_WIDTH = 1226

    # OUT_IMG_HEIGHT = 120
    # OUT_IMG_WIDTH = 160

    os.makedirs(args.outdir, exist_ok=True)
    if args.outdir_viz:
        os.makedirs(args.outdir_viz, exist_ok=True)

    imgfiles = glob.glob(os.path.join(args.indir, "*.jpg"))
    if len(imgfiles) == 0:
        imgfiles = glob.glob(os.path.join(args.indir, "*.png"))
    if len(imgfiles) == 0:
        raise ValueError(
            f"No images (.jpg or .png files) found in {args.indir}")
    imgfiles = natsorted(imgfiles)

    # Not skipping frames for Semantic KITTI Test
    # imgfiles = imgfiles[::args.stride]

    for imgidx in tqdm.trange(len(imgfiles)):

        imgfile = imgfiles[imgidx]

        stem = os.path.splitext(os.path.basename(imgfile))[0]
        mask_save_file = os.path.join(args.outdir, stem + ".pt")
        viz_save_file = None
        if args.outdir_viz:
            viz_save_file = os.path.join(args.outdir_viz, stem + ".png")

        img = read_image(imgfile, format="BGR")
        predictions, visualized_output = demo.run_on_image(img)
        start_time = time.time()
        logger.info(
            "{}: {} in {:.2f}s".format(
                imgfile,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )

        out_masks = torch.nn.functional.interpolate(
            predictions["instances"].pred_masks.unsqueeze(0), [OUT_IMG_HEIGHT, OUT_IMG_WIDTH], mode="nearest"
        )
        torch.save(out_masks[0].detach().cpu(), mask_save_file)
        plt.imshow(visualized_output.get_image()[:, :, ::1])
        plt.axis("off")
        plt.savefig(viz_save_file, bbox_inches="tight", pad_inches=0)
        # # plt.show()
        plt.close("all")
