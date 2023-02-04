"""
Extract CLIP features from an image
"""

import argparse
import json
import math
import os
import glob
from natsort import natsorted

import cv2
import numpy as np
import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from textwrap import wrap
from tqdm import tqdm, trange

import clip
import open_clip

from detectron2.utils.colormap import random_color
import pickle


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


def get_bbox_around_mask(mask):
    # mask: (img_height, img_width)
    # compute bbox around mask
    bbox = None
    nonzero_inds = torch.nonzero(mask)  # (num_nonzero, 2)
    if nonzero_inds.numel() == 0:
        topleft = [0, 0]
        botright = [mask.shape[0], mask.shape[1]]
        bbox = (topleft[0], topleft[1], botright[0], botright[1])  # (x0, y0, x1, y1)
    else:
        topleft = nonzero_inds.min(0)[0]  # (2,)
        botright = nonzero_inds.max(0)[0]  # (2,)
        bbox = (topleft[0].item(), topleft[1].item(), botright[0].item(), botright[1].item())  # (x0, y0, x1, y1)
    # x0, y0, x1, y1
    return bbox, nonzero_inds


def sample_bboxes_around_bbox(bbox, img_height, img_width, scales=[2, 4]):
    # bbox: (x0, y0, x1, y1)
    # scales: scale factor of resultant bboxes to sample
    x0, y0, x1, y1 = bbox
    bbox_height = x1 - x0 + 1
    bbox_width = y1 - y0 + 1
    ret_bboxes = []
    for sf in scales:
        # bbox with Nx size of original must be expanded by size (N-1)x
        # (orig. size = x; new added size = (N-1)x; so new size = (N-1)x + x = Nx)
        # we add (N-1)x // 2  (for each dim; i.e., x = height for x dim; x = width for y dim)
        assert sf >= 1, "scales must have values greater than or equal to 1"
        pad_height = int(math.floor((sf - 1) * bbox_height / 2))
        pad_width = int(math.floor((sf - 1) * bbox_width / 2))
        x0_new, y0_new, x1_new, y1_new = 0, 0, 0, 0
        x0_new = x0 - pad_height
        if x0_new < 0:
            x0_new = 0
        x1_new = x1 + pad_height
        if x1_new >= img_height:
            x1_new = img_height - 1
        y0_new = y0 - pad_width
        if y0_new < 0:
            y0_new = 0
        y1_new = y1 + pad_width
        if y1_new >= img_width:
            y1_new = img_width - 1
        ret_bboxes.append((x0_new, y0_new, x1_new, y1_new))
    return ret_bboxes



# IMGFILE = "1/krishnas-tabletop-smoketest.png"
# MASK_LOAD_FILE = "1/mask2former_instances.pt"
# GLOBAL_FEAT_SAVE_FILE = "1/global_feat.pt"
# # SEMIGLOBAL_FEAT_SAVE_FILE = "global_feat_to_all_filtered_masks.pt"
# SEMIGLOBAL_FEAT_SAVE_FILE = "1/global_feat_plus_mask_weighted.pt"
# GLOBAL_FEAT_LOAD_FILE = "1/global_feat.pt"
# OPENSEG_FEAT_LOAD_FILE = "1/feat_openseg.pt"
# LOCAL_FEAT_SAVE_FILE_2X = "1/per_mask_feat_2x.pt"
# LOCAL_FEAT_SAVE_FILE_4X = "1/per_mask_feat_4x.pt"

# OUT_IMG_HEIGHT = 370
# OUT_IMG_WIDTH = 1226
LOAD_IMG_HEIGHT = 370
LOAD_IMG_WIDTH = 1226
TGT_IMG_HEIGHT = 370
TGT_IMG_WIDTH = 1226
OPENCLIP_MODEL = "ViT-H-14"
OPENCLIP_PRETRAINED_DATASET = "laion2b_s32b_b79k"
# FEAT_DIM = 1024


def get_parser():
    parser = argparse.ArgumentParser(description="Extract CLIP features")
    parser.add_argument(
        "--imgdir",
        type=str,
        required=True,
        help="Directory containing input image files",
    )
    parser.add_argument(
        "--maskdir",
        type=str,
        required=True,
        help="Directory containing input instance masks",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Directory to save clip features to",
    )
    return parser


def get_image_features_sg(imgfile, maskfile, model, preprocess, semiflobal_off=False):
     # print("Reading image...")
    img = cv2.imread(imgfile)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (LOAD_IMG_WIDTH, LOAD_IMG_HEIGHT))
    img = torch.from_numpy(img)

    """
    Extract and save global feat vec
    """
    global_feat = None
    with torch.cuda.amp.autocast():
        # print("Extracting global CLIP features...")
        _img = preprocess(Image.open(imgfile)).unsqueeze(0)
        imgfeat = model.encode_image(_img.cuda())
        imgfeat /= imgfeat.norm(dim=-1, keepdim=True)
        tqdm.write(f"Image feature dims: {imgfeat.shape} \n")
        # global_feat_savefile = os.path.join(
        #     global_feat_savedir,
        #     f"feat_global_{OPENCLIP_MODEL}_{OPENCLIP_PRETRAINED_DATASET}_{LOAD_IMG_HEIGHT}_{LOAD_IMG_WIDTH}.pt",
        # )
        
        global_feat = imgfeat.detach().cpu().half()
        

    global_feat = global_feat.half().cuda()
    global_feat = torch.nn.functional.normalize(global_feat, dim=-1)  # --> (1, 1024)
    feat_dim = global_feat.shape[-1]

    cosine_similarity = torch.nn.CosineSimilarity(dim=-1)


    """
    Extract per-mask features
    """
    # Output feature vector (semiglobal, 2x, 4x)
    outfeat_sg = torch.zeros(LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, feat_dim, dtype=torch.half)
    # outfeat_1x = torch.zeros(LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, feat_dim, dtype=torch.half)
    # outfeat_2x = torch.zeros(LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, feat_dim, dtype=torch.half)
    # outfeat_4x = torch.zeros(LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, feat_dim, dtype=torch.half)
    # # outfeat_zseg = torch.zeros(LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, feat_dim, dtype=torch.half)

    # print(f"Loading instance masks {maskfile}...")
    mask = torch.load(maskfile).unsqueeze(0)  # 1, num_masks, H, W
    mask = torch.nn.functional.interpolate(mask, [LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH], mode="nearest")
    num_masks = mask.shape[-3]
    pallete = get_new_pallete(num_masks)

    rois = []
    roi_similarities_with_global_vec = []
    roi_sim_per_unit_area = []
    feat_per_roi = []
    roi_nonzero_inds = []

    for _i in range(num_masks):

        # viz = torch.zeros(IMG_HEIGHT, IMG_WIDTH, 3)
        curmask = mask[0, _i]
        bbox, nonzero_inds = get_bbox_around_mask(curmask)
        x0, y0, x1, y1 = bbox
        # viz[x0:x1, y0:y1, 0] = 1.0

        bbox_area = (x1 - x0 + 1) * (y1 - y0 + 1)
        img_area = LOAD_IMG_WIDTH * LOAD_IMG_HEIGHT
        iou = bbox_area / img_area

        if iou < 0.005:
            continue

        # per-mask features
        img_roi = img[x0:x1, y0:y1]
        img_roi = Image.fromarray(img_roi.detach().cpu().numpy())
        img_roi = preprocess(img_roi).unsqueeze(0).cuda()
        roifeat = model.encode_image(img_roi)
        roifeat = torch.nn.functional.normalize(roifeat, dim=-1)
        feat_per_roi.append(roifeat)
        roi_nonzero_inds.append(nonzero_inds)

        _sim = cosine_similarity(global_feat, roifeat)

        rois.append(torch.tensor(list(bbox)))
        roi_similarities_with_global_vec.append(_sim)
        roi_sim_per_unit_area.append(_sim)# / iou)
        # print(f"{_sim.item():.3f}, {iou:.3f}, {_sim.item() / iou:.3f}")


    """
    global_clip_plus_mask_weighted
    # """
    rois = torch.stack(rois)
    scores = torch.cat(roi_sim_per_unit_area).to(rois.device)
    # nms not implemented for Long tensors
    # nms on CUDA is not stable sorted; but the CPU version is
    retained = torchvision.ops.nms(rois.float().cpu(), scores.cpu(), iou_threshold=1.0)
    feat_per_roi = torch.cat(feat_per_roi, dim=0)  # N, 1024
    
    # print(f"retained {len(retained)} masks of {rois.shape[0]} total")
    retained_rois = rois[retained]
    retained_scores = scores[retained]
    retained_feat = feat_per_roi[retained]
    retained_nonzero_inds = []
    for _roiidx in range(retained.shape[0]):
        retained_nonzero_inds.append(roi_nonzero_inds[retained[_roiidx].item()])
    
    viz = torch.zeros(LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, 3)

    mask_sim_mat = torch.nn.functional.cosine_similarity(
        retained_feat[:, :, None], retained_feat.t()[None, :, :]
    )
    mask_sim_mat.fill_diagonal_(0.)
    mask_sim_mat = mask_sim_mat.mean(1)  # avg sim of each mask with each other mask
    softmax_scores = retained_scores.cuda() - mask_sim_mat
    softmax_scores = torch.nn.functional.softmax(softmax_scores, dim=0)
    # retained_scores = retained_scores.cuda() * mask_sim_mat
    # softmax_scores = torch.nn.functional.softmax(retained_scores, dim=0).cuda()

    if semiflobal_off:
        softmax_scores = softmax_scores * 0 + 1

    for _roiidx in range(retained.shape[0]):
        _weighted_feat = softmax_scores[_roiidx] * global_feat + (1 - softmax_scores[_roiidx]) * retained_feat[_roiidx]
        _weighted_feat = torch.nn.functional.normalize(_weighted_feat, dim=-1)
        outfeat_sg[retained_nonzero_inds[_roiidx][:, 0], retained_nonzero_inds[_roiidx][:, 1]] += _weighted_feat[0].detach().cpu().half()
        outfeat_sg[retained_nonzero_inds[_roiidx][:, 0], retained_nonzero_inds[_roiidx][:, 1]] = torch.nn.functional.normalize(
            outfeat_sg[retained_nonzero_inds[_roiidx][:, 0], retained_nonzero_inds[_roiidx][:, 1]].float(), dim=-1
        ).half()

    outfeat_sg = outfeat_sg.unsqueeze(0).float()  # interpolate is not implemented for float yet in pytorch
    outfeat_sg = outfeat_sg.permute(0, 3, 1, 2)  # 1, H, W, feat_dim -> 1, feat_dim, H, W
    outfeat_sg = torch.nn.functional.interpolate(outfeat_sg, [TGT_IMG_HEIGHT, TGT_IMG_WIDTH], mode="nearest")
    outfeat_sg = outfeat_sg.permute(0, 2, 3, 1)  # 1, feat_dim, H, W --> 1, H, W, feat_dim
    outfeat_sg = torch.nn.functional.normalize(outfeat_sg, dim=-1)
    outfeat_sg = outfeat_sg[0].half() # --> H, W, feat_dim

    return outfeat_sg, global_feat



def main():
    torch.autograd.set_grad_enabled(False)

    args = get_parser().parse_args()

    maskfiles = natsorted(glob.glob(os.path.join(args.maskdir, "*.pt")))
    if len(maskfiles) == 0:
        raise ValueError(f"No instance masks (*.pt files) found in {args.maskdir}")

    print(f"Initializing OpenCLIP model: {OPENCLIP_MODEL} pre-trained on {OPENCLIP_PRETRAINED_DATASET}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        OPENCLIP_MODEL, OPENCLIP_PRETRAINED_DATASET
    )
    model.cuda()
    model.eval()
    tokenizer = open_clip.get_tokenizer(OPENCLIP_MODEL)

    global_feat_savedir = os.path.join(args.outdir, "feat_global")
    os.makedirs(global_feat_savedir, exist_ok=True)
    semiglobal_feat_savedir = os.path.join(args.outdir, "feat_semiglobal")
    os.makedirs(semiglobal_feat_savedir, exist_ok=True)
    # local_feat_savedir_zseg = os.path.join(args.outdir, "feat_local_zseg")
    # os.makedirs(local_feat_savedir_zseg, exist_ok=True)
    local_feat_savedir_1x = os.path.join(args.outdir, "feat_local_1x")
    os.makedirs(local_feat_savedir_1x, exist_ok=True)
    local_feat_savedir_2x = os.path.join(args.outdir, "feat_local_2x")
    os.makedirs(local_feat_savedir_2x, exist_ok=True)
    local_feat_savedir_4x = os.path.join(args.outdir, "feat_local_4x")
    os.makedirs(local_feat_savedir_4x, exist_ok=True)

    with open('data/zerofusion/cherry_picked_frames.pkl', 'rb') as f:
        frames = pickle.load(f)

    # for imgidx in trange(len(maskfiles)):
    for sequence, frame in frames: 
        kitti_path = "data/SEMANTIC-KITTI-DATASET/" + 'sequences/' + sequence + '/'
        maskfile = 'data/zerofusion/cherrypicked/'+ sequence + '_' + str(frame).zfill(6) + '.pt' 
        imgfile = kitti_path + 'image_2/' + str(frame).zfill(6) + '.png'
        stem = os.path.splitext(os.path.basename(maskfile))[0]
        if not os.path.exists(imgfile):
            imgfile = os.path.join(args.imgdir, stem + ".png")
            if not os.path.exists(imgfile):
                raise ValueError(f"No file {stem} (.png / .jpg) in {args.imgdir}")

        """
        Load masks, sample boxes
        """
        
        outfeat_sg, global_feat = get_image_features_sg(imgfile, maskfile, model, preprocess)

        outfile = os.path.join(semiglobal_feat_savedir, stem + ".pt")
        # print(f"Saving semiglobal feat to {outfile}...")
        # torch.save(outfeat_sg, outfile)
        print('Shape: ', outfeat_sg.shape)

        global_feat_savefile = os.path.join(global_feat_savedir, stem + ".pt")
        # tqdm.write(f"Saving to {global_feat_savefile} \n")
        # torch.save(global_feat, global_feat_savefile)

if __name__ == "__main__":
    main()
