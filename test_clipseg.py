from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import matplotlib.patches as mpatches
import numpy as np
import torch
import tyro
from clipseg import CLIPDensePredT
from matplotlib import pyplot as plt
from torchvision import transforms




def get_new_pallete(num_colors: int) -> torch.Tensor:
    """Create a color pallete given the number of distinct colors to generate.
    Args:
        num_colors (int): Number of colors to include in the pallete
    Returns:
        torch.Tensor: Generated color pallete of shape (num_colors, 3)
    """
    pallete = []
    for j in range(num_colors):
        lab = j
        r, g, b = 0, 0, 0
        i = 0
        while lab > 0:
            r |= ((lab >> 0) & 1) << (7 - i)
            g |= ((lab >> 1) & 1) << (7 - i)
            b |= ((lab >> 2) & 1) << (7 - i)
            i = i + 1
            lab >>= 3
        pallete.append([r, g, b])
    return torch.Tensor(pallete).float() / 255.0


def upsample_feat_vec(feat, target_shape):
    return torch.nn.functional.interpolate(
        feat, target_shape, mode="bilinear", align_corners=True
    )


def normalize_and_apply_pallete(upsampled_img, color, thresh=0.0):
    # 1, 1, H, W -> 1, 3, H, W
    upsampled_img = upsampled_img.repeat(1, 3, 1, 1)
    # Nomalize image
    upsampled_img = (upsampled_img - upsampled_img.min()) / (
        upsampled_img.max() - upsampled_img.min() + 1e-10
    )
    # threshold
    upsampled_img[upsampled_img < thresh] = 0.0
    # 1, 3, H, W -> H, W, 3
    upsampled_img = upsampled_img[0].permute(1, 2, 0)
    # Assign color
    return upsampled_img * color.view(1, 1, 3)


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

    checkpoint_path: str = 'data/clipseg/rd64-uni.pth'
    image_path: str = 'data/SEMANTIC-KITTI-DATASET/sequences/00/image_2/000000.png'
    prompts: Union[Tuple[str], str] = (
        "road",
        "car",
        "vegetation",
        "sidewalk",
        "building",
        "sky"
    )


if __name__ == "__main__":

    # Process command-line args (if any)
    args = tyro.cli(ProgramArgs)

    model = CLIPDensePredT(version="ViT-B/16", reduce_dim=64)
    model.eval()

    # non-strict, because we only stored decoder weights (not CLIP weights)
    # model.load_state_dict(
    #     torch.load(
    #         "checkpoints/clipseg_weights/rd64-uni.pth", map_location=torch.device("cpu")
    #     ),
    #     strict=False,
    # )
    model.load_state_dict(
        torch.load(str(args.checkpoint_path), map_location=torch.device("cpu")),
        strict=False,
    )

    # img = cv2.imread("/home/krishna/code/clipseg/example_image.jpg")
    img = cv2.imread(str(args.image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_to_disp = img.copy()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((352, 352)),
        ]
    )
    img = transform(img).unsqueeze(0)

    # prompts = ["a glass", "something to fill", "wood", "a jar", "spoon", "fork"]
    prompts = None
    if type(args.prompts) == str:
        prompts.append(args.prompts)
    else:
        prompts = list(args.prompts)

    # predict
    with torch.no_grad():
        preds = model(img.repeat(len(prompts), 1, 1, 1), prompts)[0]

    

    preds = upsample_feat_vec(preds, [img_to_disp.shape[0], img_to_disp.shape[1]])
    palletes = get_new_pallete(len(prompts))
    print(palletes)
    labels = preds.argmax(axis=0)[0]
    vis = palletes[labels]

    fig = plt.figure(figsize=(10, 7))
    frows, fcols = 2,1
    fig.add_subplot(frows, fcols, 1)
    plt.imshow(img_to_disp)
    fig.add_subplot(frows, fcols, 2)
    plt.imshow(vis)
    plt.show()
    

    # # (1, 1, net_output_height, net_output_width) -> (1, 1, img_height, img_width)
    # # for each prompt
    # upsampled_img_heatmaps = [
    #     upsample_feat_vec(
    #         preds[i].unsqueeze(0), [img_to_disp.shape[0], img_to_disp.shape[1]]
    #     )
    #     for i in range(len(prompts))
    # ]
    # upsampled_img_heatmaps = [
    #     torch.sigmoid(heatmap) for heatmap in upsampled_img_heatmaps
    # ]

    # pallete = get_new_pallete(len(prompts))
    # pallete_weight = 0.5 / (pallete.shape[0])
    # img_to_disp = 0.5 * (img_to_disp.astype(np.float32) / 255.0)
    # overlay_seg_maps = None
    # colored_imgs = []
    # for idx, heatmap in enumerate(upsampled_img_heatmaps):
    #     colored_img = normalize_and_apply_pallete(heatmap, pallete[idx])
    #     colored_imgs.append(colored_img.detach().cpu().numpy())
    #     if overlay_seg_maps is None:
    #         overlay_seg_maps = colored_img
    #     else:
    #         overlay_seg_maps = overlay_seg_maps + colored_img.detach().cpu().numpy()
    # overlay_seg_maps = overlay_seg_maps + img_to_disp

    # # plt.imshow(overlay_seg_maps)
    # # plt.legend(
    # #     handles=[
    # #         mpatches.Patch(
    # #             color=(
    # #                 pallete[i][0].item(),
    # #                 pallete[i][1].item(),
    # #                 pallete[i][2].item(),
    # #             ),
    # #             label=prompts[i],
    # #         )
    # #         for i in range(len(prompts))
    # #     ]
    # # )
    # # plt.show()

    # # Visualize predictions
    # _, ax = plt.subplots(1, len(prompts) + 1, figsize=(15, len(prompts)))
    # [a.axis("off") for a in ax.flatten()]
    # ax[0].imshow(img_to_disp)
    # [ax[i + 1].imshow(colored_imgs[i]) for i in range(len(prompts))]
    # [ax[i + 1].text(0, -15, prompts[i]) for i in range(len(prompts))]
    # plt.show()
    # breakpoint()
