from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.apis.inference import LoadImage
from mmcv.parallel import collate, scatter
from mmseg.core.evaluation import get_palette
import mmcv
import torch
from tools.maskclip_utils.prompt_engineering import zeroshot_classifier, prompt_templates
from mmseg.datasets.pipelines import Compose


def inference_segmentor_km(model, imgfile, return_feat=True):
    """Inference image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        (list[Tensor]): The segmentation result.
    """
    # cfg = model.cfg
    # device = next(model.parameters()).device  # model device
    # # build the data pipeline
    # test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    # test_pipeline = Compose(test_pipeline)
    # # prepare data
    # data = dict(img=img)
    # data = test_pipeline(data)
    # data = collate([data], samples_per_gpu=1)
    # if next(model.parameters()).is_cuda:
    #     # scatter to specified GPU
    #     data = scatter(data, [device])[0]
    # else:
    #     data['img_metas'] = [i.data[0] for i in data['img_metas']]

    import cv2
    img = cv2.imread(imgfile)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).float() / 255.
    img = img[..., :3]  # drop alpha channel, if present
    img = img.cuda()
    img = img.permute(2, 0, 1)  # C, H, W
    img = img.unsqueeze(0)  # 1, C, H, W

    imgs = [img]
    img_metas = [[{'ori_shape': img.shape, 'img_shape': img.shape, 'pad_shape': img.shape}]]

    # forward the model
    with torch.no_grad():
        # result = model.extract_feat(data['img'][0], return_feat=True)
        # print(result[0][3].shape)
        # result, feat = model.encode_decode(data["img"][0], data["img_metas"][0], return_feat=True)
        result, feat = model.encode_decode(img, img_metas, return_feat=True, upsample_feat=True)
    return result, feat



# imgfile = 'demo.png'
imgfile = "data/SEMANTIC-KITTI-DATASET/sequences/00/image_2/000000.png"
# fg_classes = ['pedestrian', 'car', 'bicycle']
# bg_classes = ['road', 'building']
fg_classes = ['blue pillow', 'painting']
bg_classes = []

text_embeddings = zeroshot_classifier('ViT-B/16', fg_classes+bg_classes, prompt_templates)
text_embeddings = text_embeddings.permute(1, 0).float()
print(text_embeddings.shape)
torch.save(text_embeddings, '../maskclip-edit/pretrain/demo_ViT_clip_text.pth')

config_file = '../maskclip-edit/configs/maskclip/maskclip_vit16_640x480_icl_km.py'
config = mmcv.Config.fromfile(config_file)
checkpoint_file = '../maskclip-edit/pretrain/ViT16_clip_backbone.pth'

num_classes = len(fg_classes + bg_classes)
config.model.decode_head.num_classes = num_classes
config.model.decode_head.text_categories = num_classes

config.data.test.fg_classes = fg_classes
config.data.test.bg_classes = bg_classes


# config.model.decode_head.num_vote = 1
# config.model.decode_head.vote_thresh = 1.
# config.model.decode_head.cls_thresh = 0.5

# build the model from a config file and a checkpoint file
model = init_segmentor(config, checkpoint_file, device='cuda:0')

_, feat = inference_segmentor_km(model, imgfile)
print(feat.shape)
breakpoint()
# import cv2
# img = cv2.imread(imgfile)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = torch.from_numpy(img).float() / 255.
# img = img[..., :3]  # drop alpha channel, if present
# img = img.cuda()
# img = img.permute(2, 0, 1)  # C, H, W
# img = img.unsqueeze(0)  # 1, C, H, W
# print(f"Image shape: {img.shape}")


# imgs = []
# img_metas = [[{'ori_shape': img.shape, 'img_shape': img.shape, 'pad_shape': img.shape}]]

# result = model(imgs, img_metas)

# # test a single image
# result = inference_segmentor(model, img)

# # show the results
# show_result_pyplot(model, img, result, None)

