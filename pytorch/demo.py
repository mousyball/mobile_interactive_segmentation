import argparse
import logging
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from easydict import EasyDict as edict

from core.model_controller import ModelHandler
from core.utils.visualize_helpers import overlay_masks

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-img', '--image', type=str,
                        default='./images/sheep.jpg',
                        help='Input image')

    parser.add_argument('--checkpoint', type=str,
                        default='./weights/hrnet32_ocr128_lvis.pth',
                        help='The path to the checkpoint. '
                             'This can be a relative path (relative to cfg.INTERACTIVE_MODELS_PATH) '
                             'or an absolute path. The file extension can be omitted.')

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='Id of GPU to use.')

    parser.add_argument('-vis', '--visualize', action='store_true',
                        default=False,
                        help='Visualize output')

    parser.add_argument('--cfg', type=str, default="config.yml",
                        help='The path to the config file.')

    return parser.parse_args()


def init(args):
    # Parameters
    EVAL_MAX_CLICKS = 20
    MODEL_THRESH = 0.5

    # Load image
    img_path = args.image
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"{img_path}")

    image = cv2.imread(img_path, -1)
    _, _, channel = image.shape
    assert channel == 3, "Channel of input image is not 3."
    # NOTE: image format is RGB
    image = cv2.cvtColor(image,
                         cv2.COLOR_BGR2RGB)

    # NOTE: Predictor Parameters
    params = {
        'predictor_params': {
            'net_clicks_limit': EVAL_MAX_CLICKS
        }
    }

    # Mocking Input
    click_list = [
        [487, 75, 1],
        [430, 200, 1],
        [450, 200, 0],
        [435, 270, 1],
        [479, 44, 1],
        [396, 96, 1]
    ]

    # Package the config
    cfg = edict({
        'image': image,
        'checkpoint': args.checkpoint,
        'gpu_id': args.gpu_id,
        'params': params,
        'click_list': click_list,
        'prob_thresh': MODEL_THRESH
    })

    return cfg


def main():
    # Prepare config
    args = parse_args()
    cfg = init(args)

    if args.visualize:
        plt.figure(figsize=(10,6))

    # Instantiate the model handler
    controller = ModelHandler(
        model_path=cfg.checkpoint,
        gpu_id=cfg.gpu_id,
        predictor_params=cfg.params,
        prob_thresh=cfg.prob_thresh
    )

    # [MAIN]
    # setup image: ToTensor, Normalize, to device
    # apply inference: add clicks, inference
    acc = 0
    num_clicks = 1
    with torch.no_grad():
        for _ in range(num_clicks):
            tic = time.time()
            controller.setup(cfg.image)
            prob_mask = controller.apply(cfg.click_list)
            logger.info(f"Add {controller.clicker.get_state()}")
            acc += time.time() - tic
        click_time = acc / num_clicks
        print(f"Elapsed time: {click_time: .5f} s")

    # Post-processing: probability to mask
    mask = (prob_mask > cfg.prob_thresh).astype(np.uint8) * 255

    if args.visualize:
        # Method#1
        vis = controller.get_visualization(alpha_blend=0.5, click_radius=3)
        plt.imshow(vis)
        plt.show()
        # Method#2
        img_np = overlay_masks(cfg.image/255, [mask])
        plt.imshow(img_np)
        plt.show()

    return mask


if __name__ == "__main__":
    mask = main()
