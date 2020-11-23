import argparse
import logging
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from core.isegm.inference import utils
from core.model_controller import InteractiveController

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()

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
    EVAL_MAX_CLICKS = 20 # TODO: config
    MODEL_THRESH = 0.5

    device = torch.device("cuda:"+str(args.gpu_id)
                          if torch.cuda.is_available()
                          else "cpu")

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"{args.checkpoint}")

    logger.info("Loading model...")
    model = utils.load_is_model(args.checkpoint, device)
    logger.info("Loading model is finished.")

    # NOTE: Parameters
    params = {
        'predictor_params': {
            'net_clicks_limit': EVAL_MAX_CLICKS
        }
    }

    ret_dict = dict(
        net=model,
        device=device,
        predictor_params=params,
        prob_thresh=MODEL_THRESH
    )
    return ret_dict


def main():
    args = parse_args()
    cfg = init(args)
    controller = InteractiveController(**cfg)

    img_path = './images/sheep.jpg'
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"{img_path}")

    img = cv2.imread(img_path, -1)
    _, _, channel = img.shape
    assert channel == 3, "Channel of input image is not 3."
    controller.set_image(img)

    if args.visualize:
        plt.figure(figsize=(10,6))

    # Mocking Input
    click_list = [
        [487, 75, 1],
        [430, 200, 1],
        [450, 200, 0],
        [435, 270, 1],
        [479, 44, 1],
        [396, 96, 1]
    ]

    # [MAIN]
    tic = time.time()
    controller.add_clicks(click_list)
    logger.info(f"Add {controller.clicker.get_state()}")
    click_time = time.time() - tic
    print(f"Elapsed time: {click_time: .5f} s")

    if args.visualize:
        vis = controller.get_visualization(alpha_blend=0.5, click_radius=3)
        plt.imshow(vis[...,::-1])
        plt.show()

    mask = (controller.current_object_prob > cfg['prob_thresh']) \
            .astype(np.uint8) * 255

    return mask


if __name__ == "__main__":
    mask = main()
