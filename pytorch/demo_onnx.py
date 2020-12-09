import argparse
import logging
import os
import time

# NOTE: 'OMP_NUM_THREADS' should be defined before onnxruntime
os.environ["OMP_NUM_THREADS"] = "4"

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from core.isegm.inference import utils
from core.utils import onnx_helpers


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--weights',
                        type=str,
                        default='./weights/hrnet32_ocr128_lvis.pth',
                        help='The path to the weights.')

    parser.add_argument('--gpu_id',
                        type=int,
                        default=0,
                        help='Id of GPU to use.')

    parser.add_argument('-vis', '--visualize',
                        action='store_true',
                        default=False,
                        help='Visualize output')

    parser.add_argument('--gpu',
                        action='store_true',
                        default=False,
                        help='GPU mode')

    parser.add_argument('--onnx',
                        action='store_true',
                        default=False,
                        help='Export ONNX model')

    parser.add_argument('--onnx_path',
                        type=str,
                        default='./weights/hrnet32_ocr128_lvis.onnx',
                        help='Exported ONNX model path')

    return parser.parse_args()


def init(args):
    EVAL_MAX_CLICKS = 20
    MODEL_THRESH = 0.5
    INPUT_SIZE = (320, 480) # training size

    if args.gpu:
        device = torch.device("cuda:"+str(args.gpu_id)
                              if torch.cuda.is_available()
                              else "cpu")
    else:
        device = torch.device("cpu")

    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"{args.weights}")

    logger.info("Loading model...")
    model = utils.load_is_model(args.weights, device, cpu_dist_maps=False)
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
        prob_thresh=MODEL_THRESH,
        input_size=INPUT_SIZE,
        orig_size=None,
        resize=None,
        image_width=None,
        net_clicks_limit=EVAL_MAX_CLICKS,
        onnx_path=args.onnx_path
    )

    return ret_dict


def visualize_helper(image, mask, click_list):
    # [Visualization]
    from core.utils.visualize_helpers import overlay_masks
    img_np = overlay_masks(image/255, [mask])
    # Visualize points
    for p in click_list:
        if p[2]:
            img_np = cv2.circle(img_np, (int(p[0]), int(p[1])), radius=5, color=(0, 255, 0), thickness=-1)
        else:
            img_np = cv2.circle(img_np, (int(p[0]), int(p[1])), radius=5, color=(255, 0, 0), thickness=-1)
    plt.imshow(img_np)
    plt.show()


def main():
    args = parse_args()
    cfg = init(args)

    # Parameters
    img_path = './images/sheep.jpg'
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"{img_path}")
    image = cv2.imread(img_path, -1)
    _, _, channel = image.shape
    assert channel == 3, "Channel of input image is not 3."
    image = cv2.cvtColor(image,
                         cv2.COLOR_BGR2RGB)

    # Mock input
    click_list = [
        [487, 75, 1],
        [430, 200, 1],
        [450, 200, 0],
        [435, 270, 1],
        [479, 44, 1],
        [396, 96, 1]
    ]

    # Prepare ONNX
    onnx_handler = onnx_helpers.ONNXHandler(cfg)
    onnx_path = cfg['onnx_path']
    if args.onnx:
        # Export ONNX model if needed.
        onnx_handler.export_onnx(export_path=onnx_path)
        onnx_handler.check_onnx_model(onnx_path=onnx_path)
    onnx_handler.init_ort_session(onnx_path=onnx_path)

    n = 1
    acc = 0
    for _ in range(n):
        tic = time.time()
        mask = onnx_helpers.onnx_interface(img_np=image,
                                           click_list=click_list,
                                           onnx_handler=onnx_handler,
                                           cfg=cfg)

        toc = time.time()
        print(f"[INFO] [SINGLE]: {toc - tic} s")
        acc += (toc - tic)
    print(f"[NOTE] [AVERAGE]: {acc/n} s")

    if args.visualize:
        visualize_helper(image, mask, click_list)


if __name__ == "__main__":
    main()

