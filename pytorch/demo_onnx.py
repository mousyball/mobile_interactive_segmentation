import argparse
import logging
import os
import time

# NOTE: 'OMP_NUM_THREADS' should be defined before onnxruntime
os.environ["OMP_NUM_THREADS"] = "4"

import cv2
import matplotlib.pyplot as plt
import torch
from easydict import EasyDict as edict

from core.isegm.inference import utils
from core.utils import onnx_helpers
from core.utils.config import get_cfg_defaults


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

    parser.add_argument('-cfg', '--config',
                        type=str,
                        default='./configs/onnx.yaml',
                        help='Config path')

    parser.add_argument('-n', '--num_iter',
                        type=int,
                        default=1,
                        help='Number of iteration in profiling')

    return parser.parse_args()


class ONNXRunner:
    def __init__(self, args) -> None:
        self.args = args
        self._config = self._init(args)
        self.onnx_handler = self._onnx_init(args, self._config)

    @property
    def config(self):
        return self._config

    def _init(self, args):
        yaml_cfg = self._parse_yaml_config(args.config)
        device = self._get_device(args.gpu_id, args.gpu)
        model = self._load_pytorch_model(weights=args.weights,
                                         device=device,
                                         onnx_enable=args.onnx)

        # NOTE: Parameters
        params = {
            'predictor_params': {
                'net_clicks_limit': yaml_cfg.MAX_CLICKS
            }
        }

        cfg_dict = edict(
            net=model,
            device=device,
            predictor_params=params,
            prob_thresh=yaml_cfg.PROB_THRESH,
            input_size=yaml_cfg.INPUT_SIZE,
            orig_size=None,
            resize=None,
            image_width=None,
            net_clicks_limit=yaml_cfg.MAX_CLICKS,
            onnx_path=args.onnx_path
        )

        return cfg_dict

    @staticmethod
    def _parse_yaml_config(config_path):
        cfg = get_cfg_defaults()
        cfg.merge_from_file(config_path)
        cfg.freeze()

        return edict(dict(
            MAX_CLICKS=cfg.EVAL.MAX_CLICKS,
            PROB_THRESH=cfg.EVAL.OUTPUT_PROBABILITY_THRESH,
            INPUT_SIZE=cfg.EVAL.INPUT_SIZE))

    @staticmethod
    def _get_device(gpu_id, gpu=False):
        if gpu:
            device = torch.device("cuda:"+str(gpu_id)
                                if torch.cuda.is_available()
                                else "cpu")
        else:
            device = torch.device("cpu")

        return device

    @staticmethod
    def _load_pytorch_model(weights, device, onnx_enable=False):
        if onnx_enable:
            logger.info("Loading model...")
            model = utils.load_is_model(weights, device, cpu_dist_maps=False)
            logger.info("Loading model is finished.")
            return model
        else:
            logger.info("[SKIP] Loading pytorch model is off.")
            logger.info("[SKIP] Exporting ONNX model is disable.")

    @staticmethod
    def _load_image(image_path):
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"{image_path}")

        image = cv2.imread(image_path, -1)
        _, _, channel = image.shape
        assert channel == 3, "Channel of input image is not 3."

        image = cv2.cvtColor(image,
                             cv2.COLOR_BGR2RGB)

        return image

    @staticmethod
    def _onnx_init(args, config):
        onnx_handler = onnx_helpers.ONNXHandler(config)
        onnx_path = args.onnx_path
        if args.onnx:
            # Export ONNX model if needed.
            onnx_handler.export_onnx(export_path=onnx_path)
            onnx_handler.check_onnx_model(onnx_path=onnx_path)
        onnx_handler.init_ort_session(onnx_path=onnx_path)

        return onnx_handler

    @staticmethod
    def _onnx_main(image, click_list, onnx_handler, config, num_iter=1):
        acc = 0
        for _ in range(num_iter):
            tic = time.time()
            mask = onnx_helpers.onnx_interface(img_np=image,
                                               click_list=click_list,
                                               onnx_handler=onnx_handler,
                                               cfg=config)
            toc = time.time()
            logger.info(f"[SINGLE]: {toc - tic:.5f} s")
            acc += (toc - tic)
        logger.info(f"[AVERAGE]: {acc/num_iter:.5f} s")

        return mask

    @staticmethod
    def visualize_helper(image, mask, click_list):
        from core.utils.visualize_helpers import overlay_masks

        # Visualize mask
        img_np = overlay_masks(image/255, [mask])
        # Visualize points
        for p in click_list:
            if p[2]:
                img_np = cv2.circle(img_np, (int(p[0]), int(p[1])), radius=5, color=(0, 255, 0), thickness=-1)
            else:
                img_np = cv2.circle(img_np, (int(p[0]), int(p[1])), radius=5, color=(255, 0, 0), thickness=-1)
        plt.imshow(img_np)
        plt.show()

    def apply(self, image, click_list):
        mask = self._onnx_main(image=image,
                               click_list=click_list,
                               onnx_handler=self.onnx_handler,
                               config=self.config,
                               num_iter=self.args.num_iter)

        return mask


def main():
    # Initialize parser and onnx runner
    args = parse_args()
    onnx_runner = ONNXRunner(args)

    # Input image
    img_path = './images/sheep.jpg'
    image = onnx_runner._load_image(img_path)

    # Mock input
    click_list = [
        [487, 75, 1],
        [430, 200, 1],
        [450, 200, 0],
        [435, 270, 1],
        [479, 44, 1],
        [396, 96, 1]
    ]

    # Apply inference with ONNX model
    mask = onnx_runner.apply(image=image,
                             click_list=click_list)

    # Visualize if needed
    if args.visualize:
        onnx_runner.visualize_helper(image, mask, click_list)


if __name__ == "__main__":
    main()

