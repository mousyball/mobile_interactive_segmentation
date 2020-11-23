
import logging
import os
from pathlib import Path

import torch
from torchvision import transforms

from .isegm.inference import clicker
from .isegm.inference.predictors import get_predictor
from .isegm.inference.utils import load_deeplab_is_model, load_hrnet_is_model
from .isegm.utils.vis import draw_with_blend_and_clicks

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ModelHandler:
    def __init__(self,
                 model_path,
                 gpu_id,
                 predictor_params,
                 prob_thresh=0.5) -> None:

        # Model
        self.device = self._get_device(gpu_id)
        self.net = self._load_model(model_path, self.device)

        # Parameters
        self.clicker = clicker.Clicker()
        self.states = []
        self.pred = None
        self.image = None
        self.image_nd = None
        self.predictor = None
        self.predictor_params = predictor_params
        self.prob_thresh = prob_thresh

        # Torch: parameter and transforms
        torch.backends.cudnn.deterministic = True
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])

    def _get_device(self, gpu_id):
        device = torch.device("cuda:"+str(gpu_id)
                              if torch.cuda.is_available()
                              else "cpu")
        return device

    def _load_model(self, checkpoint, device):
        logger.info("Loading model...")
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"{checkpoint}")

        if isinstance(checkpoint, (str, Path)):
            state_dict = torch.load(checkpoint, map_location='cpu')
        else:
            state_dict = checkpoint

        for k in state_dict.keys():
            if 'feature_extractor.stage2.0.branches' in k:
                self.net = load_hrnet_is_model(state_dict, device, backbone='auto')
                break

        if self.net is None:
            self.net = load_deeplab_is_model(state_dict, device, backbone='auto')

        logger.info("Loading model is finished.")

        return self.net

    def apply(self, click_list):
        # Add input clicks
        self.add_clicks(click_list)
        # Get prediction
        self.pred = self.inference()
        return self.pred

    def setup(self, image, predictor_params=None):
        self.image = image
        self.image_nd = self.transforms(image).to(self.device)
        if predictor_params:
            self._reset_predictor(predictor_params)
        else:
            self._reset_predictor(self.predictor_params)

    def inference(self):
        self.pred = self.predictor.get_prediction(self.clicker)
        return self.pred

    def add_click(self, x, y, is_positive):
        self.states.append({
            'clicker': self.clicker.get_state(),
        })

        click = clicker.Click(is_positive=is_positive, coords=(y, x))
        self.clicker.add_click(click)

    def add_clicks(self, click_list):
        self.states.append({
            'clicker': self.clicker.get_state(),
        })

        for _click in click_list:
            click = clicker.Click(is_positive=_click[2], coords=(_click[1], _click[0]))
            self.clicker.add_click(click)

    def undo_click(self):
        prev_state = self.states.pop()['clicker']
        self.clicker.set_state(prev_state)

    def _reset_predictor(self, predictor_params=None):
        self.clicker.reset_clicks()
        self.states = []
        if predictor_params is not None:
            self.predictor_params = predictor_params
        self.predictor = get_predictor(self.net, device=self.device,
                                       **self.predictor_params)
        if self.image_nd is not None:
            self.predictor.set_input_image(self.image_nd)

    @property
    def current_object_prob(self):
        return self.pred if self.pred.any() else None

    def get_visualization(self, alpha_blend, click_radius):
        if self.image is None:
            return None

        results_mask_for_vis = self.pred > self.prob_thresh

        vis = draw_with_blend_and_clicks(self.image,
                                         mask=results_mask_for_vis,
                                         alpha=alpha_blend,
                                         clicks_list=self.clicker.clicks_list, radius=click_radius)

        return vis


class InteractiveController:
    def __init__(self, net, device, predictor_params,
                 prob_thresh=0.5,
                 **kwargs):
        self.net = net.to(device)
        self.prob_thresh = prob_thresh
        self.clicker = clicker.Clicker()
        self.states = []
        self.pred = None

        self.image = None
        self.image_nd = None
        self.predictor = None
        self.device = device
        self.predictor_params = predictor_params
        self.reset_predictor(self.predictor_params)

    def set_image(self, image, predictor_params=None):
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])

        self.image = image
        self.image_nd = input_transform(image).to(self.device)
        self.reset_predictor(predictor_params)

    def add_click(self, x, y, is_positive):
        self.states.append({
            'clicker': self.clicker.get_state(),
        })

        click = clicker.Click(is_positive=is_positive, coords=(y, x))
        self.clicker.add_click(click)
        self.pred = self.predictor.get_prediction(self.clicker)
        torch.cuda.empty_cache()

    def add_clicks(self, click_list):
        self.states.append({
            'clicker': self.clicker.get_state(),
        })

        for _click in click_list:
            click = clicker.Click(is_positive=_click[2], coords=(_click[1], _click[0]))
            self.clicker.add_click(click)
        self.pred = self.predictor.get_prediction(self.clicker)
        torch.cuda.empty_cache()

    def reset_predictor(self, predictor_params=None):
        self.clicker.reset_clicks()
        self.states = []
        if predictor_params is not None:
            self.predictor_params = predictor_params
        self.predictor = get_predictor(self.net, device=self.device,
                                       **self.predictor_params)
        if self.image_nd is not None:
            self.predictor.set_input_image(self.image_nd)

    @property
    def current_object_prob(self):
        return self.pred if self.pred.any() else None

    def get_visualization(self, alpha_blend, click_radius):
        if self.image is None:
            return None

        results_mask_for_vis = self.pred > self.prob_thresh#self.result_mask

        vis = draw_with_blend_and_clicks(self.image,
                                         mask=results_mask_for_vis,
                                         alpha=alpha_blend,
                                         clicks_list=self.clicker.clicks_list, radius=click_radius)

        return vis

