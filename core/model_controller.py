
import torch
from .isegm.inference import clicker
from .isegm.inference.predictors import get_predictor
from .isegm.utils.vis import draw_with_blend_and_clicks
from torchvision import transforms


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

