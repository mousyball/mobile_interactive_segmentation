import torch
import torch.nn.functional as F

from ...inference.transforms import AddHorizontalFlip, SigmoidForPred


class BasePredictor(object):
    def __init__(self, net, device,
                 net_clicks_limit=None,
                 with_flip=False,
                 **kwargs):

        self.net = net
        self.with_flip = with_flip
        self.net_clicks_limit = net_clicks_limit
        self.original_image = None
        self.device = device

        self.transforms = []

        # NOTE: trans: nothing, inv_trans: sigmoid
        self.transforms.append(SigmoidForPred())
        if with_flip:
            # NOTE: trans:
            # image_nd: (1, 3, H, W) -> cat with flip(W) -> (2, 3, H, W)
            # clicks_lists: clicks_lists + clicks_lists_flipped
            # NOTE: inv_trans:
            # prob_map, prob_map_flipped
            # return 0.5 * (prob_map + torch.flip(prob_map_flipped, dims=[3]))
            self.transforms.append(AddHorizontalFlip())

    def set_input_image(self, image_nd):
        self.original_image = image_nd.to(self.device)
        if len(self.original_image.shape) == 3:
            self.original_image = self.original_image.unsqueeze(0)

    def get_prediction(self, clicker):
        clicks_list = clicker.get_clicks()

        # NOTE:
        # image_nd: (1, 3, H, W) -> cat with flip(W) -> (2, 3, H, W)
        # clicks_lists: [[clicks_list]] -> [[clicks_list], [clicks_lists_flipped]]
        image_nd, clicks_lists = self.apply_transforms(
            self.original_image, [clicks_list]
        )

        pred_logits = self._get_prediction(image_nd, clicks_lists)
        prediction = F.interpolate(pred_logits,
                                   mode='bilinear',
                                   align_corners=True,
                                   size=image_nd.size()[2:])

        for t in reversed(self.transforms):
            prediction = t.inv_transform(prediction)

        return prediction.cpu().numpy()[0, 0]

    def _get_prediction(self, image_nd, clicks_lists):
        points_nd = self.get_points_nd(clicks_lists)
        return self.net(image_nd, points_nd)['instances']

    def apply_transforms(self, image_nd, clicks_lists):
        for t in self.transforms:
            image_nd, clicks_lists = t.transform(image_nd, clicks_lists)

        return image_nd, clicks_lists

    def get_points_nd(self, clicks_lists):
        total_clicks = []
        num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
        num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
        num_max_points = max(num_pos_clicks + num_neg_clicks)

        if self.net_clicks_limit is not None:
            num_max_points = min(self.net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            clicks_list = clicks_list[:self.net_clicks_limit]
            pos_clicks = [click.coords for click in clicks_list if click.is_positive]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1)]

            neg_clicks = [click.coords for click in clicks_list if not click.is_positive]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)

        return torch.tensor(total_clicks, device=self.device)

