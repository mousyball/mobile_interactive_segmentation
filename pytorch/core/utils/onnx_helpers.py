import logging
import os
from operator import mul, truediv

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from core.isegm.inference import clicker
from skimage.transform import resize as skresize


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ONNXHandler:
    def __init__(self, cfg) -> None:
        """Initialize from config.

        Args:
            cfg (dict): config
        """
        self.parse_cfg(cfg)
        self.ort_session = None

    def parse_cfg(self, cfg):
        """Parse config into attributes.

        Args:
            cfg (dict): config
        """
        for key, value in cfg.items():
            if key == 'predictor_params':
                setattr(self, 'net_clicks_limit', value['predictor_params']['net_clicks_limit'])
            else:
                setattr(self, key, value)

    def export_onnx(self, export_path) -> None:
        """Export ONNX model with settings.

        Args:
            export_path (str): path where the ONNX model is exported.
        """
        EVAL_MAX_CLICKS = self.net_clicks_limit
        POINT_LENGTH = EVAL_MAX_CLICKS * 2
        HEIGHT, WIDTH = self.input_size

        # NOTE: dim=0: orig_img + flip_img = 2
        _image = torch.randn(2, 3, HEIGHT, WIDTH,
                             device=self.device,
                             dtype=torch.float32)
        _points = torch.ones(2, POINT_LENGTH, 2,
                             device=self.device,
                             dtype=torch.int32)

        # Providing input and output names sets the display names for values
        # within the model's graph. Setting these does not change the semantics
        # of the graph; it is only for readability.
        #
        # The inputs to the network consist of the flat list of inputs (i.e.
        # the values you would pass to the forward() method) followed by the
        # flat list of parameters. You can partially specify names, i.e. provide
        # a list here shorter than the number of inputs to the model, and we will
        # only set that subset of names, starting from the beginning.
        input_names = [ "image" ] + [ "points" ]
        output_names = [ "output"]

        # NOTE: Dynamic Axes make input dimension dynamic.
        dynamic_axes = {'points': {1: 'num_pts'}}

        # NOTE: Paramters Explanation
        # * args: input arguments. Wrap multiple inputs as tuple.
        # * f: path where the ONNX model is exported.
        # * do_constant_folding: enable constant-folding optimization
        # * input_names: setup input names as a list of string
        # * output_names: setup output names as a list of string
        # * opset_version: opset version of ONNX model. Latest one is recommended.
        # * operator_export_type:
        #   * OperatorExportTypes.ONNX: normal mode
        #   * OperatorExportTypes.ONNX_ATEN_FALLBACK: check 'ATen' node in debug mode
        # * dynamic_axes: define dynamic dimension inputs
        torch.onnx.export(self.net,
                          args=(_image, _points),
                          f=export_path,
                          export_params=True,
                          do_constant_folding=True,
                          verbose=True,
                          input_names=input_names,
                          output_names=output_names,
                          opset_version=12,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                          dynamic_axes=dynamic_axes)

    @staticmethod
    def check_onnx_model(onnx_path):
        """Check ONNX model if it is legit.

        Args:
            onnx_path (str): ONNX model path
        """
        # Load the ONNX model
        model = onnx.load(onnx_path)

        # Check that the IR is well formed
        onnx.checker.check_model(model)

        # Print a human readable representation of the graph
        onnx.helper.printable_graph(model.graph)

    def init_ort_session(self, onnx_path):
        """Initialize ONNX Runtime session

        Args:
            onnx_path (str): ONNX model path
        """
        # Setup options for optimization
        sess_options = ort.SessionOptions()
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.ort_session = ort.InferenceSession(onnx_path,
                                                sess_options=sess_options)

    def inference_ort(self, image, points):
        """Inference with ONNX Runtime session

        Args:
            image (np.array): processed image array
            points (np.array): processed points array

        Returns:
            np.array: probability array from model output
        """
        outputs = self.ort_session.run(None,
                                       {'image': image.astype(np.float32),
                                        'points': points.astype(np.int32)})
        return outputs


class ImageHelper:
    def __init__(self, image, input_size) -> None:
        """Initialize image helper for processing image.

        Args:
            image ([str, np.array]): input image path or array
            input_size (tuple): input size of model
        """
        self.input_image = image
        if isinstance(image, str):
            self.input_image = self._load_image(image)
        self.orig_size = self.input_image.shape[:2]
        self.input_size = input_size
        # 'Resize' is TRUE if input_size is not equal to image size (orig_size).
        self.resize = True if self.input_size != self.orig_size else False

        self._image_nd = self._preprocessing(self.input_image)

    @property
    def image_nd(self):
        return self._image_nd

    @property
    def input_shape(self):
        return self.input_image.shape

    def _load_image(self, img_path):
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"{img_path}")
        image = cv2.imread(img_path, -1)
        hei, wid, channel = image.shape
        self.orig_size = (hei, wid)
        assert channel == 3, "Channel of input image is not 3."
        image = cv2.cvtColor(image,
                             cv2.COLOR_BGR2RGB)
        return image

    @staticmethod
    def _np_resize_image(image, size, dtype='int'):
        """Resize image for np.array

        NOTE:
            * Resized result from cv2 and skimage is different. Just use a workaround to resize image in floating type.

        Args:
            image (np.array): image array
            size (tuple): input size of model
            dtype (str, optional): data type of image. Defaults to 'int'.

        Raises:
            NotImplementedError: dtype is allowed 'int' or 'float' only.

        Returns:
            np.array: resized image
        """
        if dtype == 'int':
            _size = (size[1], size[0]) # (H,W) to (W,H)
            return cv2.resize(image.astype('uint8'),
                              _size,
                              interpolation=cv2.INTER_LINEAR)
        elif dtype == 'float':
            return skresize(image,
                            size,
                            order=0,
                            mode='constant',
                            preserve_range=True)
        else:
            raise NotImplementedError(f"'{dtype}' is not a valid dtype.")

    @staticmethod
    def _np_transpose(image):
        """Transpose array dimension from (H,W,C) to (C,H,W)

        Args:
            image (np.array): image array

        Returns:
            np.array: resized image array
        """
        return np.transpose(image, (2, 0, 1))

    @staticmethod
    def _np_normalize(image):
        """Normalize image array

        Args:
            image (np.array): image array

        Returns:
            np.array: normalized image array
        """
        _image = image / 255.0
        _mean = np.array([[[.485]], [[.456]], [[.406]]]) # shape: (3, 1, 1)
        _std = np.array([[[.229]], [[.224]], [[.225]]])
        _image = (_image - _mean) / _std
        return _image

    @staticmethod
    def _np_flip_n_cat(image):
        """Horizontal flipping and concatenation for model input

        Args:
            image (np.array): image array

        Returns:
            np.array: result array
        """
        image_flip = np.flip(image, (2)) # flip the channel 2: width
        _image = np.expand_dims(image, axis=0)
        image_flip = np.expand_dims(image_flip, axis=0)
        return np.concatenate((_image, image_flip), axis=0)

    def _preprocessing(self, input_image):
        """Preprocess image for model input

        Args:
            input_image (np.array): input image

        Returns:
            np.array: preprocessed image
        """
        if self.resize:
            input_image = self._np_resize_image(input_image,
                                                self.input_size,
                                                dtype='int')
        image = self._np_transpose(input_image)
        image = self._np_normalize(image)
        image = self._np_flip_n_cat(image)
        return image

    @staticmethod
    def _np_sigmoid(prediction):
        """Sigmoid function for activation

        NOTE:
            * [WARN] Numerical stability is not handled.

        Args:
            prediction (np.array): predicted output

        Returns:
            np.array: probability map after activation
        """
        x = prediction
        prob_map = 1 / (1 + np.exp(-x))
        return prob_map

    @staticmethod
    def _np_merge_prediction(prediction):
        """Merge two layers output into one.

        Args:
            prediction (np.array): probability map with 2 layers

        Returns:
            np.array: single layer probability map
        """
        prob_map = prediction[0][0]
        prob_map_flipped = np.flip(prediction[1][0], (1)) # (H, W)
        _prob = 0.5 * (prob_map + prob_map_flipped)
        return _prob

    @staticmethod
    def _np_get_mask(prob_map, prob_thresh=0.5):
        """Binarize probability map into mask.

        Args:
            prob_map (np.array): probability map which range is [0, 1].
            prob_thresh (float, optional): probability threshold. Defaults to 0.5.

        Returns:
            np.array: mask which range is [0, 255].
        """
        mask = (prob_map > prob_thresh) * 255
        return mask.astype(np.uint8)

    def postprocessing(self, prediction, prob_thresh=0.5):
        """Post-process for model output

        Args:
            prediction (np.array): predicted result from model output
            prob_thresh (float, optional): probability threshold. Defaults to 0.5.

        Returns:
            np.array: mask
        """
        prob_map = self._np_sigmoid(prediction)
        prob_map = self._np_merge_prediction(prob_map)
        if self.resize :
            prob_map = self._np_resize_image(prob_map,
                                                self.orig_size,
                                                dtype='float')
        mask = self._np_get_mask(prob_map, prob_thresh=prob_thresh)
        return mask


class PointsHelper:
    def __init__(self,
                 click_list,
                 image_width,
                 input_size,
                 orig_size,
                 resize=False,
                 net_clicks_limit=20) -> None:
        """Initialize points helper for processing user clicks.

        Args:
            click_list (list): a list of list with shape (n, 3)
            image_width (int): image width
            input_size (tuple): (height, width)
            orig_size (tuple): (height, width)
            resize (bool, optional): flag to resize. Defaults to False.
            net_clicks_limit (int, optional): limitation to the number of clicks. Defaults to 20.
        """
        self.click_list = click_list
        self.image_width = image_width
        self.net_clicks_limit = net_clicks_limit
        self.input_size = input_size
        self.orig_size = orig_size
        self.resize = resize
        if self.resize:
            self.image_width = self.input_size[1]

        self._points_nd = self._preprocessing()

    @property
    def points_nd(self):
        return self._points_nd

    @staticmethod
    def _get_points_nd(clicks_lists, net_clicks_limit):
        """Generate specific format of points array.

        Args:
            clicks_lists (List[List]): clicks_lists with raw and flipped clicks.
            net_clicks_limit (int): limitation to the number of clicks.

        Returns:
            np.array: specific format of points array with some (-1, -1) filling points.
        """
        total_clicks = []
        num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
        num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
        num_max_points = max(num_pos_clicks + num_neg_clicks)

        if net_clicks_limit is not None:
            num_max_points = min(net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            clicks_list = clicks_list[:net_clicks_limit]
            pos_clicks = [click.coords for click in clicks_list if click.is_positive]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1)]

            neg_clicks = [click.coords for click in clicks_list if not click.is_positive]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)

        return np.array(total_clicks)

    @staticmethod
    def _points_transform(clicks_lists, image_width):
        """Transform original points list into flipped points and concatenate these two list.

        Args:
            clicks_lists (List[List]): clicks list. (Ex: [clicks_lists])
            image_width (int): image width for flipping

        Returns:
            List[List]: clicks list with re-formating. (Ex: [clicks_lists, clicks_lists_flipped])
        """
        clicks_lists_flipped = []
        for clicks_list in clicks_lists:
            clicks_list_flipped = []
            for click in clicks_list:
                # Horizontal flip
                _y = image_width - click.coords[1] - 1
                _click = clicker.Click(is_positive=click.is_positive,
                                       coords=(click.coords[0], _y))
                clicks_list_flipped.append(_click)
            clicks_lists_flipped.append(clicks_list_flipped)
        clicks_lists = clicks_lists + clicks_lists_flipped
        return clicks_lists

    @staticmethod
    def _get_clickers(click_list):
        """Wrap clicks by 'Clicker' class

        Args:
            click_list (List[List]): user click list

        Returns:
            Clicker: clicker object
        """
        clickers = clicker.Clicker()
        for _click in click_list:
            click = clicker.Click(is_positive=_click[2],
                                  coords=(_click[1], _click[0])) # (x, y)
            clickers.add_click(click)
        return clickers

    @staticmethod
    def _remapping_coord(click_list, input_size, orig_size):
        """Remap the coordinate if flag of resize is TRUE.

        Args:
            click_list (List[List]): user click list with shape (n, 3)
            input_size (tuple): input size of model (H, W)
            orig_size (tuple): original image size (H, W)

        Returns:
            List[List]: click list after coordinate remapping
        """
        input_coord = [input_size[1], input_size[0], 1]
        orig_coord = [orig_size[1], orig_size[0], 1]
        _click_list = list()
        for click in click_list:
            click = list(map(truediv, click, orig_coord))
            click = list(map(mul, click, input_coord))
            click = list(map(int, click))
            _click_list.append(click)
        return _click_list

    def _preprocessing(self):
        """Pre-processing the user clicks to points array

        Returns:
            np.array: points array for model input
        """
        if self.resize:
            self.click_list = self._remapping_coord(self.click_list,
                                                    self.input_size,
                                                    self.orig_size)
        clickers = self._get_clickers(self.click_list)
        clicks_list = clickers.get_clicks()
        clicks_lists = self._points_transform([clicks_list], self.image_width)
        points_nd = self._get_points_nd(clicks_lists, self.net_clicks_limit)
        return points_nd


def onnx_interface(img_np, click_list, onnx_handler, cfg):
    """ONNX interface contained main flow

    Args:
        img_np (np.array): image array in RBG-format
        click_list (List[List]): user click list
        onnx_handler (ONNXHandler): ONNX handler
        cfg (dict): config

    Returns:
        np.array: mask array with original image shape (H, W)
    """
    # [pre-processing][image]
    img_helper = ImageHelper(img_np,
                             input_size=cfg['input_size'])

    # Update config for PointsHelper
    cfg['orig_size'] = img_helper.orig_size
    cfg['resize'] = img_helper.resize
    cfg['image_width'] = img_helper.input_shape[1]

    # [pre-processing][points]
    pts_helper = PointsHelper(click_list,
                              image_width=cfg['image_width'],
                              input_size=cfg['input_size'],
                              orig_size=cfg['orig_size'],
                              resize=cfg['resize'],
                              net_clicks_limit=cfg['net_clicks_limit'])

    # [MAIN] inference
    outputs = onnx_handler.inference_ort(image=img_helper.image_nd,
                                         points=pts_helper.points_nd)

    # [post-processing]
    mask = img_helper.postprocessing(prediction=outputs[0],
                                     prob_thresh=cfg['prob_thresh'])

    return mask

