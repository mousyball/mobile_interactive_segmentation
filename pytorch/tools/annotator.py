import argparse
import logging
import os
import random
from glob import glob
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from core.model_controller import ModelHandler
from easydict import EasyDict as edict
from PIL import Image


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

random.seed(878)


class Annotator(object):
    def __init__(self, img_path,
                 img_list=None,
                 controller=None,
                 predictor_params=None,
                 save_path=None):
        assert predictor_params is not None, "Predictor parameters is None."
        assert controller is not None, "Controller is None."

        self.save_path = save_path
        self.file = Path(img_path).name
        self.img = self._load_image(img_path)
        self.clicks = np.empty([0,3],dtype=np.int64)
        self.pred = np.zeros(self.img.shape[:2],dtype=np.uint8)
        self.merge =  self.__gene_merge(self.pred, self.img, self.clicks)
        self.predictor_params = predictor_params

        self.controller = controller
        self.controller.setup(self.img, self.predictor_params)
        self.img_list = [img_path] if img_list is None else img_list
        self.img_cnt = 0

    @staticmethod
    def _load_image(img_path):
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"{img_path}")

        image = cv2.imread(img_path, -1)
        _, _, channel = image.shape
        assert channel == 3, "Channel of input image is not 3."
        # NOTE: image format is RGB
        image = cv2.cvtColor(image,
                             cv2.COLOR_BGR2RGB)
        return image

    def __gene_merge(self,pred,img,clicks,r=3,cb=1,b=1,if_first=True):
        pred_mask=cv2.merge([pred*255,pred*255,np.zeros_like(pred)])
        result= np.uint8(np.clip(img*0.7+pred_mask*0.3,0,255))
        if b>0:
            contours,_=cv2.findContours(pred,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result,contours,-1,(255,255,255),b)
        for pt in clicks:
            cv2.circle(result,tuple(pt[:2]),r,(255,0,0) if pt[2]==1 else (0,0,255),-1)
            cv2.circle(result,tuple(pt[:2]),r,(255,255,255),cb)
        if if_first and len(clicks)!=0:
            cv2.circle(result,tuple(clicks[0,:2]),r,(0,255,0),cb)
        return result

    def __update(self):
        self.ax1.imshow(self.merge)
        self.fig.canvas.draw()

    def __reset(self):
        self.clicks =  np.empty([0,3],dtype=np.int64)
        self.pred = np.zeros(self.img.shape[:2],dtype=np.uint8)
        self.merge =  self.__gene_merge(self.pred, self.img, self.clicks)

        self.controller.setup(self.img, self.predictor_params)
        self.__update()

    def __predict(self):
        prob = self.controller.inference()
        self.pred = (prob > 0.5).astype(np.uint8) * 255
        self.merge =  self.__gene_merge(self.pred, self.img, self.clicks)
        self.__update()

    def __on_key_press(self,event):
        if event.key=='left':
            self.ax1.cla()
            self.img_cnt -= 1
            if self.img_cnt < 0:
                self.img_cnt = 0
            self.file = Path(self.img_list[self.img_cnt]).name
            self.fig.suptitle('( file : {} )'.format(self.file),fontsize=16)
            self.img = np.array(Image.open(self.img_list[self.img_cnt]))
            self.__reset()
        elif event.key=='right':
            self.ax1.cla()
            self.img_cnt += 1
            if self.img_cnt >= len(self.img_list):
                self.img_cnt = len(self.img_list) - 1
            self.file = Path(self.img_list[self.img_cnt]).name
            self.fig.suptitle('( file : {} )'.format(self.file),fontsize=16)
            self.img = np.array(Image.open(self.img_list[self.img_cnt]))
            self.__reset()

        if event.key=='ctrl+z':
            self.clicks=self.clicks[:-1,:]
            self.controller.undo_click()
            if len(self.clicks)!=0:
                self.__predict()
            else:
                self.__reset()
        elif event.key=='ctrl+r':
            self.__reset()
        elif event.key=='escape':
            plt.close()
        elif event.key=='enter':
            if self.save_path is not None:
                Image.fromarray(self.pred).save(self.save_path)
                print('save mask in [{}]!'.format(self.save_path))
            plt.close()

    def __on_button_press(self,event):
        if (event.xdata is None) or (event.ydata is None):return
        if event.button==1 or  event.button==3:
            x,y= int(event.xdata+0.5), int(event.ydata+0.5)
            self.clicks=np.append(self.clicks,np.array([[x,y,(3-event.button)/2]],dtype=np.int64),axis=0)

            self.controller.add_click(
                x=self.clicks[-1][0],
                y=self.clicks[-1][1],
                is_positive=self.clicks[-1][2]
            )
            self.__predict()

    def main(self):
        self.fig = plt.figure('Annotator')
        self.fig.canvas.mpl_connect('key_press_event', self.__on_key_press)
        self.fig.canvas.mpl_connect("button_press_event",  self.__on_button_press)
        self.fig.suptitle('( file : {} )'.format(self.file),fontsize=16)
        self.ax1 = self.fig.add_subplot(1,1,1)
        self.ax1.axis('off')
        self.ax1.imshow(self.merge)
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Annotator")
    parser.add_argument('--checkpoint',
                        type=str,
                        default='./weights/hrnet32_ocr128_lvis.pth',
                        help='The path to the checkpoint.')
    parser.add_argument('--gpu_id',
                        type=int,
                        default=0,
                        help='Id of GPU to use.')

    parser.add_argument('--mode',
                        type=str,
                        default='f-BRS-B',
                        help='f-BRS mode.')

    parser.add_argument('-vis', '--visualize',
                        action='store_true',
                        default=False,
                        help='Visualize output')

    parser.add_argument('--cfg',
                        type=str,
                        default="config.yml",
                        help='The path to the config file.')

    parser.add_argument('--input',
                        type=str,
                        default='test.jpg',
                        help='input image')

    parser.add_argument('--output',
                        type=str,
                        default='test_mask.png',
                        help='output mask')

    return parser.parse_args()


def init(args):
    # Parameters
    EVAL_MAX_CLICKS = 20
    MODEL_THRESH = 0.5

    # NOTE: image_list to be verified
    img_list = glob(r'./images/*')

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
        'image': './images/bear.jpg',
        'img_list': img_list,
        'checkpoint': args.checkpoint,
        'gpu_id': args.gpu_id,
        'params': params,
        'click_list': click_list,
        'prob_thresh': MODEL_THRESH
    })

    return cfg

if __name__ == "__main__":
    # Prepare config
    args = parse_args()
    cfg = init(args)

    # Instantiate the model handler
    controller = ModelHandler(
        model_path=cfg.checkpoint,
        gpu_id=cfg.gpu_id,
        predictor_params=cfg.params,
        prob_thresh=cfg.prob_thresh
    )

    # Interactive UI
    anno = Annotator(
        img_path=cfg.image,
        img_list=cfg.img_list,
        controller=controller,
        predictor_params=cfg.params,
        save_path="./results/annotated_result.jpg"
    )

    anno.main()

