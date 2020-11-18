import argparse
import logging
import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from fbrs.isegm.inference import utils
from core.model_controller import InteractiveController
from glob import glob


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

random.seed(878)


#Forbidden  Key: QSFKL
class Annotator(object):
    def __init__(self, img_path, model,
                 img_list=None,
                 controller=None,
                 predictor_params=None,
                 save_path=None):
        assert predictor_params is not None, "Predictor parameters is None."
        assert controller is not None, "Controller is None."

        self.model, self.save_path = model, save_path
        self.file = Path(img_path).name
        self.img = np.array(Image.open(img_path))
        self.clicks = np.empty([0,3],dtype=np.int64)
        self.pred = np.zeros(self.img.shape[:2],dtype=np.uint8)
        self.merge =  self.__gene_merge(self.pred, self.img, self.clicks)
        self.predictor_params = predictor_params

        # TODO:
        self.controller = controller
        self.controller.set_image(self.img)
        self.img_list = [img_path] if img_list is None else img_list
        self.img_cnt = 0

    def __gene_merge(self,pred,img,clicks,r=9,cb=2,b=2,if_first=True):
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
        # TODO:
        self.controller.set_image(self.img)
        self.controller.reset_last_object(self.predictor_params)
        self.__update()

    def __predict(self):
        # TODO:
        self.pred = (self.controller.current_object_prob > 0.5).astype(np.uint8) * 255
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
            self.controller.undo_click() # TODO:
            if len(self.clicks)!=0:
                self.__predict()
            else:
                self.__reset()
        elif event.key=='ctrl+r':
            self.controller.reset_last_object(self.predictor_params) # TODO:
            self.__reset()
        elif event.key=='escape':
            plt.close()
        elif event.key=='enter':
            if self.save_path is not None:
                #Image.fromarray(self.pred*255).save(self.save_path)
                Image.fromarray(self.pred).save(self.save_path)
                print('save mask in [{}]!'.format(self.save_path))
            plt.close()

    def __on_button_press(self,event):
        if (event.xdata is None) or (event.ydata is None):return
        if event.button==1 or  event.button==3:
            x,y= int(event.xdata+0.5), int(event.ydata+0.5)
            self.clicks=np.append(self.clicks,np.array([[x,y,(3-event.button)/2]],dtype=np.int64),axis=0)
            # TODO:
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
    parser.add_argument('--checkpoint', type=str,
                        default='./fbrs/weights/hrnet32_ocr128_lvis.pth',
                        help='The path to the checkpoint. '
                             'This can be a relative path (relative to cfg.INTERACTIVE_MODELS_PATH) '
                             'or an absolute path. The file extension can be omitted.')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='Id of GPU to use.')

    parser.add_argument('--mode', type=str, default='f-BRS-B',
                        help='f-BRS mode.')

    parser.add_argument('-vis', '--visualize', action='store_true',
                        default=False,
                        help='Visualize output')
    parser.add_argument('--cfg', type=str, default="config.yml",
                        help='The path to the config file.')
    parser.add_argument('--input', type=str, default='test.jpg', help='input image')
    parser.add_argument('--output', type=str, default='test_mask.png', help='output mask')

    return parser.parse_args()


def init(args):
    # Possible choices: 'NoBRS', 'f-BRS-A', 'f-BRS-B', 'f-BRS-C', 'RGB-BRS', 'DistMap-BRS'
    USE_ZOOM_IN = False
    MAX_CLICKS = 20
    MODEL_THRESH = 0.5
    MODE = args.mode

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
        'zoom_in_params': {
            'use_zoom_in': USE_ZOOM_IN,
            'skip_clicks': 1,
            'target_size': 600,
            'expansion_ratio': 1.4
        },
        'predictor_params': {
            'net_clicks_limit': MAX_CLICKS
        },
        'brs_mode': MODE,
        'prob_thresh': MODEL_THRESH,
        #'lbfgs_max_iters': 20
    }

    # NOTE: image_list to be verified
    img_list = ['images/17790319373_bd19b24cfc_k.jpg', 'images/sheep.jpg', 'images/bear.jpg']
    img_list = glob(r'./images/50/*')
    # img_list = glob(r'/home/cilin/pytorch_repo/pycococreator/projects/human17/images/official_125/125_100_00-01_00/*') # blur, dark

    ret_dict = dict(
        net=model,
        device=device,
        predictor_params=params,
        prob_thresh=MODEL_THRESH,
        img_list=img_list
    )
    return ret_dict


if __name__ == "__main__":
    args = parse_args()
    cfg = init(args)
    controller = InteractiveController(**cfg)
    img_path = './images/frame0025.jpg'

    anno = Annotator(
        img_path=img_path,
        img_list=cfg['img_list'],
        model=cfg['net'],
        controller=controller,
        predictor_params=cfg['predictor_params'],
        save_path="./results/annotated_result.jpg"
    )

    anno.main()

