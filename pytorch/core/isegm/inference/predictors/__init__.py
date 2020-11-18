from .base import BasePredictor


def get_predictor(net, device,
                     with_flip=True,
                     predictor_params=None):

    predictor = BasePredictor(net,
                              device,
                              with_flip=with_flip,
                              **predictor_params)

    return predictor

