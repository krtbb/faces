import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from glob import glob

from models.apps import FaceModel
from utils.analysis import load_model
from utils.generals import load_json

class Encoder(object):
    def __init__(self, log_dir, epoch=-1):
        self.log_dir = log_dir
        
        # load configs
        json_path = os.path.join(self.log_dir, 'train_config.json')
        saved_params_paths = sorted(glob(os.path.join(self.log_dir, 'model_*.h5')))
        if epoch == -1:
            saved_params_paths = sorted(glob(os.path.join(self.log_dir, 'model_*.h5')))
            saved_params_path = saved_params_paths[-1]
        elif epoch >= 0:
            saved_params_path = os.path.join(self.log_dir, 'model_{}.h5'.format(epoch))
        
        # restore models
        config = load_json(json_path)
        self.model = load_model(config)
        self.model.load_weights(saved_params_path)
        self.insize = self.model.input_shape[1]
        self.outsize = self.model.output_shape[1]

    def __call__(self, imgs, batchsize=None, verbose=0, use_multiprocessing=False):
        return self.model.predict(imgs, batch_size=batchsize, verbose=verbose, use_multiprocessing=use_multiprocessing)

        