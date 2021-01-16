import copy
import cv2
import datetime
import json
import numpy as np
import os
import sys
import tensorflow as tf
import datetime

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)

from glob import glob
from tqdm import tqdm

from models.Encoder import Encoder

from utils.losses import pairwise_loss, triplet_loss
from utils.preparations import channelSwap, normalize, resize
from utils.preprocess import load_and_preprocess_image

def predict(log_dir, epoch):
    # prepare model
    if 'logs' in log_dir:
        encoder = Encoder(log_dir, epoch=-1)
    else:
        raise ValueError('Invalid log_dir(`logs` is not in log_dir')

    # prepare dataset

    resized = cv2.resize(img, (encoder.insize, encoder.insize))
    
    # execute evaluation