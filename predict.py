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

from models.micro import FirstCNN
from models.apps import FaceModel, app_net

from utils.losses import pairwise_loss, triplet_loss
from utils.preparations import channelSwap, normalize, resize
from utils.preprocess import load_and_preprocess_image

