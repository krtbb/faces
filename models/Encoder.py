import cv2
import numpy as np
import os
import sys
import tensorflow as tf

class Encoder(object):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        
        savedmodel_path = os.path.join(self.log_dir, 'model.saved_model')
        self.model = tf.saved_model.load(savedmodel_path)