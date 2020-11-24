import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from glob import glob
from tqdm import tqdm

from models.gen1 import FirstCNN
from models.mobilenet import mobilenet

def main(
        train_list,
        test_list,
    ):
    # load data
    train_names = load_list(train_list)
    test_names = load_list(test_list)

    # preprocess

    # load model

    # save config

    # prepare training

    # execute train

def load_list(path):
    with open(path) as f:
        lines = f.readlines()
    lines = list(map(lambda x: x.strip(), lines))
    return lines
