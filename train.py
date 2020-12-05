import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from glob import glob
from tqdm import tqdm

from models.micro import FirstCNN
from models.apps import app_net

from losses import pairwise_loss, triplet_loss
from utils.preparations import channelSwap, normalize, resize
from preprocess import load_and_preprocess_image

def main(
        train_list,
        test_list,
        batchsize = 32, 
        insize = 32,
        outsize = 128,
        model_name = 'first',
        epsilon = 1e+5
    ):
    # load data
    def preprocess(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [insize, insize])
        image /= 255.0
        return image

    train_names = load_list(train_list)
    train_dataset_paths = tf.data.Dataset.from_tensor_slices(train_names)
    train_dataset = train_dataset_paths.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_names = load_list(test_list)
    test_dataset_paths = tf.data.Dataset.from_tensor_slices(test_names)
    test_dataset = test_dataset_paths.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    for dataset in [train_dataset, test_dataset]:
        dataset = dataset.shuffle(buffer_size=len(list(dataset)))
        dataset = dataset.repeat()
        dataset = dataset.batch(batchsize)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    # load model
    app_net_list = ['mobilenet', 'resnet50', 'resnet101', 'resnet152']
    if model_name == 'first':
        model = FirstCNN(insize=32, outsize=128)
    elif model_name in app_net_list:
        model = app_net(
            archname = model_name,
            input_shape = (insize, insize, 3),
            z_dim = outsize
        )
    else:
        raise ValueError('Invalid model_name: {}'.format(model_name))

    # load loss functions
    if loss_name == 'pairwise':
        #@tf.function
        def loss_func(x, y, equal):
            return pairwise_loss(x, y, equal, epsilon=epsilon)
    elif loss_name == 'triplet':
        #@tf.function
        def loss_func(x, y, z):
            return triplet_loss(x, y, z, epsilon=epsilon)
    elif loss_name == 'arcface':
        raise NotImplementedError()
    else:
        raise ValueError('Invalid loss_name: {}'.format(loss_name))

    @tf.function
    def train_step(x, y):
        x_ = model(x)
        y_ = model(y)
        

    # save config

    # prepare training

    # execute train

def load_list(path):
    with open(path) as f:
        lines = f.readlines()
    lines = list(map(lambda x: x.strip(), lines))
    return lines
