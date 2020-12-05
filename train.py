import copy
import cv2
import datetime
import json
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

def load_list(path):
    with open(path) as f:
        lines = f.readlines()
    lines = list(map(lambda x: x.strip(), lines))
    return lines

def main(
        train_list,
        test_list,
        epochs,
        batchsize = 32, 
        insize = 32,
        outsize = 128,
        model_name = 'first',
        loss_name = 'pairwise',
        epsilon = 1e+5
    ):
    training_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Load data
    def preprocess(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [insize, insize])
        image /= 255.0
        return image

    train_names = load_list(train_list)
    train_dataset_paths = tf.data.Dataset.from_tensor_slices(train_names)
    train_dataset = train_dataset_paths.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset_x = tf.data.Dataset.zip((train_dataset, list(map(lambda x: x.split('/')[-2], train_names))))
    test_names = load_list(test_list)
    test_dataset_paths = tf.data.Dataset.from_tensor_slices(test_names)
    test_dataset = test_dataset_paths.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset_x = tf.data.Dataset.zip((test_dataset, list(map(lambda x: x.split('/')[-2], test_names))))
    for dataset in [train_dataset_x, test_dataset_x]:
        dataset = dataset.shuffle(buffer_size=len(list(dataset)))
        dataset = dataset.repeat()
        dataset = dataset.batch(batchsize)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('vertical'),
        #tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    ])
    
    # Load model
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

    # Load loss functions
    if loss_name == 'pairwise':
        def loss_func(x, y, equal):
            return pairwise_loss(x, y, equal, epsilon=epsilon)
    elif loss_name == 'triplet':
        def loss_func(x, y, z):
            return triplet_loss(x, y, z, epsilon=epsilon)
    elif loss_name == 'arcface':
        raise NotImplementedError()
    else:
        raise ValueError('Invalid loss_name: {}'.format(loss_name))

    # Define graph
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    if loss_name == 'pairwise':
        @tf.function
        def train_step(x, y, equal):
            with tf.GradientTape() as tape:
                x_ = model(x)
                y_ = model(y)
                loss = loss_func(x_, y_, equal)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(loss)
        @tf.function
        def test_step(x, y, equal):
            x_ = model(x)
            y_ = model(y)
            loss = loss_func(x_, y_, equal)
            test_loss(loss)

    # save config
    config = {}
    config['train_list'] = train_list
    config['test_list'] = test_list
    config['batchsize'] = batchsize
    config['epoch'] = epochs
    config['insize'] = insize
    config['outsize'] = outsize
    config['model_name'] = model_name
    config['loss_name'] = loss_name
    config['epsilon'] = epsilon
    with open('logs/{}/train_config.json'.format(training_id), 'w') as f:
        json.dump(config, f)

    # prepare training

    # execute train
    train_dataset_y = copy.deepcopy(train_dataset_x)
    test_dataset_y = copy.deepcopy(test_dataset_x)
    if loss_name == 'triplet':
        train_dataset_z = copy.deepcopy(train_dataset_x)
        test_dataset_z = copy.deepcopy(test_dataset_x)
    header = 'epoch trainloss, testloss'
    template = '{} {:.6f} {:.6f}'
    with open('logs/{}/history.csv'.format(training_id), 'w') as f:
        f.write(header + '\n')
    for epoch in range(epochs):     
        for (x_images, x_names), (y_images, y_names) in zip(train_dataset_x, train_dataset_y):
            train_step(x_images, y_images, x_names==y_names)
        for (x_images, x_names), (y_images, y_names) in zip(test_dataset_x, test_dataset_y):
            test_step(x_images, y_images, x_names==y_names)
        print('{} |'.format(epochs) + template.format(epoch+1, train_loss.result(), test_loss.result()))
        with open('logs/{}/history.csv'.format(training_id), 'a') as f:
            f.write(template.format(epoch+1, train_loss.result(), test_loss.result())+'\n')
        
        train_loss.reset_states()
        test_loss.reset_states()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('train_list', type=str, help='list for train data')
    parser.add_argument('test_list', type=str, help='list for test data')
    parser.add_argument('--modelname', '-M', type=str, default='resnet50', help='name of model architecture')
    parser.add_argument('--lossname', '-L', type=str, default='pairwise', help='name of loss function')
    parser.add_argument('--epochs', '-E', type=int, default=100)
    parser.add_argument('--batchsize', '-B', type=int, default=32)
    parser.add_argument('--insize', '-SI', type=int, default=32, help='size of input images, int, default=32')
    parser.add_argument('--outsize', '-SO', type=int, default=128, help='size of embedded variables, int, default=128')
    parser.add_argument('--epsilon', '-P', type=float, default=1e+5, help='constants for margins in loss'
    args = parser.parse_args()

    main(
        args.train_list,
        args.test_list,
        args.epochs,
        batchsize = args.batchsize,
        insize = args.insize,
        outsize = args.outsize,
        model_name = args.modelname,
        loss_name = args.lossname,
        epsilon = args.epsilon
    )

