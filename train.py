"""
TODO: 
  * add evaluation metrics
  * add data augmentation
  * add optimizer scheduler
  * add other models
"""

import copy
import cv2
import datetime
import json
import numpy as np
import os
import sys
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
tf.enable_eager_execution(config=config)

from glob import glob
from tqdm import tqdm

from models.micro import FirstCNN
from models.apps import FaceModel, app_net

from utils.analysis import load_model
from utils.Augment import Augmentor
from utils.losses import pairwise_loss, triplet_loss
from utils.preparations import channelSwap, normalize, resize
from utils.preprocess import load_and_preprocess_image
from utils.generals import load_list

def main(
        train_list,
        test_list,
        epochs,
        batchsize = 32, 
        lr = 0.001,
        insize = 32,
        outsize = 128,
        model_name = 'first',
        loss_name = 'pairwise',
        epsilon = 1e+5,
        debug = False
    ):
    training_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Load data
    print('Load data...', end='')
    def preprocess_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [insize, insize])
        image /= 255.0
        return image

    def preprocess_label(path):
        label = tf.strings.split(path, sep='/', result_type='RaggedTensor')[-2]
        label = tf.strings.to_number(label, tf.int32)
        return label

    def get_num_classes(*args):
        path = args[0][0]
        data_root_path = '/'.join(path.split('/')[:-2])
        return len(os.listdir(data_root_path))

    def generate_paths_dataset(data_names, batchsize, batch=True):
        data_num = len(data_names)
        dataset_paths = tf.data.Dataset.from_tensor_slices(data_names)
        if batch:
            dataset_paths = dataset_paths.shuffle(buffer_size=data_num)
            dataset_paths = dataset_paths.batch(batchsize)
            dataset_paths = dataset_paths.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset_paths
    
    def load_images_dataset(ds_paths, batchsize, data_num):
        dataset_image = ds_paths.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset_label = ds_paths.map(preprocess_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = tf.data.Dataset.zip((dataset_image, dataset_label))
        dataset = dataset.shuffle(buffer_size=data_num)
        dataset = dataset.batch(batchsize)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    def generate_images_dataset(data_names, batchsize):
        dataset_paths = generate_paths_dataset(data_names, batchsize, batch=False)
        dataset = load_images_dataset(dataset_paths, batchsize, len(data_names))
        return dataset

    train_names = load_list(train_list)
    test_names = load_list(test_list)
    if debug:
        train_names = train_names[:64]
        test_names = test_names[:64]
    num_classes = get_num_classes(train_names, test_names)
    print('Use {} traindata, {} testdata.'.format(len(train_names), len(test_names)))

    train_dataset_x = generate_images_dataset(train_names, batchsize)
    test_dataset_x = generate_images_dataset(test_names, batchsize)
    if loss_name in ['pairwise', 'triplet']:
        train_dataset_y = generate_images_dataset(train_names, batchsize)
        test_dataset_y = generate_images_dataset(test_names, batchsize)
    if loss_name in ['triplet']:
        train_dataset_z = generate_images_dataset(train_names, batchsize)
        test_dataset_z = generate_images_dataset(test_names, batchsize)

    #augmentor = Augmentor(use_brightness=False, use_darkness=False)
    #def rotate(img):
    #    return tf.keras.preprocessing.image.random_rotation(img, 30, row_axis=0, col_axis=1, channel_axis=2)
    #augmentor.funcs.append(rotate)
    augmentor = lambda x: x

    print('Finished.')
    
    # Load model
    print('Defining model...', end='')
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
    model = load_model(config)
    
    print('Finished.')
    model.summary()

    # Load loss functions
    print('Defining loss function...', end='')
    if loss_name == 'pairwise':
        METRICS = 'distance'
        def loss_func(x, y, equal):
            return pairwise_loss(x, y, equal, epsilon=epsilon)
    elif loss_name == 'triplet':
        METRICS = 'distance'
        def loss_func(x, y, z):
            return triplet_loss(x, y, z, epsilon=epsilon)
    elif loss_name == 'arcface':
        METRICS = 'classification'
        from arcface_modules.models import ArcHead
        from arcface_modules.losses import SoftmaxLoss
        margin = 0.5
        logist_scale = 64
        archead = ArcHead(num_classes=num_classes, margin=margin,
                          logist_scale=logist_scale)
        softmax_fn = SoftmaxLoss()
        def loss_func(z, labels):
            logist = archead(z, labels)
            return softmax_fn(labels, logist)
        
    else:
        raise ValueError('Invalid loss_name: {}'.format(loss_name))
    print('Finished.')

    # Define graph
    print('Defining train_step()...', end='')
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    
    if METRICS == 'distance':
        assert loss_name in ['pairwise', 'triplet']

        # pairwise operations
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

        # triplet operations
        elif loss_name == 'triplet':
            @tf.function
            def train_step(x, y, z):
                with tf.GradientTape() as tape:
                    x_ = model(x)
                    y_ = model(y)
                    z_ = model(z)
                    loss = loss_func(x_, y_, z_)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                train_loss(loss)

            @tf.function
            def test_step(x, y, z):
                x_ = model(x)
                y_ = model(y)
                z_ = model(z)
                loss = loss_func(x_, y_, z_)
                test_loss(loss)

    elif METRICS == 'classification':

        # arcface operations
        #@tf.function
        def train_step(x, labels):
            with tf.GradientTape() as tape:
                x_ = model(x)
                loss = loss_func(x_, labels)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(loss)
        
        #@tf.function
        def test_step(x, labels):
            x_ = model(x)
            loss = loss_func(x_, labels)
            test_loss(loss)
    print('Finished.')

    print('Preparing training configure...', end='')
    # save config
    if not os.path.exists('logs/{}'.format(training_id)):
        os.makedirs('logs/{}'.format(training_id))
    with open('logs/{}/train_config.json'.format(training_id), 'w') as f:
        json.dump(config, f)

    # execute train
    header = 'epoch time trainloss testloss'
    template = '{} {} {:.12f} {:.12f}'
    with open('logs/{}/history.csv'.format(training_id), 'w') as f:
        f.write(header + '\n')
    print('Finished.')

    print('Start training.')
    ts_old = datetime.datetime.now()
    previous_saved_ts = datetime.datetime.now()
    for epoch in range(epochs):
        if loss_name == 'pairwise':
            for (x_images, x_labels), (y_images, y_labels) in zip(train_dataset_x, train_dataset_y):
                train_step(augmentor(x_images), auguentor(y_images), x_labels==y_labels)
            for (x_images, x_labels), (y_images, y_labels) in zip(test_dataset_x, test_dataset_y):
                test_step(x_images, y_images, x_labels==y_labels)
        elif loss_name == 'triplet':
            raise NotImplementedError()
            # TODO 
            # * arguments for train_step() are Invalid.
            for (x_images, x_labels), (y_images, y_labels), (z_images, z_labels) in tqdm(zip(train_dataset_x, train_dataset_y, train_dataset_z)):
                train_step(augmentor(x_images), augmentor(y_images), augmentor(z_images))
            for (x_images, x_labels), (y_images, y_labels), (z_images, z_labels) in zip(test_dataset_x, test_dataset_y, test_dataset_z):
                test_step(x_images, y_images, z_images)
        elif loss_name == 'arcface':
            #for images, labels in train_dataset_x:
            for images, labels in train_dataset_x:
                train_step(augmentor(images), labels)
            for images, labels in test_dataset_x:
                test_step(images, labels)

        # logging
        ts_new = datetime.datetime.now()
        print('{} |'.format(epochs) + template.format(epoch+1, ts_new-ts_old, train_loss.result(), test_loss.result()))
        with open('logs/{}/history.csv'.format(training_id), 'a') as f:
            f.write(template.format(epoch+1, ts_new-ts_old, train_loss.result(), test_loss.result())+'\n')
        ts_old = datetime.datetime.now()

        # saving models
        if epoch==0 or (ts_new - previous_saved_ts).total_seconds() > 3600:
            save_dir = 'logs/{}'.format(training_id)
            tf.saved_model.save(model, '{}/model.saved_model'.format(save_dir))
            model.save_weights('{}/model_{}.h5'.format(save_dir, epoch+1))
            previous_saved_ts = datetime.datetime.now()

        # reset loss states
        train_loss.reset_states()
        test_loss.reset_states()
    
    # save final result
    model.save_weights('{}/model_final.h5'.format(save_dir))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('train_list', type=str, help='list for train data')
    parser.add_argument('test_list', type=str, help='list for test data')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--modelname', '-M', type=str, default='ResNet50', help='name of model architecture')
    parser.add_argument('--lossname', '-L', type=str, default='pairwise', help='name of loss function')
    parser.add_argument('--epochs', '-E', type=int, default=100)
    parser.add_argument('--batchsize', '-B', type=int, default=16)
    parser.add_argument('--insize', '-SI', type=int, default=32, help='size of input images, int, default=32')
    parser.add_argument('--outsize', '-SO', type=int, default=128, help='size of embedded variables, int, default=128')
    parser.add_argument('--epsilon', '-P', type=float, default=1e+5, help='constants for margins in loss')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    main(
        args.train_list,
        args.test_list,
        args.epochs,
        lr = args.lr,
        batchsize = args.batchsize,
        insize = args.insize,
        outsize = args.outsize,
        model_name = args.modelname,
        loss_name = args.lossname,
        epsilon = args.epsilon,
        debug = args.debug
    )

