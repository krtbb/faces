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
from models.apps import FaceModel, app_net

from utils.losses import pairwise_loss, triplet_loss
from utils.preparations import channelSwap, normalize, resize
from utils.preprocess import load_and_preprocess_image

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
        label = tf.strings.split(path, sep='/')[-2]
        label = tf.strings.to_number(label, tf.int32)
        return label

    def get_num_classes(*args):
        path = args[0][0]
        data_root_path = '/'.join(path.split('/')[:-2])
        return len(os.listdir(data_root_path))

    train_names = load_list(train_list)
    test_names = load_list(test_list)
    if debug:
        train_names = train_names[:64]
        test_names = test_names[:64]
    num_classes = get_num_classes(train_names, test_names)

    train_dataset_paths = tf.data.Dataset.from_tensor_slices(train_names)
    train_dataset_image = train_dataset_paths.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset_label = train_dataset_paths.map(preprocess_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset_x = tf.data.Dataset.zip((train_dataset_image, train_dataset_label))
    train_dataset_x = train_dataset_x.shuffle(buffer_size=len(list(train_dataset_x)))
    train_dataset_x = train_dataset_x.repeat()
    train_dataset_x = train_dataset_x.batch(batchsize)
    train_dataset_x = train_dataset_x.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    test_dataset_paths = tf.data.Dataset.from_tensor_slices(test_names)
    test_dataset_image = test_dataset_paths.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset_label = test_dataset_paths.map(preprocess_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset_x = tf.data.Dataset.zip((test_dataset_image, test_dataset_label))
    test_dataset_x = test_dataset_x.shuffle(buffer_size=len(list(test_dataset_x)))
    test_dataset_x = test_dataset_x.repeat()
    test_dataset_x = test_dataset_x.batch(batchsize)
    test_dataset_x = test_dataset_x.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    data_augmentation = tf.keras.Sequential([
        #tf.keras.layers.experimental.preprocessing.RandomFlip('vertical'),
        #tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    ])
    print('Finished.')
    
    # Load model
    print('Defining model...', end='')
    model = FaceModel(size=insize, channels=3, z_dim=outsize,
                      backbone_type=model_name, use_pretrain=False,
                      w_decay=5e-4, name='facemodel')
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
    optimizer = tf.keras.optimizers.Adam()
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
    if not os.path.exists('logs/{}'.format(training_id)):
        os.makedirs('logs/{}'.format(training_id))
    with open('logs/{}/train_config.json'.format(training_id), 'w') as f:
        json.dump(config, f)

    # prepare training

    # execute train
    if loss_name in ['pairwise', 'triplet']:
        train_dataset_y = copy.deepcopy(train_dataset_x)
        test_dataset_y = copy.deepcopy(test_dataset_x)
    if loss_name in ['triplet']:
        train_dataset_z = copy.deepcopy(train_dataset_x)
        test_dataset_z = copy.deepcopy(test_dataset_x)
    header = 'epoch trainloss, testloss'
    template = '{} {:.6f} {:.6f}'
    with open('logs/{}/history.csv'.format(training_id), 'w') as f:
        f.write(header + '\n')
    print('Finished.')

    print('Start training.')
    for epoch in range(epochs):
        if loss_name == 'pairwise':
            for (x_images, x_labels), (y_images, y_labels) in zip(train_dataset_x, train_dataset_y):
                train_step(x_images, y_images, x_labels==y_labels)
            for (x_images, x_labels), (y_images, y_labels) in zip(test_dataset_x, test_dataset_y):
                test_step(x_images, y_images, x_labels==y_labels)
        elif loss_name == 'triplet':
            for (x_images, x_labels), (y_images, y_labels), (z_images, z_labels) in zip(train_dataset_x, train_dataset_y, train_dataset_z):
                train_step(x_images, y_images, z_images)
            for (x_images, x_labels), (y_images, y_labels), (z_images, z_labels) in zip(test_dataset_x, test_dataset_y, test_dataset_z):
                test_step(x_images, y_images, z_images)
        elif loss_name == 'arcface':
            for images, labels in train_dataset_x:
                train_step(images, labels)
            for images, labels in test_dataset_x:
                test_step(images, labels)

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
    parser.add_argument('--modelname', '-M', type=str, default='ResNet50', help='name of model architecture')
    parser.add_argument('--lossname', '-L', type=str, default='pairwise', help='name of loss function')
    parser.add_argument('--epochs', '-E', type=int, default=100)
    parser.add_argument('--batchsize', '-B', type=int, default=32)
    parser.add_argument('--insize', '-SI', type=int, default=32, help='size of input images, int, default=32')
    parser.add_argument('--outsize', '-SO', type=int, default=128, help='size of embedded variables, int, default=128')
    parser.add_argument('--epsilon', '-P', type=float, default=1e+5, help='constants for margins in loss')
    parser.add_argument('--debug', action='store_true')
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
        epsilon = args.epsilon,
        debug = args.debug
    )

