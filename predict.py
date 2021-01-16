import cv2
import numpy as np
import os
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
tf.enable_eager_execution(config=config)

from models.Encoder import Encoder

from utils.generals import load_json, load_list

def calc_distance_sum(array, labels):
    distance_sum = 0
    for i in range(len(array)):
        for j in range(len(array)):
            if j <= i:
                continue
            if labels[i] == labels[j]:
                distance_sum += np.sqrt(np.sum((array[i]-array[j])**2))
            else:
                distance_sum += 1 / np.sqrt(np.sum((array[i]-array[j])**2))
    return distance_sum

def predict(list_path, log_dir, epoch):
    # load_list
    eval_names = load_list(list_path)

    # prepare model
    if 'logs' in log_dir:
        json_path = os.path.join(log_dir, 'train_config.json')
        config = load_json(json_path)
        encoder = Encoder(log_dir, epoch=-1)
    else:
        raise ValueError('Invalid log_dir(`logs` is not in log_dir')

    # prepare dataset
    images = np.zeros((len(eval_names), encoder.insize, encoder.insize, 3))
    labels = np.array(list(map(lambda x: x.split('/')[-2], eval_names)))
    for i, name in enumerate(eval_names):
        bgr = cv2.imread(name)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (encoder.insize, encoder.insize))
        images[i] = resized
    images /= 255.

    # execute evaluation
    encodings = encoder(images[:200])
    distance = calc_distance_sum(encodings, labels)
    
    return distance

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('log_dir', help='target directory for trained parameters')
    parser.add_argument('list', help='target list of image paths')
    parser.add_argument('--epoch', default=-1, type=int, help='epoch, default=-1, using latest log')
    args = parser.parse_args()

    distance = predict(args.list, args.log_dir, args.epoch)
    print(distance)