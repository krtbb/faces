import tensorflow as tf

from tensorflow.keras import applications as app
from tensorflow.keras import layers as L

from arcface_modules.models import Backbone, OutputLayer, ArcHead, NormHead

def FaceModel(size=64, channels=3, z_dim=128,
              backbone_type = 'ResNet50',
              use_pretrain = True,
              w_decay = 5e-4,
              name = 'facemodel'):
    """Face Feature Embedding Model"""
    x = inputs = L.Input([size, size, channels], name='input_image')

    x = Backbone(backbone_type=backbone_type, use_pretrain=use_pretrain)(x)

    z = OutputLayer(z_dim, w_decay=w_decay)(x)

    return tf.keras.models.Model(inputs, z, name=name)

def app_net(
        archname = 'mobilenet',
        z_dim = 128,
        input_shape=(160, 160, 3)
    ):
    if archname == 'mobilenet':
        basemodel = app.MobileNetV2(
            input_shape = input_shape,
            include_top = False,
            weights = 'imagenet'
        )
    elif archname == 'resnet50':
        basemodel = app.ResNet50(
            input_shape = input_shape,
            include_top = False,
            weights = 'imagenet'
        )
    elif archname == 'resnet101':
        basemodel = app.ResNet101(
            input_shape = input_shape,
            include_top = False,
            weights = 'imagenet'
        )
    elif archname == 'resnet152':
        basemodel = app.ResNet152(
            input_shape = input_shape,
            include_top = False,
            weights = 'imagenet'
        )
    else:
        raise ValueError('Invalid archname: {}'.format(archname))

    gal = L.GlobalAveragePooling2D()
    dense = L.Dense(z_dim)

    inputs = tf.keras.Input(shape=input_shape)
    h = basemodel(inputs, training=True)
    h = gal(h)
    h = dense(h)
    outputs = tf.tanh(h)
    
    model = tf.keras.Model(inputs, outputs)

    return model
