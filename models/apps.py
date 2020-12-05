import tensorflow as tf

from tensorflow.keras import applications as app
from tensorflow.keras import layers as L

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