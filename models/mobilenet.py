import tensorflow as tf

from tensorflow.keras import applications as app
from tensorflow.keras import layers as L

def mobilenet(input_shape=(160, 160, 3)):
    basemodel = app.MobileNetV2(
        input_shape = input_shape,
        include_top = False,
        weights = 'imagenet'
    )
    gal = L.GlobalAveragePooling2D()
    dense = L.Dense(128)

    inputs = tf.keras.Input(shape=input_shape)
    h = basemodel(inputs, training=True)
    h = gal(h)
    outputs = dense(h)
    
    model = tf.keras.Model(inputs, outputs)

    return model