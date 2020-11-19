import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Model

class FirstCNN(Model):
    """
    refered: https://github.com/Vishwesh4/Face-Feature-Extraction
    conv2d (93, 93, 32)
    maxpool (46, 46, 32)
    dropout
    conv2d (44, 44, 64)
    maxpool (22, 22, 64)
    dropout
    conv2d (21, 21, 128)
    maxpool (10, 10, 128)
    conv2d (10, 10, 256)
    maxpool (5, 5, 256)
    flatten (6400)
    dense 1000
    dense 1000
    dense 30

    inputs should be (32, 32, 3) due to outputs of running system.
    conv2d (32, 32, 128)
    maxpool 
    """
    def __init__(
            self, 
            insize = 32, 
            outsize = 128,
            shrink_method = 'maxpool',
            activation = 'relu'
            ):
        super(FirstCNN, self).__init__()
        self.insize = 32
        self.outsize = 128

        if shrink_method == 'maxpool':
            def shrink(a, b, s):
                return MaxPooling2D(s)
        elif shrink_method == 'conv':
            def shrink(c, k, s):
                return Conv2D(c, k, strides=s, padding='same', activation=activation)
        else:
            raise ValueError('Invalid shrink_method: {}'.format(shrink_method))
        
        if insize == 32:
            self.conv1 = Conv2D(128, 5, strides=1, paddings='same', activation=activation) # (32, 32, 128)
            self.shrink1 = shrink(128, 5, 2) # (16, 16, 128)
            self.conv2 = Conv2D(64, 4, strides=1, paddings='same', activation=activation) # (16, 16, 64)
            self.shrink2 = shrink(64, 4, 2) # (8, 8, 64) = 4096
        else:
            raise ValueError('Invalid insize: {}'.format(insize))
        
        self.l1 = Dense()


    def __call__(self)
        return 0
        