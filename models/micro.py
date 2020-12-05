import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
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
            self.dropout1 = Dropout(0.1)
            self.conv2 = Conv2D(64, 4, strides=1, paddings='same', activation=activation) # (16, 16, 64)
            self.shrink2 = shrink(64, 4, 2) # (8, 8, 64) = 4096
            self.dropout2 = Dropout(0.1)
            self.conv3 = Conv2D(64, 3, strides=1, paddings='same', activation=activation) # (8, 8, 32)
            self.shrink3 = shrink(64, 3, 2) # (4, 4, 64) = 1024
        else:
            raise ValueError('Invalid insize: {}'.format(insize))
        
        self.l1 = Dense(1024, activation=activation)
        self.l2 = Dense(512, activation=activation)
        self.l3 = Dense(256, activation=activation)
        self.l4 = Dense(self.outsize, activation=activation)

    def __call__(self, x):
        h = self.dropout1(self.shrink1(self.conv1(x)))
        h = self.dropout2(self.shrink2(self.conv2(x)))
        h = self.shrink3(self.conv3(x))
        h = Flatten()(h)
        h = self.l1(h)
        h = self.l2(h)
        h = self.l3(h)
        y = self.l4(h)

        return y
        