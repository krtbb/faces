import tensorflow as tf
import math


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """Make trainable=False freeze BN for real (the og version is sad).
       ref: https://github.com/zzh8829/yolov3-tf2
    """
    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


class ArcMarginPenaltyLogists(tf.keras.layers.Layer):
    """ArcMarginPenaltyLogists"""
    def __init__(self, num_classes, margin=0.5, logist_scale=64, **kwargs):
        ### define hyper parameters which are independent to input data
        super(ArcMarginPenaltyLogists, self).__init__(**kwargs)
        self.num_classes = num_classes # int
        self.margin = margin # float
        self.logist_scale = logist_scale # int

    def build(self, input_shape):
        ### define parameters like w, b, and so on.
        self.w = self.add_weight(
            "weights", shape=[int(input_shape[-1]), self.num_classes]) # [dim, classes]
        self.cos_m = tf.identity(math.cos(self.margin), name='cos_m') # float
        self.sin_m = tf.identity(math.sin(self.margin), name='sin_m') # float
        self.th = tf.identity(math.cos(math.pi - self.margin), name='th') # float
        self.mm = tf.multiply(self.sin_m, self.margin, name='mm')  # float

    def call(self, embds, labels):
        ### define forward propagations

        # tf.nn.l2.normalize -> lambda x: sqrt(max(sum(x**2), epsilon))
        normed_embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd') # [, dim]
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights') # [dim, classes]

        cos_t = tf.matmul(normed_embds, normed_w, name='cos_t') # [classes]
        sin_t = tf.sqrt(1. - cos_t ** 2, name='sin_t') # [classes]

        # tf.subtract -> lambda x, y: x - y
        cos_mt = tf.subtract(
            cos_t * self.cos_m, sin_t * self.sin_m, name='cos_mt') # [classes]

        # tf.where(cond, x, y) -> return x if (cond) else y
        cos_mt = tf.where(cos_t > self.th, cos_mt, cos_t - self.mm) # [classes]

        mask = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes,
                          name='one_hot_mask') # [classes]

        logists = tf.where(mask == 1., cos_mt, cos_t) # [classes]
        logists = tf.multiply(logists, self.logist_scale, 'arcface_logist') # [classes]

        return logists # [classes]
