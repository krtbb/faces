import numpy as np
import tensorflow as tf

def sum_square(x, y):
    return tf.reduce_sum((x-y)**2, axis=None)

def negative_log_likelihood(x_mu, x_sigma, y, batchsize=1, min_variance=1e-6, sep=False):
    x_sigma_ = x_sigma + min_variance
    x_sigma_recip = tf.exp(-np.log(x_sigma_))
    diff = x_mu - y
    if sep:
        return tf.reduce_sum( 0.5 * diff * diff * x_sigma_recip ) / batchsize, \
               tf.reduce_sum( 0.5 * np.log(x_sigma_) ) / batchsize
    else:
        return tf.reduce_sum( 0.5 * (diff*diff*x_sigma_recip + np.log(x_sigma_))) / batchsize

def pairwise_loss(x, y, equal=True, epsilon=1e+5, dist_method=sum_square):
    """
    *args*
    x --> type=np.ndarray, predicted vector
    y --> type=np.ndarray, predicted vector
    equal --> type=bool, means that x and y is the same person.
    epsilon --> type=float, a margin for not-equal samples.

    *return*
    type=np.float32, loss values
    """
    if equal:
        return dist_method(x, y)
    else:
        loss_ = epsilon - dist_method(x,y)
        return tf.cond(loss_>0, lambda: loss_, lambda: 0.)
        #return tf.reduce_max(0, epsilon - dist_method(x, y) )

def triplet_loss(x, p, n, epsilon=1e+5, dist_method=sum_square):
    """
    *args*
    x --> type=np.ndarray, predicted vector
    p --> type=np.ndarray, predicted vector from the same person with `x`
    n --> type=np.ndarray, predicted vector from a different person with `x`
    epsilon --> type=float, a margin.

    *return*
    type=np.float32, loss values
    """
    return np.max(0, epsilon + dist_method(x, p) - dist_method(x, ns))