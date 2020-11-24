import numpy as np

def sum_square(x, y):
    return np.sum((x-y)**2, axis=None)

def negative_log_likelihood(x_mu, x_sigma, y, batchsize=1, min_variance=1e-6, sep=False):
    x_sigma_ = x_sigma + min_variance
    x_sigma_recip = np.exp(-np.log(x_sigma_))
    diff = x_mu - y
    if sep:
        return np.sum( 0.5 * diff * diff * x_sigma_recip ) / batchsize, \
               np.sum( 0.5 * np.log(x_sigma_) ) / batchsize
    else:
        return np.sum( 0.5 * (diff*diff*x_sigma_recip + np.log(x_sigma_))) / batchsize