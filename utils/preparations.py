import cv2
import numpy as np

def channelSwap(data):
    rank = len(data.shape)
    data = data.copy()
    if rank == 3:
        _ = data[:,:,0].copy()
        data[:,:,0] = data[:,:,2].copy()
        data[:,:,2] = _
        return data
    elif rank == 4:
        _ = data[:,:,:,0].copy()
        data[:,:,:,0] = data[:,:,:,2].copy()
        data[:,:,:,2] = _
        return data
    elif rank == 5:
        _ = data[:,:,:,:,0].copy()
        data[:,:,:,:,0] = data[:,:,:,:,2].copy()
        data[:,:,:,:,2] = _
        return data

def normalize(data, in_range, out_range):
    compressed = (data - in_range[0]) / (in_range[1] - in_range[0])
    stretched = compressed * (out_range[1] - out_range[0]) + out_range[0]
    return stretched

def resize(image_tensor, size):
    rank = len(image_tensor.shape)
    if rank == 3:
        return cv2.resize(image_tensor, size)
    elif rank > 3:
        l = []
        for image_subtensor in image_tensor:
            l.append(resize(image_subtensor, size))
        return np.array(l)
    else:
        raise ValueError('Invalid tensor shape: {}'.format(image_tensor.shape))