from tensorflow.python.client import device_lib
print('GPU: {}'.format(bool(device_lib.list_local_devices())))
