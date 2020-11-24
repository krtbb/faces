import tensorflow as tf

def load_image(path):
    image = tf.io.read_file(path)
    return image

def preprocess_image(image, size=[192, 192]):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, size)
    image /= 255.0

    return image

def load_and_preprocess_image(path):
    image = load_image(path)
    image_preprocessed = preprocess_image(image)
    return image_preprocessed