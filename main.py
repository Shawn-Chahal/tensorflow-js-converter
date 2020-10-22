import os
import tensorflow as tf
import tensorflowjs as tfjs


model = tf.keras.models.load_model(os.path.join('tf-python', 'model_connect-four.h5'))
tfjs.converters.save_keras_model(model, os.path.join('tf-js', 'model_connect-four'))
