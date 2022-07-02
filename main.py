import os
import tensorflow as tf
import tensorflowjs as tfjs

# IMPORTANT: Only convert one model in a single run of the script.
name = "model_natural-or-artificial"

model = tf.keras.models.load_model(os.path.join("tf-python", f"{name}.h5"))
tfjs.converters.save_keras_model(model, os.path.join("tf-js", f"{name}"))
