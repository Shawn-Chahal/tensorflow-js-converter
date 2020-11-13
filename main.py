import os
import tensorflow as tf
import tensorflowjs as tfjs


# IMPORTANT: Only convert one model in a single run of the script.
name = "connect-four"
epoch = "0620"

model = tf.keras.models.load_model(
    os.path.join("tf-python", f"model_{ name }_{epoch}.h5")
)
tfjs.converters.save_keras_model(model, os.path.join("tf-js", f"model_{ name }"))
