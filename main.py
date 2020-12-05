import os
import tensorflow as tf
import tensorflowjs as tfjs


# IMPORTANT: Only convert one model in a single run of the script.
name = "gen_model_059"
modify = 0

model = tf.keras.models.load_model(os.path.join("tf-python", f"{ name }.h5"))

if modify == 1:
    model.summary()
    outputs = model.get_layer("add_3").output
    outputs = tf.keras.layers.Activation('tanh')(outputs)
    model = tf.keras.Model(inputs=model.inputs, outputs=outputs)

model.summary()
tfjs.converters.save_keras_model(model, os.path.join("tf-js", f"{ name }"))
