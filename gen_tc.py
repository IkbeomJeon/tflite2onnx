import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np
import tflite

# inputs
x = tf.keras.Input((20, 30, 3), dtype='float32', name='a')

# operator
ctlayer = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size = (2,2), strides = (2, 2))


# build Keras model
model = tf.keras.Model(x, ctlayer(x))
model.summary()

# convert to TFLite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
buf = converter.convert()

filename_save = 'transpose_conv.float32'
# save it
with open(filename_save+'.tflite', 'wb') as f:
    f.write(buf)
    

