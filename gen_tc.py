import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf


# inputs
x = tf.keras.Input((20, 30, 128), dtype='float32', name='x')


# operator
transposeConv = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size = (2,2), strides = (2, 2))


# build Keras model
model_log = tf.keras.Model(x, tf.keras.backend.log(x))
model_conv = tf.keras.Model(x, transposeConv(x))


export_model_list = {
    'log.float32': model_log, 
    'transpose_conv.float32' : model_conv
    }
 

# convert to TFLite model

for filename, model in export_model_list.items():
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    buf = converter.convert()

    # save it
    with open(filename, 'wb') as f:
        f.write(buf)
    

