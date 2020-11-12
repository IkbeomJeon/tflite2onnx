import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf



# inputs
x = tf.keras.Input((80, 60, 32), dtype='float32', name='x')


# operator
transposeConv = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size = (2,2), strides = (2, 2), use_bias = False)(x)
transposeConv2 = tf.keras.layers.Conv2DTranspose(filters=10, kernel_size = (4,4), strides = (2, 2), use_bias = False)(x)
transposeConv3 = tf.keras.layers.Conv2DTranspose(filters=4, kernel_size = (5,5), strides = (4, 4), use_bias = True)(x)
transposeConv_relu = tf.keras.layers.Conv2DTranspose(filters=4, kernel_size = (5,5), strides = (4, 4), use_bias = True, activation = 'relu')(x)



# build Keras model
model_log = tf.keras.Model(x, tf.keras.backend.log(x))
model_conv = tf.keras.Model(x, transposeConv)
model_conv2 = tf.keras.Model(x, transposeConv2)
model_conv3 = tf.keras.Model(x, transposeConv3)
model_conv_relu = tf.keras.Model(x, transposeConv_relu)



export_model_list = {
    #'log.float32': model_log,
    'conv-transpose.float32' : model_conv,
    'conv-transpose2.float32' : model_conv2,
    'conv-transpose3.float32' : model_conv3,
    'conv-transpose_relu.float32' : model_conv_relu,

    }
 

# convert to TFLite model

for filename, model in export_model_list.items():
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    buf = converter.convert()

    targetdir = '../assets/tests'
    tflite_path = f'{targetdir}/{filename}.tflite'
    # save it
    with open(tflite_path, 'wb') as f:
        f.write(buf)
    

