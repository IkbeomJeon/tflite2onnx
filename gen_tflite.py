import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

defaultPath = '.\\assets\\tests\\'
# inputs
iN = 1
iW = 1
iH = 2
iC = 3
iM = iC

x = tf.keras.Input((iH, iW, iC), dtype='float32', name='x')


# operator
tconv = tf.keras.layers.Conv2DTranspose(
    filters=iM, kernel_size = (2,2), strides = (2, 2), use_bias = False)
tconv_Relu = tf.keras.layers.Conv2DTranspose(
    filters=iM, kernel_size = (2,2), strides = (2, 2), use_bias = False, activation ='relu')


# build Keras model
model_log = tf.keras.Model(x, tf.keras.backend.log(x))
model_tconv = tf.keras.Model(x, tconv(x))
model_tconv_relu = tf.keras.Model(x, tconv_Relu(x))


export_model_list = {
    #'log.float32': model_log, 
    'conv-transpose.float32' : model_tconv,
    #'conv-transpose_relu.float32' : model_tconv_relu,
    }
 

# convert to TFLite model

for filename, model in export_model_list.items():
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    buf = converter.convert()

    # save it
    with open(defaultPath+filename+'.tflite', 'wb') as f:
        f.write(buf)
    

