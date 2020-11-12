# -*- coding: utf-8 -*-

import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
import seaborn as sns; #sns.set_theme()
import matplotlib.pyplot as plt
import time


#image_path = './samples/shose/1866.png'
image_path = 'e:/samples/shose/shose9.jpg'
#image_path = './samples/chair/chair5.jpg'


# Load the TFLite model and allocate tensors.
#model_path = 'models/object_detection_3d_chair.tflite'
tflite_path = 'e:/models/mp-shose.tflite'

interpreter = tflite.Interpreter(tflite_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
output_shape = output_details[0]['shape'] # heatmap
output_shape1 = output_details[1]['shape'] # offsetmap
output_shape2 = output_details[2]['shape'] # mask, coordinate map
print(input_shape)
print(output_shape)
print(output_shape1)
print(output_shape2)


input_image = Image.open(image_path)
input_image = input_image.resize((640,480))

input_data = np.array(input_image, dtype = np.float32)
#print(input_image.getpixel((1,2)))

#print(input_data.shape)
input_data = np.einsum('ijk->jik', input_data)
input_data = np.expand_dims(input_data,axis=0)
print(input_data.shape)

#input_data = input_data/127.0 - 1 # [-1,1]
input_data = input_data/255.0 #[0,1]
interpreter.set_tensor(input_details[0]['index'], input_data)

start = time.perf_counter()
start_time = time.time()
interpreter.invoke()
spent = time.perf_counter()-start
spent_time = time.time() - start_time
print("invoke() tick : " + str(spent*1000))
print("invoke() time : " + str(spent_time))

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])

output_data = np.squeeze(output_data, axis=0)
output_data = np.squeeze(output_data, axis=2)
output_data = np.einsum('ij->ji', output_data)

print(output_data)

plt.subplots(figsize = (8,6))
sns.heatmap(output_data)
plt.show()

#img = Image.fromarray(output_data, 'RGB')
#input_image.show()