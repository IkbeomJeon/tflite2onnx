##
import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns; #sns.set_theme()
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

## load model
image_path = 'e:/samples/shose/shose9.jpg'
model_path = 'e:/models/mp-shose'

loaded = tf.saved_model.load(model_path)

## show model outpts
print(list(loaded.signatures.keys()))  # ["serving_default"]
infer = loaded.signatures["serving_default"]
print(infer.structured_outputs)


## make input tensor

img = tf.keras.preprocessing.image.load_img(image_path, target_size=[640, 480])
# plt.subplots(figsize = (640,480))
# plt.imshow(img);
# plt.show()
input_data = tf.keras.preprocessing.image.img_to_array(img)
##
input_image = Image.open(image_path)
input_image = input_image.resize((640,480))
input_data = np.array(input_image, dtype = np.float32)
input_data = np.einsum('ijk->kji', input_data)
input_data = input_data[tf.newaxis,...]
input_data = input_data/255.0 #[0,1]
x = tf.convert_to_tensor(input_data, dtype = tf.float32)
print(x.shape)

## inference
heatmap = infer(x)['output_0']
offsetmap = infer(x)['output_0']
print(heatmap.shape)
##
output_data = np.squeeze(heatmap, axis=0)
output_data2 = np.squeeze(output_data, axis=0)
output_data2 = np.einsum('ij->ji', output_data2)
print(output_data2.shape)
##
#output_data = np.einsum('ij->ji', output_data)

plt.subplots(figsize = (8,6))
sns.heatmap(output_data2)
plt.show()

