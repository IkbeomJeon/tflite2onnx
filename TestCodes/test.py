import tflite2onnx as tf

#filename = 'log.float32'
filename = 'object_detection_3d_chair'

tflite_path = f'assets/tests/{filename}.tflite'
onnx_path = f'../{filename}.onnx'

tf.convert(tflite_path, onnx_path)