import tflite2onnx

tflite_path = 'e:/models/object_detection_3d_sneakers.tflite'

onnx_path = 'e:/models/object_detection_3d_sneakers.onnx'

tflite2onnx.convert(tflite_path, onnx_path)