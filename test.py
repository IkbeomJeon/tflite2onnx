import tflite2onnx

tflite_path = 'e:/models/transpose_conv.float32.tflite'

onnx_path = 'e:/models/transpose_conv.float32.onnx'

tflite2onnx.convert(tflite_path, onnx_path)