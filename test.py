import tflite2onnx

#filename = 'log.float32'
filename = 'object_detection_3d_sneakers'


tflite_path = f'assets/tests/{filename}.tflite'
onnx_path = f'./{filename}.onnx'


tflite2onnx.convert(tflite_path, onnx_path)