# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 21:15:40 2020

@author: IkTiger
"""
import tflite
import onnx
import numpy as np
from onnx import numpy_helper
import shrub
from tflite2onnx.layout import Layout

filename = 'conv-transpose.float32'

tflite_path = f'assets/tests/{filename}.tflite'
onnx_path = f'./{filename}.onnx'


model_onnx = onnx.load(onnx_path)
weights = model_onnx.graph.initializer

wlayout = Layout('CMHW', 'CHWM')

w = numpy_helper.to_array(weights[0])
#w2 = wlayout.transform(w)
w2 = np.reshape(w, (3,2,2,3))






if not np.allclose(w, w2, atol=1e-5, rtol=1e-5):
     print("Tensor %d mismatch!")
     


