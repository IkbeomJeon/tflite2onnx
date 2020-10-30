import onnx
from onnx_tf.backend import prepare


onnx_path = 'e:/models/mp-shose.onnx'
tf_path = 'e:/models/mp-shose'

onnx_model = onnx.load(onnx_path)  # load onnx model
tf_rep = prepare(onnx_model)  # prepare tf representation
tf_rep.export_graph(tf_path)  # export the model