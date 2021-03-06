import os
import logging

import shrub
import tflite2onnx as t2o

shrub.util.formatLogging(logging.DEBUG)


def end2end_test(model_name, use_layout):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    tflm_dir = os.path.abspath(cur_dir + '/../assets/tests')
    tflm_name = model_name + '.tflite'
    onnx_name = model_name + '.onnx'
    tflm_path = os.path.join(tflm_dir, tflm_name)
    t2o.convert(tflm_path, onnx_name)

    m = shrub.tflite.parse(tflm_path)
    m.genInput()

    onnx_ret = shrub.onnx.run(onnx_name, m.inputs, use_layout)
    tflite_ret = shrub.tflite.run(tflm_path, m.inputs)
    assert(shrub.network.cmpTensors(onnx_ret, tflite_ret, useLayout=use_layout))


def test_ops_implicit_layout():
    # this ops will stop layout propagation
    OP_LIST = (
        'avgpooling.float32',
        'avgpool-concat.float32',
        'conv.float32',
        'conv-dilation.float32',
        'conv-relu.float32',
        'conv-relu6.float32',
        'conv-stride.float32',
        'depthwise-conv.float32',
        'depthwise-conv-stride.float32',
        'fullyconnected.float32',
        'fullyconnected-relu6.float32',
        'maxpooling.float32',
    )

    for op in OP_LIST:
        end2end_test(op, 'NCHW')


def test_ops_post_propagation():
    # this ops need post-propagation handling
    OP_LIST = (
        'concat.float32',
        'mean.float32',
        'padding.float32',
        'reshape.float32',
        'softmax.float32',
        'split.float32',
        'stridedslice-beginmask.float32',
        'stridedslice-endmask.float32',
        'stridedslice-stride.float32',
        'stridedslice.float32',
        'transpose.float32',
    )

    for op in OP_LIST:
        end2end_test(op, 'NHWC')


def test_ops_layout_transparent():
    # this ops are very wild :)
    OP_LIST = (
        'abs.float32',
        'add.float32',
        'add-relu.float32',
        'mul.float32',
        'relu6.float32',
        'relu.float32',
    )

    for op in OP_LIST:
        end2end_test(op, 'NHWC')


def test_networks():
    NETWORK_LIST = (
        'mobilenet_v1_0.25_128',
    )

    for net in NETWORK_LIST:
        end2end_test(net, 'NCHW')


if __name__ == '__main__':
    test_ops_implicit_layout()
    test_ops_post_propagation()
    test_ops_layout_transparent()
    test_networks()
