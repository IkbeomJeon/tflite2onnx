import logging
import tflite

from tflite2onnx.layout import Layout
from tflite2onnx.op.activation import handleFusedActivation
from tflite2onnx.op.common import Operator
from tflite2onnx.op.padding import PaddingMapping

logger = logging.getLogger('tflite2onnx')


class Transpose_Conv(Operator):
    TypeMapping = {
    
        tflite.BuiltinOperator.TRANSPOSE_CONV: 'Transpose_Conv',
    }

    def __init__(self, TFactory, index):
        super().__init__(TFactory, index)

        self.attrs['output_shape'] = []
        self.attrs['kernel_shape'] = []
        self.attrs['strides'] = []
        self.attrs['auto_pad'] = 'SAME_UPPER'  # See ComputePaddingHeightWidth() of TFLite
                       
        self.setInited()

    @property
    def type(self):
        return 'Transpose_Conv'

    @property
    def parse(self):
        logger.debug("Parsing %s...", self.type)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        
        assert(opcode is tflite.BuiltinOperator.TRANSPOSE_CONV)
        assert(op.InputsLength() == 3)
        assert(op.OutputsLength() == 1)
           
        #output shape
        olayout = Layout('NHWC', 'NCHW')
        output_shape = self.parseInput(0, olayout)
        
        # weight
        wlayout = Layout('CHWM', 'MCHW') 
        weight = self.parseInput(1, wlayout)
    
        # input
        #ilayout = Layout('NHWC', 'NCHW')
        #self.parseInput(2, ilayout)
                         
    
        # output
        output = self.parseOutput(0, olayout)
        #assert(output.shape == output_shape)
        
        # options
        op_opt = op.BuiltinOptions()
    
        option = tflite.TransposeConvOptions()
                               
        option.Init(op_opt.Bytes, op_opt.Pos)

        self.attrs['kernel_shape'] = weight.shape[1:3]        
        self.attrs['output_shape'] = output.shape[1:3] ##
        self.attrs['strides'] = [option.StrideH(), option.StrideW()]
        self.attrs['auto_pad'] = PaddingMapping[option.Padding()]

        #handleFusedActivation(self, option, output)

        self.setParsed()

    def propagatableTensors(self):
        return list()

    def transform(self):
        pass
