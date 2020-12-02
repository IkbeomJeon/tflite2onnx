import logging
import tflite

from tflite2onnx.op.common import Operator

logger = logging.getLogger('tflite2onnx')


#
class Logistic(Operator):
    TypeMapping = {
        tflite.BuiltinOperator.LOGISTIC: 'Sigmoid',
    }

    def __init__(self, TFactory, index):
        super().__init__(TFactory, index)
        self.setInited()

    @property
    def type(self):
        return 'Sigmoid'

    def parse(self):
        logger.debug("Parsing %s...", self.type)

        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert (opcode in self.TypeMapping)

        assert (op.InputsLength() == 1)
        assert (op.OutputsLength() == 1)
        self.parseInput(0)
        self.parseOutput(0)

        self.setParsed()

    def propagatableTensors(self):
        return self.inputs + self.outputs

    def transform(self):
        pass