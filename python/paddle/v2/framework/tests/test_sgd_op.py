import unittest
import numpy
from op_test_util import OpTestMeta


class TestSGD(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "sgd"
        w = numpy.random.random((102, 105)).astype("float32")
        g = numpy.random.random((102, 105)).astype("float32")
        lr = 0.1

        self.inputs = {'param': w, 'grad': g}
        self.attrs = {'learning_rate': lr}
        self.outputs = {'param_out': w - lr * g}


if __name__ == "__main__":
    unittest.main()
