import unittest
import numpy as np
from gradient_checker import GradientChecker, create_op
from op_test_util import OpTestMeta


class TestPadOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "pad"
        self.inputs = {'X': np.random.random((16, 16)).astype("float32"), }
        self.attrs['paddings'] = ((0, 1), (2, 3))
        self.attrs['pad_value'] = 0
        self.outputs = {
            'Out': np.pad(self.inputs['X'],
                          self.attrs['paddings'],
                          mode='constant',
                          constant_value=0)
        }


class PadGradOpTest(GradientChecker):
    def test_pad(self):
        op = Operator("pad", paddings=((0, 1), (2, 3)), pad_value=0)
        inputs = {'X': np.random.random((16, 16)).astype("float32"), }

        self.check_grad(op, inputs, set(["X"]), "Out", max_relative_error=0.5)


if __name__ == '__main__':
    unittest.main()
