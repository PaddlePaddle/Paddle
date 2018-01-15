import unittest
import numpy as np
from op_test import OpTest
import paddle.v2.fluid.core as core


class TestFillOp(OpTest):
    def setUp(self):
        self.op_type = "fill"
        val = np.random.random(size=[100, 200])
        self.inputs = {}
        self.attrs = {
            'value': val.flatten().tolist(),
            'shape': [100, 200],
            'dtype': int(core.DataType.FP64)
        }
        self.outputs = {'Out': val.astype('float64')}

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
