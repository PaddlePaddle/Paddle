import op_test
import unittest
import numpy as np
import paddle.v2.framework.core as core


class TestCastOp(op_test.OpTest):
    def setUp(self):
        ipt = np.random.random(size=[10, 10])
        self.inputs = {'X': ipt.astype('float32')}
        self.outputs = {'Out': ipt.astype('float64')}
        self.attrs = {
            'in_data_type': int(core.DataType.FP32),
            'out_data_type': int(core.DataType.FP64)
        }
        self.op_type = 'cast'

    def test_check_output(self):
        self.check_output()

    def test_grad(self):
        self.check_grad(['X'], ['Out'])


if __name__ == '__main__':
    unittest.main()
