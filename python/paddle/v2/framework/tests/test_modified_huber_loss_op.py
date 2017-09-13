import unittest
import numpy as np
from op_test import OpTest


def modified_huber_loss_forward(val):
    if val < -1:
        return -4 * val
    elif val < 1:
        return (1 - val) * (1 - val)
    else:
        return 0


class TestModifiedHuberLossOp(OpTest):
    def setUp(self):
        self.op_type = 'modified_huber_loss'
        samples_num = 32
        self.inputs = {
            'X': np.random.uniform(-1, 1., (samples_num, 1)).astype('float32'),
            'Y': np.random.choice([0, 1], samples_num).reshape((samples_num, 1))
        }
        product_res = self.inputs['X'] * (2 * self.inputs['Y'] - 1)
        loss = np.vectorize(modified_huber_loss_forward)(product_res)

        self.outputs = {
            'IntermediateVal': product_res,
            'Out': loss.reshape((samples_num, 1))
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', max_relative_error=0.005)


if __name__ == '__main__':
    unittest.main()
