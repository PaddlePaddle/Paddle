import unittest
import numpy as np
from op_test import OpTest


def modified_huber_loss_forward(val):
    if val < -1:
        return -4. * val
    elif val < 1:
        return (1. - val) * (1. - val)
    else:
        return 0.


class TestModifiedHuberLossOp(OpTest):
    def setUp(self):
        self.op_type = 'modified_huber_loss'
        samples_num = 32

        x_np = np.random.uniform(-2., 2., (samples_num, 1)).astype('float32')
        y_np = np.random.choice([0, 1], samples_num).reshape(
            (samples_num, 1)).astype('float32')
        product_res = x_np * (2. * y_np - 1.)
        # keep away from the junction of piecewise function
        for pos, val in np.ndenumerate(product_res):
            while abs(val - 1.) < 0.05:
                x_np[pos] = np.random.uniform(-2., 2.)
                y_np[pos] = np.random.choice([0, 1])
                product_res[pos] = x_np[pos] * (2 * y_np[pos] - 1)
                val = product_res[pos]

        self.inputs = {'X': x_np, 'Y': y_np}
        loss = np.vectorize(modified_huber_loss_forward)(product_res)

        self.outputs = {
            'IntermediateVal': product_res,
            'Out': loss.reshape((samples_num, 1))
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', max_relative_error=0.01)


if __name__ == '__main__':
    unittest.main()
