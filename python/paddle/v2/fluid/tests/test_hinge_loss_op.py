import unittest
import numpy as np
from op_test import OpTest


class TestHingeLossOp(OpTest):
    def setUp(self):
        self.op_type = 'hinge_loss'
        samples_num = 64
        logits = np.random.uniform(-10, 10, (samples_num, 1)).astype('float32')
        labels = np.random.randint(0, 2, (samples_num, 1)).astype('float32')

        self.inputs = {
            'Logits': logits,
            'Labels': labels,
        }
        loss = np.maximum(1.0 - (2 * labels - 1) * logits, 0)
        self.outputs = {'Loss': loss}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['Logits'], 'Loss', max_relative_error=0.008)


if __name__ == '__main__':
    unittest.main()
