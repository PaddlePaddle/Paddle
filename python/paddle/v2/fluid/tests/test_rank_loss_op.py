import unittest
import numpy as np
from op_test import OpTest


class TestRankLossOp(OpTest):
    def setUp(self):
        self.op_type = "rank_loss"
        batch_size = 5
        # labels_{i} = {0, 1.0} or {0, 0.5, 1.0}
        label = np.random.randint(0, 2, size=(batch_size, 1)).astype("float32")
        left = np.random.random((batch_size, 1)).astype("float32")
        right = np.random.random((batch_size, 1)).astype("float32")
        loss = np.log(1.0 + np.exp(left - right)) - label * (left - right)
        self.inputs = {'Label': label, 'Left': left, 'Right': right}
        self.outputs = {'Out': loss}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["Left", "Right"], "Out")

    def test_check_grad_ignore_left(self):
        self.check_grad(["Right"], "Out", no_grad_set=set('Left'))

    def test_check_grad_ignore_right(self):
        self.check_grad(["Left"], "Out", no_grad_set=set('Right'))


if __name__ == '__main__':
    unittest.main()
