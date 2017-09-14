import unittest
import numpy as np
from op_test import OpTest


class TestReshapeOp(OpTest):
    def setUp(self):
        self.op_type = "rank_loss"
        num = 5
        # P = {0, 1.0} or {0, 0.5, 1.0}
        P = np.random.randint(0, 2, size=(num, num)).astype("float32")
        Oi = np.random.random((num, num)).astype("float32")
        Oj = np.random.random((num, num)).astype("float32")
        O = Oi - Oj
        Out = np.log(1.0 + np.exp(O)) - P * O
        self.inputs = {'P': P, 'Oi': Oi, 'Oj': Oj}
        self.outputs = {'Out': Out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["Oj"], "Out")


if __name__ == '__main__':
    unittest.main()
