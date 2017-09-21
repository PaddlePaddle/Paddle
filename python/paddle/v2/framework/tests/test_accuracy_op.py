import unittest
import numpy as np
from op_test import OpTest


class TestAccuracyOp(OpTest):
    def setUp(self):
        self.op_type = "accuracy"
        n = 8192
        infer = np.random.randint(0, 2, (n, 1)).astype("int")
        label = np.random.randint(0, 2, (n, )).astype("int")
        self.inputs = {'Inference': infer, "Label": label}
        num_correct = 0
        for rowid in xrange(n):
            for ele in infer[rowid]:
                if ele == label[rowid]:
                    num_correct += 1
                    break
        self.outputs = {'Accuracy': [num_correct / float(n)]}

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
