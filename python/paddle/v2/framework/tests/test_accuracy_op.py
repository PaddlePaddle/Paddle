import unittest
import numpy as np
from op_test import OpTest


class TestAccuracyOp(OpTest):
    def setUp(self):
        self.op_type = "accuracy"
        n = 8192
        infer = np.random.random((n, 1)).astype("float32")
        indices = np.random.randint(0, 2, (n, 1))
        label = np.random.randint(0, 2, (n, 1))
        self.inputs = {'Out': infer, 'Indices': indices, "Label": label}
        num_correct = 0
        for rowid in xrange(n):
            for ele in indices[rowid]:
                if ele == label[rowid]:
                    num_correct += 1
                    break
        self.outputs = {
            'Accuracy': np.array([num_correct / float(n)]).astype("float32")
        }

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
