import unittest
import numpy as np
from gradient_checker import GradientChecker, create_op
from op_test_util import OpTestMeta


class TestAccuracyOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "accuracy"
        infer = np.random.randint(0, 2, (32, 1)).astype("int")
        label = np.random.randint(0, 2, (32, )).astype("int")
        self.inputs = {'Inference': infer, "Label": label}
        num_correct = 0
        for rowid in xrange(32):
            for ele in infer[rowid]:
                if ele == label[rowid]:
                    num_correct += 1
                    break
        self.outputs = {'Accuracy': [num_correct / 32.0]}


if __name__ == '__main__':
    unittest.main()
