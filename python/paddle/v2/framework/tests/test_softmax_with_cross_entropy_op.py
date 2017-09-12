import unittest
import numpy as np
import pdb

from op_test import OpTest
from test_softmax_op import stable_softmax


class TestSoftmaxWithCrossEntropyOp(OpTest):
    def setUp(self):
        self.op_type = "softmax_with_cross_entropy"

        MAX_BATCH_SIZE = 23
        MAX_CLASS_NUM = 255

        batch_size = np.random.randint(1, MAX_BATCH_SIZE, 1)[0]
        class_num = np.random.randint(2, MAX_CLASS_NUM, 1)[0]

        logits = np.random.uniform(0.1, 1.0,
                                   [batch_size, class_num]).astype("float32")
        softmax = np.apply_along_axis(stable_softmax, 1, logits)
        labels = np.random.randint(0, class_num, batch_size, dtype="int32")

        cross_entropy = [
            -np.log(softmax[i][labels[i]]) for i in range(softmax.shape[0])
        ]

        self.inputs = {"Logits": logits, "Label": labels}
        self.outputs = {"Loss": cross_entropy}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        pass


if __name__ == "__main__":
    unittest.main()
