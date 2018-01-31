import unittest
import numpy as np
import random
from op_test import OpTest, randomize_probability


class TestNoisyCrossEntropyOp(OpTest):
    """Test noisy cross-entropy with discrete one-hot labels.
    """

    def setUp(self):
        self.op_type = "noisy_cross_entropy"
        batch_size = 10
        class_num = 25
        noise = 0.0
        X = randomize_probability(batch_size, class_num, dtype='float64')
        label = np.random.randint(0, class_num, (batch_size, 1), dtype="int64")
        noisy_mat = np.full((batch_size, class_num), noise / (50.0 - 1.0))
        for i in range(batch_size):
            noisy_mat[i][label[i][0]] = 1.0 - noise
        noisy_cross_entropy = np.asmatrix(-np.sum(np.log(X) * noisy_mat, axis=1), dtype="float64").transpose()
        self.inputs = {"X": X, "Label": label}
        self.outputs = {"Y": noisy_cross_entropy}
        self.attrs = {"noise": noise}

    def test_check_output(self):
        self.check_output()
        print("check output done")
        
    def test_check_grad(self):
        self.check_grad(["X"], "Y", numeric_grad_delta=0.001)
        print("check gradient done.")


if __name__ == "__main__":
    unittest.main()
