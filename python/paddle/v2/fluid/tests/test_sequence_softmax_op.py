import unittest
import numpy as np
from op_test import OpTest
from test_softmax_op import stable_softmax


class TestSequenceSoftmaxOp(OpTest):
    def setUp(self):
        self.op_type = "sequence_softmax"
        x = np.random.uniform(0.1, 1, (11, 1)).astype("float32")
        lod = [[0, 4, 5, 8, 11]]

        out = np.zeros((11, 1)).astype("float32")
        for i in range(4):
            sub_x = x[lod[0][i]:lod[0][i + 1], :]
            sub_x = sub_x.reshape(1, lod[0][i + 1] - lod[0][i])
            sub_out = stable_softmax(sub_x)
            out[lod[0][i]:lod[0][i + 1], :] = sub_out.reshape(
                lod[0][i + 1] - lod[0][i], 1)

        self.inputs = {"X": (x, lod)}
        self.outputs = {"Out": out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out", max_relative_error=0.01)


if __name__ == "__main__":
    unittest.main()
