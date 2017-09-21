import unittest
import numpy as np
from op_test import OpTest


class TestFillZerosLikeOp(OpTest):
    def setUp(self):
        self.op_type = "fill_zeros_like"
        self.inputs = {'Src': np.random.random((219, 232)).astype("float32")}
        self.outputs = {'Dst': np.zeros_like(self.inputs["Src"])}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
