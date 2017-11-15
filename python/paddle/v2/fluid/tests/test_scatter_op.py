import unittest
import numpy as np
from op_test import OpTest


class TestScatterOp(OpTest):
    def setUp(self):
        self.op_type = "scatter"
        ref_np = np.ones((3, 3)).astype("float32")
        index_np = np.array([1, 2]).astype("int32")
        updates_np = np.random.random((2, 3)).astype("float32")
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        self.inputs = {'Ref': ref_np, 'Index': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['Updates'], 'Out', in_place=True)


if __name__ == "__main__":
    unittest.main()
