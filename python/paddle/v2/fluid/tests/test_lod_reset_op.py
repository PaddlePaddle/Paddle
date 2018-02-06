import unittest
import numpy as np
from op_test import OpTest


class TestLodResetOpByAttr(OpTest):
    def setUp(self):
        self.op_type = "lod_reset"
        x = np.random.random((10, 20)).astype("float32")
        lod = [[0, 3, 5, 10]]
        target_lod_0 = [0, 7, 10]
        self.inputs = {'X': (x, lod)}
        self.attrs = {'target_lod': target_lod_0}
        self.outputs = {'Out': (x, [target_lod_0])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestLodResetOpByInput(OpTest):
    def setUp(self):
        self.op_type = "lod_reset"
        x = np.random.random((10, 20)).astype("float32")
        lod = [[0, 3, 5, 10]]
        target_lod_0 = [0, 4, 7, 10]
        self.inputs = {
            'X': (x, lod),
            'TargetLoD': np.array([target_lod_0]).astype('int32')
        }
        self.outputs = {'Out': (x, [target_lod_0])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out", no_grad_set=set("TargetLoD"))


class TestLodResetOpBoth(OpTest):
    def setUp(self):
        self.op_type = "lod_reset"
        x = np.random.random((10, 20)).astype("float32")
        lod = [[0, 3, 5, 10]]
        target_lod_0_attr = [0, 7, 10]
        target_lod_0_in = [0, 4, 7, 10]
        self.inputs = {
            'X': (x, lod),
            'TargetLoD': np.array(target_lod_0_in).astype('int32')
        }
        self.attrs = {'target_lod': target_lod_0_attr}
        self.outputs = {'Out': (x, [target_lod_0_in])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out", no_grad_set=set("TargetLoD"))


if __name__ == '__main__':
    unittest.main()
