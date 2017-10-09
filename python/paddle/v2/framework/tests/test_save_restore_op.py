import paddle.v2 as paddle
from paddle.v2.framework import core
from paddle.v2.framework.op import Operator
from op_test import OpTest
import numpy as np
import unittest
import os, sys

ABSOLUTE_PATH = "/tmp/PADDLE_TEST_MODEL"  # directory for save parameter files.

scope = core.Scope()
place = core.CPUPlace()
dev_ctx = core.DeviceContext.create(place)


# Saver is a demo for saveing arbitrary tensor
# it is a wrapper of c++ save/restore op
def Saver(var_list=None):
    net = core.Net.create()
    save_tensors = []
    for var in var_list:
        tensor = scope.find_var(var)
        save_tensors.append(tensor)
    save_op = Operator("save", X=save_tensors, absolutePath=ABSOLUTE_PATH)
    net.append_op(save_op)
    net.infer_shape(scope)
    net.run(scope, dev_ctx)


class TestSaver(unittest.TestCase):
    def test_save_tensors(self):
        a = scope.new_var("a")
        b = scope.new_var("b")
        Saver(["a", "b"])
        self.assertTrue(os.path.exists(ABSOLUTE_PATH))


class TestSaveOp(OpTest):
    def setUp(self):
        self.op_type = "save"
        x0 = np.random.random((2, 3)).astype("float32")
        x1 = np.random.random((1, 2)).astype("float32")
        x2 = np.random.random((2, 1)).astype("float32")

        self.inputs = {
            "X": [("x0", x0), ("x1", x1), ("x2", x2)],
            "absolute_path": ABSOLUTE_PATH
        }

    def test_check_output(self):
        if os.path.exists(ABSOLUTE_PATH):
            os.rmdir(ABSOLUTE_PATH)
        self.check_output()
        self.assertTrue(os.path.exists(ABSOLUTE_PATH))


# Must run savetest first
class TestRestoreOp(OpTest):
    def setUp(self):
        self.op_type = "restore"
        x0 = np.random.random((2, 3)).astype("float32")
        x1 = np.random.random((1, 2)).astype("float32")
        x2 = np.random.random((2, 1)).astype("float32")
        self.check_results = [x0, x1, x2]

        self.outputs = {"Out": [], "absolute_path": ABSOLUTE_PATH}

    def test_check_output(self):
        self.check_output()
        for e in self.check_results:
            self.assertTrue(e in self.outputs["Out"])
