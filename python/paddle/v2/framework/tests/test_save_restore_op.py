from paddle.v2.framework import core
from paddle.v2.framework.op import Operator
import numpy as np
from op_test import OpTest, create_op
import unittest
import os, sys

ABSOLUTE_PATH = "/tmp/MODEL"  # directory for save parameter files.

scope = core.Scope()
place = core.CPUPlace()
dev_ctx = core.DeviceContext.create(place)

# Saver is a demo for saving arbitrary tensor.
# TODO(dzh): Saver also need to save ProgramDesc. Should be done after
# python API done.


def Saver(var_list=None):
    net = core.Net.create()
    save_tensors = []
    save_op = Operator("save", X=save_tensors, absolutePath=ABSOLUTE_PATH)
    net.append_op(save_op)
    net.run(scope, dev_ctx)


class TestSaver(unittest.TestCase):
    def test_save_tensors(self):
        a = scope.var("a")
        b = scope.var("b")
        Saver(["a", "b"])
        self.assertTrue(os.path.exists(ABSOLUTE_PATH))


class TestSaveOp(OpTest):
    def setUp(self):
        self.op_type = "save"
        x0 = np.ones((1, 1)).astype("float32")
        x1 = np.ones((1, 1)).astype("float32")
        x2 = np.ones((2, 1)).astype("float32")

        self.inputs = {"X": [("x0", x0), ("x1", x1), ("x2", x2)], }

        self.attrs = {"absolutePath": ABSOLUTE_PATH}

    def test_check_output(self):
        if os.path.exists(ABSOLUTE_PATH):
            try:
                if os.path.isdir(ABSOLUTE_PATH):
                    os.rmdir(ABSOLUTE_PATH)
                elif os.path.isfile(ABSOLUTE_PATH):
                    os.remove(ABSOLUTE_PATH)
            except OSError:
                pass
        self.check_output()
        self.assertTrue(os.path.exists(ABSOLUTE_PATH))


# must run saveTest first
class TestRestoreOp(OpTest):
    def setUp(self):
        self.op_type = "restore"

        src_x0 = np.ones((2, 3)).astype("float32")
        src_x1 = np.ones((1, 2)).astype("float32")
        src_x2 = np.ones((2, 1)).astype("float32")

        x0 = scope.var('x0').get_tensor()
        x1 = scope.var('x1').get_tensor()
        x2 = scope.var('x2').get_tensor()

        self.src_tensors = [src_x0, src_x1, src_x2]
        self.inputs = {}
        self.outputs = {"Out": [("x0", x0), ("x1", x1), ("x2", x2)]}

        self.attrs = {"absolutePath": ABSOLUTE_PATH}

    def test_check_output(self):
        self.check_output()
        self.dst_tensors = [
            scope.find_var("x0"), scope.find_var("x1"), scope.find_var("x2")
        ]
        for src, dst in zip(self.save_tensors, self.dst_tensors):
            self.assertTrue(np.array_equal(src, dst))


if __name__ == "__main__":
    unittest.main()
