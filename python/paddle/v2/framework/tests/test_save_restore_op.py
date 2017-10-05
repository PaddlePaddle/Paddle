import paddle.v2 as paddle
from paddle.v2.framework import core
from paddle.v2.framework.op import Operator
from op_test import OpTest
import numpy as np
import unittest
import os, sys

ABSOLUTE_PATH = "/tmp/"  # directory for save parameter files.


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
        self.check_output()
        self.assertTrue(
            os.path.exists(os.path.join(ABSOLUTE_PATH, "x0")),
            " parameter not saved")


class TestRestoreOp(OpTest):
    def setUp(self):
        self.op_type = "restore"
        x0 = np.random.random((2, 3)).astype("float32")
        x1 = np.random.random((1, 2)).astype("float32")
        x2 = np.random.random((2, 1)).astype("float32")

        self.inputs = {
            "X": [("x0", x0), ("x1", x1), ("x2", x2)],
            "absolute_path": ABSOLUTE_PATH
        }

    # def test_check_output(self):
    #   self.check_output()
    #   self.assertTrue(os.path.exists(os.path.join(ABSOLUTE_PATH, "x0")),
    #                   " parameter not saved")
