import paddle.v2 as paddle
from paddle.v2.framework import core
from paddle.v2.framework.op import Operator
from op_test import OpTest
import numpy as np
import unittest
import os

ABSOLUTE_PATH = "/tmp/PADDLE_MODEL"  # directory for save parameter files.

scope = core.Scope()
place = core.CPUPlace()
dev_ctx = core.DeviceContext.create(place)


class TestCheckpointOp(OpTest):
    def setUp(self):
        self.op_type = "checkpoint"
        x0 = np.random.random((2, 3)).astype("float32")
        x1 = np.random.random((1, 2)).astype("float32")
        x2 = np.random.random((2, 1)).astype("float32")

        self.attrs = {"absolute_path": ABSOLUTE_PATH, "interval": 100}

    def test_check_output(self):
        self.assertTrue(os.path.exists(ABSOLUTE_PATH))


#TODO(dzh): add a joint testing case to checkpoint/save

if __name__ == "__main__":
    unittest.main()
