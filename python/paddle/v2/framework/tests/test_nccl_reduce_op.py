import unittest, os
import numpy as np
import paddle.v2 as paddle
from paddle.v2.framework.op import Operator
import paddle.v2.framework.core as core
from op_test import OpTest, create_op, set_input

gpu_list = "0,1,2,3"
g_scope = core.Scope()
g_ctx = core.DeviceContext.create(core.CPUPlace())

if not core.is_compile_gpu() or not gpu_list:
    exit(0)


class TestNCCLReduce(OpTest):
    def setUp(self):
        self.op_type = "ncclReduce"
        self.gpus = [int(g) for g in gpu_list.split(",")]

        self.scope = g_scope.var("Communicator").get_communicator()
        self.outputs = {"Communicator": self.scope.var("Communicator")}

    def test_check_output(self):
        self.check_output()
