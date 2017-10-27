import unittest, os
import numpy as np
import paddle.v2 as paddle
from paddle.v2.framework.op import Operator
import paddle.v2.framework.core as core
from op_test import OpTest, create_op, set_input

if not core.is_compile_gpu():
    exit(0)

gpu_count = core.get_cuda_device_count()

if gpu_count <= 1:
    exit(0)

g_scope = core.Scope()
g_ctx = core.DeviceContext.create(core.CPUPlace())


class TestNCCLInit(unittest.TestCase):
    def test_init(self):
        self.op_type = "ncclInit"
        self.gpus = range(gpu_count)

        self.inputs = {}
        self.attrs = {"gpus": self.gpus}
        g_scope.var("Communicator").get_communicator()
        self.outputs = {"Communicator": g_scope.find_var("Communicator")}
        nccl_init = create_op(
            g_scope,
            op_type=self.op_type,
            inputs=self.inputs,
            outputs=self.outputs,
            attrs=self.attrs)
        nccl_init.run(g_scope, g_ctx)


if __name__ == "__main__":
    unittest.main()
