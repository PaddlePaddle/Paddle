import unittest, os
import numpy as np
import paddle.v2 as paddle
from paddle.v2.framework.op import Operator
import paddle.v2.framework.core as core
from op_test import OpTest, create_op, set_input

# gpu_list = os.environ["NV_LIST"]
gpu_list = "0,1,2,3"

if not core.is_compile_gpu() or not gpu_list:
    exit(0)

g_scope = core.Scope()
g_ctx = core.DeviceContext.create(core.CPUPlace())


class TestNCCLInit(OpTest):
    def setUp(self):
        self.op_type = "ncclInit"
        self.gpus = [int(g) for g in gpu_list.split(",")]

        self.attrs = {"gpus": self.gpus}
        self.scope = g_scope.var("Communicator")
        self.outputs = {"Communicator": self.scope.var("Communicator")}

    def test_check_output(self):
        self.check_output()


class TestNCCLAllReduce(unittest.TestCase):
    def setUp(self):
        # cpu allreduce for check
        def allreduce(tensors, gpus):
            num_device = len(gpus)
            assert (
                len(tensors) == num_device), "not match of tensor and device"
            Out = tensors
            for i in range(1, len(tensors)):
                Out[0] += Out[i]

            for i in range(1, len(tensors)):
                Out[i] = Out[0]

            return Out

        self.op_type = "ncclAllReduce"

        self.gpus = [int(g) for g in gpu_list.split(",")]

        self.g_scope = core.Scope()
        self.g_ctx = core.DeviceContext.create(core.CPUPlace())
        self.scopes = []
        self.ops = []
        self.places = []

        self.input_data = []

        for i in range(len(self.gpus)):
            self.input_data.append(np.random.random((32, 32)))
        self.output_data = allreduce(self.input_data, self.gpus)

        nccl_init = Operator("ncclInit", Out="Communicator", gpus=self.gpus)
        nccl_init.run(self.g_scope, self.g_ctx)

        for i in range(len(self.gpus)):
            # insert kid scope
            scope = self.g_scope.new_scope()
            place = core.GPUPlace(self.gpus[i])

            inputs = {
                "X": self.input_data[i],
                "Communicator": scope.find_var("Communicator")
            }
            outputs = {"Out": self.output_data[i]}
            # attrs = {"gpus": self.gpus}

            op = create_op(scope, self.op_type, inputs, outputs, attrs)
            set_input(scope, op, inputs, place)

            self.scopes.append(scope)
            self.ops.append(op)
            self.places.append(place)

    def test_output(self):
        idx = 0
        for scope, place, op in zip(self.scopes, self.places, self.ops):
            ctx = core.DeviceContext.create(place)
            op.run(scope, ctx)

        for out_name, out_dup in Operator.get_op_outputs(self.op.type()):
            actual = np.array(scope.find_var(out_name).get_tensor())
            expect = self.output_data[idx]

            idx += 1
            self.assertTrue(actual, expect), "has diff"


# if __name__ == "__main__":
#     unittest.main()
# usage : export NV_LIST=0,1,2,3 python *.py

# os.environ["NV_LIST"] = ["0,1,2,3"]

if __name__ == "__main__":
    unittest.main()
