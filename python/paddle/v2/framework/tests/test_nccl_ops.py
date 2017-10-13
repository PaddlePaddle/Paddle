import unittest, os
import numpy as np
import paddle.v2 as paddle
from paddle.v2.framework.op import Operator
import paddle.v2.framework.core as core
from op_test import OpTest, create_op, set_input

gpu_list = os.environ["NV_LIST"]

if not core.is_compile_gpu() or not gpu_list:
    exit(0)


def allreduce(tensors, num_device):
    assert (len(tensors) == num_device), "not match of tensor and device"
    Out = tensors
    for i in range(1, len(tensors)):
        Out[0] += Out[i]

    for i in range(1, len(tensors)):
        Out[i] = Out[0]

    return Out


class TestNCCLAllReduce(unittest.TestCase):
    def __init__(self):
        self.op_type = "nnclAllReduce"

        self.gpus = [int(g) for g in gpu_list]

        self.scopes = []
        self.ops = []
        self.places = []

        self.input_data = []
        for i in range(len(self.gpus)):
            input_data.append(np.random.random((32, 32)))
        self.output_data = allreduce(input_data)

        for i in range(len(self.gpus)):
            scope = core.Scope()
            place = core.GPUPlace(self.gpus[i])
            inputs = {"X": self.input_data[i]}
            outputs = {"Out": self.output_data[i]}
            attrs = {"gpus": self.gpus}

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


if __name__ == "__main__":
    # usage : export NV_LIST=0,1,2,3 python *.py

    os.environ["NV_LIST"] = ["0,1,2,3"]
    unittest.main()
