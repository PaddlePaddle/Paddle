import unittest, os
from threading import Thread
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
gpus = [int(g) for g in gpu_list.split(",")]


# ground truth
def allreduce(tensors, gpus):
    num_device = len(gpus)
    assert (len(tensors) == num_device), "not match of tensor and device"
    Out = tensors
    for i in range(1, len(tensors)):
        Out[0] += Out[i]

    for i in range(1, len(tensors)):
        Out[i] = Out[0]

    return Out


input_data = [
    np.random.random((32, 32)).astype("float32") for i in range(len(gpus))
]
output_data = allreduce(input_data, gpus)

# output_vars = [g_scope.var("Out_"+str(i)).get_tensor()
#                for i in range(len(gpus))]


def thread_allreduce_op(thread_id, gpu_id):
    i = gpu_id
    scope = g_scope.new_scope()
    place = core.GPUPlace(gpus[i])
    inputs = {
        "X": input_data[i],
        "Communicator": scope.find_var("Communicator")
    }
    outputs = {"Out": output_data[i]}

    op = create_op(scope, "ncclAllReduce", inputs, outputs, attrs={})
    place = core.GPUPlace(gpus[i])
    set_input(scope, op, inputs, place)

    ctx = core.DeviceContext.create(place)

    print "thread_id : ", thread_id, "gpu_id : ", gpu_id, " invoke allreduce"
    op.run(scope, ctx)
    print "thread_id : ", thread_id, "gpu_id : ", gpu_id, " allreduce Done."


class TestNCCLAllReduce(unittest.TestCase):
    def setUp(self):
        self.op_type = "ncclAllReduce"

        nccl_init = create_op(
            g_scope,
            op_type="ncclInit",
            inputs={},
            outputs={
                "Communicator": g_scope.var("Communicator").get_communicator()
            },
            attrs={"gpus": gpus})
        nccl_init.run(g_scope, g_ctx)

    def test_output(self):
        ops = []
        for i in range(len(gpus)):
            th = Thread(
                target=thread_allreduce_op, args=(
                    i,
                    gpus[i], ))
            th.start()
            ops.append(ops)
        for th in ops:
            th.join()

        idx = 0
        for out_name, out_dup in Operator.get_op_outputs(self.op.type()):
            actual = np.array(scope.find_var(out_name).get_tensor())
            expect = output_data[idx]

            idx += 1
            self.assertTrue(actual, expect), "has diff"


if __name__ == "__main__":
    unittest.main()
