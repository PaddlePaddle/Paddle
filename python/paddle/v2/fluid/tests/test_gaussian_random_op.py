import unittest
import numpy

import paddle.v2.fluid as fluid
import paddle.v2.fluid.core as core
from paddle.v2.fluid.op import Operator
from paddle.v2.fluid.executor import Executor


class TestGaussianRandomOp(unittest.TestCase):
    def setUp(self):
        self.op_type = "gaussian_random"
        self.inputs = {}
        self.attrs = {"shape": [1000, 784], "mean": .0, "std": 1., "seed": 10}

        self.outputs = ["Out"]

    def test_cpu(self):
        self.gaussian_random_test(place=fluid.CPUPlace())

    def test_gpu(self):
        if core.is_compile_gpu():
            self.gaussian_random_test(place=fluid.CUDAPlace(0))

    def gaussian_random_test(self, place):

        program = fluid.Program()
        block = program.global_block()
        vout = block.create_var(name="Out")
        op = block.append_op(
            type=self.op_type, outputs={"Out": vout}, attrs=self.attrs)

        op.desc.infer_var_type(block.desc)
        op.desc.infer_shape(block.desc)

        fetch_list = []
        for var_name in self.outputs:
            fetch_list.append(block.var(var_name))

        exe = Executor(place)
        outs = exe.run(program, fetch_list=fetch_list)
        tensor = outs[0]

        self.assertAlmostEqual(numpy.mean(tensor), .0, delta=0.1)
        self.assertAlmostEqual(numpy.std(tensor), 1., delta=0.1)


if __name__ == "__main__":
    unittest.main()
