import unittest
import numpy

from paddle.v2.fluid.op import Operator
import paddle.v2.fluid.core as core
import paddle.v2.fluid as fluid


class TestUniformRandomOp(unittest.TestCase):
    def setUp(self):
        self.op_type = "uniform_random"
        self.inputs = {}
        self.attrs = {
            "shape": [1000, 784],
            "min": -5.0,
            "max": 10.0,
            "seed": 10
        }
        self.outputs = ["Out"]

    def test_cpu(self):
        self.uniform_random_test(place=core.CPUPlace())

    def test_gpu(self):
        if core.is_compile_gpu():
            self.uniform_random_test(place=core.CUDAPlace(0))

    def uniform_random_test(self, place):
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

        exe = fluid.Executor(place)
        outs = exe.run(program, fetch_list=fetch_list)

        tensor = outs[0]

        self.assertAlmostEqual(tensor.mean(), 2.5, delta=0.1)


if __name__ == "__main__":
    unittest.main()
