import unittest

import paddle.v2.fluid as fluid
import paddle.v2.fluid.layers as layers
import paddle.v2.fluid.framework as framework
import paddle.v2.fluid.optimizer as optimizer
from paddle.v2.fluid.backward import calc_gradient


class TestCalcGradient(unittest.TestCase):
    def test_calc_gradient(self):
        x = layers.create_parameter(dtype="float32", shape=[5, 10])
        y = layers.create_parameter(dtype="float32", shape=[10, 8])
        mul_out = layers.mul(x=x, y=y)
        mean_out = layers.mean(x=mul_out)
        a = calc_gradient(mean_out, mul_out)
        b = calc_gradient(mean_out, x)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        exe.run(fluid.default_main_program(), feed={}, fetch_list=[a, b])


if __name__ == "__main__":
    unittest.main()
