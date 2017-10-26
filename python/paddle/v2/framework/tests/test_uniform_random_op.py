import unittest
from paddle.v2.framework.op import Operator
import paddle.v2.framework.core as core
import numpy


class TestUniformRandomOp(unittest.TestCase):
    def test_uniform_random_cpu(self):
        self.uniform_random_test(place=core.CPUPlace())

    def test_uniform_random_gpu(self):
        if core.is_compile_gpu():
            self.uniform_random_test(place=core.GPUPlace(0))

    def uniform_random_test(self, place):
        scope = core.Scope()
        scope.var('X').get_tensor()

        op = Operator(
            "uniform_random",
            Out='X',
            shape=[1000, 784],
            min=-5.0,
            max=10.0,
            seed=10)

        ctx = core.DeviceContext.create(place)
        op.run(scope, ctx)
        tensor = numpy.array(scope.find_var('X').get_tensor())
        self.assertAlmostEqual(tensor.mean(), 2.5, delta=0.1)


if __name__ == "__main__":
    unittest.main()
