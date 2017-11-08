import unittest
import paddle.v2.framework.core as core
from paddle.v2.framework.op import Operator
import numpy


class TestGaussianRandomOp(unittest.TestCase):
    def test_cpu(self):
        self.gaussian_random_test(place=core.CPUPlace())

    def test_gpu(self):
        if core.is_compile_gpu():
            self.gaussian_random_test(place=core.GPUPlace(0))

    def gaussian_random_test(self, place):
        scope = core.Scope()
        scope.var('Out').get_tensor()

        op = Operator(
            "gaussian_random",
            Out='Out',
            shape=[1000, 784],
            mean=.0,
            std=1.,
            seed=10)

        context = core.DeviceContext.create(place)
        op.run(scope, context)
        tensor = numpy.array(scope.find_var('Out').get_tensor())
        self.assertAlmostEqual(numpy.mean(tensor), .0, delta=0.1)
        self.assertAlmostEqual(numpy.std(tensor), 1., delta=0.1)


if __name__ == "__main__":
    unittest.main()
