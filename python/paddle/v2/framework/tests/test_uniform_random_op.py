import unittest
from paddle.v2.framework.op import Operator
import paddle.v2.framework.core as core
import numpy
import math


class TestUniformRandomOp(unittest.TestCase):
    def test_cpu(self):
        self._run_tests(place=core.CPUPlace())

    def test_gpu(self):
        if core.is_compile_gpu():
            self._run_tests(place=core.GPUPlace(0))

    def _run_tests(self, place):
        self.uniform_random_test(place)
        self.uniform_random_given_seed_test(place)
        self.uniform_random_random_seed_test(place)

    def _define_op(self, seed):
        scope = core.Scope()
        scope.var("Out").get_tensor()

        op = Operator(
            "uniform_random",
            Out="Out",
            shape=[1000, 784],
            min=-5.0,
            max=10.0,
            seed=seed)
        return scope, op

    def uniform_random_test(self, place):
        scope, op = self._define_op(10)

        ctx = core.DeviceContext.create(place)
        op.run(scope, ctx)
        tensor = numpy.array(scope.find_var("Out").get_tensor())
        self.assertAlmostEqual(tensor.mean(), 2.5, delta=0.1)
        except_std = math.sqrt((10.0 - (-5.0))**2 / 12.)
        self.assertAlmostEqual(tensor.std(), except_std, delta=0.1)

    def uniform_random_given_seed_test(self, place):
        scope, op = self._define_op(seed=10)

        context = core.DeviceContext.create(place)
        op.run(scope, context)
        tensor = numpy.array(scope.find_var("Out").get_tensor())
        tensor_expected = numpy.copy(tensor)

        op.run(scope, context)
        tensor = numpy.array(scope.find_var("Out").get_tensor())

        self.assertTrue(numpy.array_equal(tensor, tensor_expected))

    def uniform_random_random_seed_test(self, place):
        scope, op = self._define_op(seed=0)

        context = core.DeviceContext.create(place)
        op.run(scope, context)
        tensor = numpy.array(scope.find_var("Out").get_tensor())
        tensor1 = numpy.copy(tensor)

        op.run(scope, context)
        tensor2 = numpy.array(scope.find_var("Out").get_tensor())

        false_num = (tensor1 != tensor2).sum()
        self.assertGreater(false_num, int(0.95 * 1000 * 784))


if __name__ == "__main__":
    unittest.main()
