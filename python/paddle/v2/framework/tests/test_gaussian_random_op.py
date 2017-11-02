import unittest
import paddle.v2.framework.core as core
from paddle.v2.framework.op import Operator
import numpy


class TestGaussianRandomOp(unittest.TestCase):
    def test_cpu(self):
        self._run_tests(place=core.CPUPlace())

    def test_gpu(self):
        if core.is_compile_gpu():
            self._run_tests(place=core.GPUPlace(0))

    def _run_tests(self, place):
        self.gaussian_random_test(place)
        self.gaussian_random_given_seed_test(place)
        self.gaussian_random_random_seed_test(place)

    def _define_op(self, seed):
        """using non-default values for mean and std"""
        scope = core.Scope()
        scope.var("Out").get_tensor()

        op = Operator(
            "gaussian_random",
            Out="Out",
            shape=[1000, 784],
            mean=0.5,  # using non-default value
            std=6,  # using non-default value
            seed=seed)
        return scope, op

    def gaussian_random_test(self, place):
        scope, op = self._define_op(seed=10)

        context = core.DeviceContext.create(place)
        op.run(scope, context)
        tensor = numpy.array(scope.find_var("Out").get_tensor())
        self.assertAlmostEqual(tensor.mean(), 0.5, delta=0.1)
        self.assertAlmostEqual(tensor.std(), 6, delta=0.1)

    def gaussian_random_given_seed_test(self, place):
        scope, op = self._define_op(seed=10)

        context = core.DeviceContext.create(place)
        op.run(scope, context)
        tensor = numpy.array(scope.find_var("Out").get_tensor())
        tensor_expected = numpy.copy(tensor)

        op.run(scope, context)
        tensor = numpy.array(scope.find_var("Out").get_tensor())

        self.assertTrue(numpy.array_equal(tensor, tensor_expected))

    def gaussian_random_random_seed_test(self, place):
        scope, op = self._define_op(seed=0)

        context = core.DeviceContext.create(place)
        op.run(scope, context)
        tensor = numpy.array(scope.find_var("Out").get_tensor())
        tensor1 = numpy.copy(tensor)

        op.run(scope, context)
        tensor2 = numpy.array(scope.find_var("Out").get_tensor())

        # Although tensor1 and tensor2 are sampled randomly, some values may
        # still be identical by chance. Therefore, we choose a not-so-big,
        # not-so-small ratio 0.95 here. 
        false_num = (tensor1 != tensor2).sum()
        self.assertGreater(false_num, int(0.95 * 1000 * 784))


if __name__ == "__main__":
    unittest.main()
