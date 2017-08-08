import unittest
import paddle.v2.framework.core as core
import paddle.v2.framework.op as Operator
import numpy


class GaussianRandomTest(unittest.TestCase):
    def test_cpu(self):
        self.test_gaussian_random(place=core.CPUPlace())

    def test_gpu(self):
        self.test_gaussian_random(place=core.GPUPlace(0))

    def test_gaussian_random(self, place):
        scope = core.Scope()
        scope.new_var("Out").get_tensor()
        op = Operator(
            "gaussian_random",
            Out="Out",
            dims=[1000, 784],
            mean=.0,
            std=1.,
            seed=0)
        op.infer_shape(scope)
        context = core.DeviceContext.create(place)
        op.run(scope, context)
        tensor = numpy.array(scope.find_var("Out").get_tensor())
        self.assertAlmostEqual(numpy.mean(tensor), .0, places=3)
        self.assertAlmostEqual(numpy.std(tensor), 1., places=3)


if __name__ == '__main__':
    unittest.main()
