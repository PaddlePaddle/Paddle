import unittest
import numpy
from paddle.v2.framework.op import Operator
from gradient_checker import GradientChecker
from gradient_checker import get_numeric_gradient


class GetNumericGradientTest(unittest.TestCase):
    def test_add_op(self):
        add_op = Operator('add', X="X", Y="Y", Out="Z")
        x = numpy.random.random((10, 1)).astype("float32")
        y = numpy.random.random((10, 1)).astype("float32")

        arr = get_numeric_gradient(add_op, {'X': x, "Y": y}, 'Z', 'X')
        self.assertAlmostEqual(arr.mean(), 1.0, delta=1e-4)

    def test_softmax_op(self):
        def stable_softmax(x):
            """Compute the softmax of vector x in a numerically stable way."""
            shiftx = x - numpy.max(x)
            exps = numpy.exp(shiftx)
            return exps / numpy.sum(exps)

        def label_softmax_grad(Y, dY):
            dX = Y * 0.0
            for i in range(Y.shape[0]):
                d = numpy.dot(Y[i, :], dY[i, :])
                dX[i, :] = Y[i, :] * (dY[i, :] - d)
            return dX

        softmax_op = Operator("softmax", X="X", Y="Y")

        X = numpy.random.random((2, 2)).astype("float32")
        Y = numpy.apply_along_axis(stable_softmax, 1, X)
        dY = numpy.ones(Y.shape)
        dX = label_softmax_grad(Y, dY)

        arr = get_numeric_gradient(softmax_op, {"X": X}, 'Y', 'X')
        numpy.testing.assert_almost_equal(arr, dX, decimal=1e-2)


if __name__ == '__main__':
    unittest.main()
