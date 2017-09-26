import unittest
import numpy as np
import paddle.v2.framework.core as core
from op_test import TestUtils


class GetNumericGradientTest(unittest.TestCase):
    def test_add_op(self):
        x = np.random.random((10, 1)).astype("float32")
        y = np.random.random((10, 1)).astype("float32")
        z = x + y
        scope = core.Scope()
        add_op = TestUtils.create_op(scope, "add", {'X': x,
                                                    'Y': y}, {'Out': z}, dict())
        arr = TestUtils.get_numeric_gradient(scope, add_op, {'X': x,
                                                             'Y': y}, 'X',
                                             ['Out'])
        self.assertAlmostEqual(arr.mean(), 1.0, delta=1e-4)

    def test_add_op_strict(self):
        x = np.random.random((10, 1)).astype("float32")
        y = np.random.random((10, 1)).astype("float32")
        z = x + y
        scope = core.Scope()
        add_op = TestUtils.create_op(scope, "add", {'X': x,
                                                    'Y': y}, {'Out': z}, dict())
        arr = TestUtils.get_numeric_gradient(
            scope, add_op, {'X': x,
                            'Y': y}, 'X', ['Out'], strict=True)
        self.assertAlmostEqual(
            arr.all(), np.identity(
                10, dtype=np.float64).all(), delta=1e-6)

    def test_softmax_op(self):
        def stable_softmax(x):
            """Compute the softmax of vector x in a numerically stable way."""
            shiftx = x - np.max(x)
            exps = np.exp(shiftx)
            return exps / np.sum(exps)

        def label_softmax_grad(Y, dY):
            dX = Y * 0.0
            for i in range(Y.shape[0]):
                d = np.dot(Y[i, :], dY[i, :])
                dX[i, :] = Y[i, :] * (dY[i, :] - d)
            return dX

        X = np.random.random((2, 2)).astype("float32")
        Y = np.apply_along_axis(stable_softmax, 1, X)
        dY = np.ones(Y.shape)
        dX = label_softmax_grad(Y, dY)

        scope = core.Scope()
        softmax_op = TestUtils.create_op(scope, "softmax", {"X": X}, {"Y": Y},
                                         dict())

        arr = TestUtils.get_numeric_gradient(scope, softmax_op, {"X": X}, "X",
                                             "Y")
        np.testing.assert_almost_equal(arr, dX, decimal=1e-2)

    def test_softmax_op_strict(self):
        def stable_softmax(x):
            """Compute the softmax of vector x in a numerically stable way."""
            shiftx = x - np.max(x)
            exps = np.exp(shiftx)
            return exps / np.sum(exps)

        def label_softmax_grad(Y, dY):
            dX = Y * 0.0
            for i in range(Y.shape[0]):
                d = np.dot(Y[i, :], dY[i, :])
                dX[i, :] = Y[i, :] * (dY[i, :] - d)
            return dX

        X = np.random.random((2, 2)).astype("float32")
        Y = np.apply_along_axis(stable_softmax, 1, X)
        dX = np.zeros((X.size, 0)).astype("float32")
        dY = np.zeros(Y.shape)
        for i in xrange(X.size):
            dY.ravel()[i] = 1
            dX = np.concatenate(
                (dX, label_softmax_grad(Y, dY).reshape(X.size, 1)), axis=1)
            dY.ravel()[i] = 0

        scope = core.Scope()
        softmax_op = TestUtils.create_op(scope, "softmax", {"X": X}, {"Y": Y},
                                         dict())

        arr = TestUtils.get_numeric_gradient(
            scope, softmax_op, {"X": X}, "X", "Y", strict=True)
        np.testing.assert_almost_equal(arr, dX, decimal=1e-6)


if __name__ == "__main__":
    unittest.main()
