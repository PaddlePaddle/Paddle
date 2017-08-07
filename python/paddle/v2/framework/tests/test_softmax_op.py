import unittest

import numpy as np
import paddle.v2.framework.core as core
from paddle.v2.framework.op import Operator

from op_test_util import OpTestMeta
from gradient_checker import GradientChecker, create_op


def stable_softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


class TestSoftmaxOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "softmax"
        self.inputs = {'X': np.random.random((32, 100)).astype("float32")}
        self.outputs = {
            'Y': np.apply_along_axis(stable_softmax, 1, self.inputs['X'])
        }


# Reference implementation of cross entropy with soft labels
def label_softmax_grad(Y, dY):
    dX = Y * 0.0
    for i in range(Y.shape[0]):
        d = np.dot(Y[i, :], dY[i, :])
        dX[i, :] = Y[i, :] * (dY[i, :] - d)
    return dX


class TestSoftmaxGradOp(unittest.TestCase):
    def test_softmax_grad(self):
        op = Operator('softmax', X="X", Y="Y")
        backward_op = core.Operator.backward(op, set())
        self.assertEqual(backward_op.type(), "softmax_grad")
        expected = '''Op(softmax_grad), inputs:(X, Y, Y@GRAD), outputs:(X@GRAD).'''
        self.assertEqual(expected, str(backward_op))

        batch_size = 3
        class_num = 5
        # Initialize X and add 1e-2 for numerical stability
        Y = np.random.rand(batch_size, class_num).astype(np.float32)
        Y = Y + 1e-2
        dY = np.random.rand(batch_size, class_num).astype(np.float32)

        expected = label_softmax_grad(Y, dY)

        scope = core.Scope()
        places = []
        places.append(core.CPUPlace())
        if core.is_compile_gpu():
            places.append(core.GPUPlace(0))

        for place in places:
            y = scope.new_var("Y")
            y_tensor = y.get_tensor()
            y_tensor.set_dims([batch_size, class_num])
            y_tensor.alloc_float(place)
            y_tensor.set(Y, place)

            dy = scope.new_var("Y@GRAD")
            dy_tensor = dy.get_tensor()
            dy_tensor.set_dims([batch_size, class_num])
            dy_tensor.alloc_float(place)
            dy_tensor.set(dY, place)

            x = scope.new_var("X")
            dx = scope.new_var("X@GRAD")

            tensor = scope.find_var("X@GRAD").get_tensor()
            backward_op.infer_shape(scope)
            self.assertEqual([batch_size, class_num], tensor.shape())

            ctx = core.DeviceContext.create(place)
            backward_op.run(scope, ctx)
            actual = np.array(tensor)

            np.testing.assert_almost_equal(actual, expected, decimal=3)


class SoftmaxGradOpTest(GradientChecker):
    def test_softmax(self):
        op = create_op("softmax")
        inputs = {"X": np.random.random((3, 4)).astype("float32")}
        self.check_grad(op, inputs, set("X"), "Y")


if __name__ == '__main__':
    unittest.main()
