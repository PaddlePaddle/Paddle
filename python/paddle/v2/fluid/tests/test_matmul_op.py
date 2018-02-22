import unittest
import numpy as np
from op_test import OpTest


def generate_compatible_shapes(dim_X, dim_Y, transpose_X, transpose_Y):
    BATCH_SIZE = 2
    M = 3
    N = 4
    K = 5
    if (dim_X == 1 and transpose_X) or (dim_Y == 1 and transpose_Y):
        K = 1
    if dim_X == 1:
        if transpose_X:
            shape_X = [M]
        else:
            shape_X = [K]
    if dim_Y == 1:
        if transpose_Y:
            shape_Y = [N]
        else:
            shape_Y = [K]
    if dim_X >= 2:
        if transpose_X:
            shape_X = [K, M]
        else:
            shape_X = [M, K]
    if dim_X == 3:
        shape_X = [BATCH_SIZE] + shape_X
    if dim_Y >= 2:
        if transpose_Y:
            shape_Y = [N, K]
        else:
            shape_Y = [K, N]
    if dim_Y == 3:
        shape_Y = [BATCH_SIZE] + shape_Y
    return shape_X, shape_Y


def reference_matmul(X, Y, transpose_X=False, transpose_Y=False):
    """Reference forward implementation using np.matmul."""
    # np.matmul does not support the transpose flags, so we manually
    # transpose X and Y appropriately.
    if transpose_X:
        if X.ndim == 1:
            X = X.reshape((X.size, 1))
        elif X.ndim == 2:
            X = X.T
        elif X.ndim == 3:
            X = np.transpose(X, (0, 2, 1))
        else:
            raise ValueError('X must have between 1 and 3 dimensions')
    if transpose_Y:
        if Y.ndim == 1:
            Y = Y.reshape((1, Y.size))
        elif Y.ndim == 2:
            Y = Y.T
        elif Y.ndim == 3:
            Y = np.transpose(Y, (0, 2, 1))
        else:
            raise ValueError('Y must have between 1 and 3 dimensions')
    Out = np.matmul(X, Y)
    if not Out.shape:
        # We do not support 0-dimensional Tensors (scalars). So where
        # np.matmul outputs a scalar, we must convert to a Tensor of
        # shape (1, ) instead.
        # Everywhere else, we are compatible with np.matmul.
        Out = np.array([Out], dtype="float32")
    return Out


class Generator(object):
    def setUp(self):
        self.op_type = "matmul"
        X = np.random.random(self.shape_X).astype("float32")
        Y = np.random.random(self.shape_Y).astype("float32")
        Out = reference_matmul(X, Y, self.transpose_X, self.transpose_Y)
        self.inputs = {'X': X, 'Y': Y}
        self.attrs = {
            'transpose_X': self.transpose_X,
            'transpose_Y': self.transpose_Y
        }
        self.outputs = {'Out': Out}

    def test_check_output(self):
        self.check_output(atol=1e-2)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', max_relative_error=0.5)

    def test_check_grad_ignore_x(self):
        self.check_grad(
            ['Y'], 'Out', max_relative_error=0.5, no_grad_set=set("X"))

    def test_check_grad_ignore_y(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.5, no_grad_set=set('Y'))


# Generate test cases for all possibilities
for dim_X in [1, 2, 3]:
    for dim_Y in [1, 2, 3]:
        for transpose_X in [False, True]:
            for transpose_Y in [False, True]:
                test_name = (
                    'TestMatMulOp_dimX_{}_dim_Y_{}_transX_{}_transY_{}'.format(
                        dim_X, dim_Y, transpose_X, transpose_Y))
                shape_X, shape_Y = generate_compatible_shapes(
                    dim_X, dim_Y, transpose_X, transpose_Y)
                test_class = type(test_name, (Generator, OpTest), {
                    'shape_X': shape_X,
                    'shape_Y': shape_Y,
                    'transpose_X': transpose_X,
                    'transpose_Y': transpose_Y,
                })
                globals()[test_name] = test_class

if __name__ == "__main__":
    unittest.main()
