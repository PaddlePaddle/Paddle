import unittest
import numpy as np
from op_test import OpTest


def max_pool2D_forward_naive(x, ksize, strides, paddings=[0, 0], global_pool=0):

    N, C, H, W = x.shape
    if global_pool == 1:
        ksize = [H, W]
    H_out = (H - ksize[0] + 2 * paddings[0]) / strides[0] + 1
    W_out = (W - ksize[1] + 2 * paddings[1]) / strides[1] + 1
    out = np.zeros((N, C, H_out, W_out))
    for i in xrange(H_out):
        for j in xrange(W_out):
            r_start = np.max((i * strides[0] - paddings[0], 0))
            r_end = np.min((i * strides[0] + ksize[0] - paddings[0], H))
            c_start = np.max((j * strides[1] - paddings[1], 0))
            c_end = np.min((j * strides[1] + ksize[1] - paddings[1], W))
            x_masked = x[:, :, r_start:r_end, c_start:c_end]

            out[:, :, i, j] = np.max(x_masked, axis=(2, 3))
    return out


def avg_pool2D_forward_naive(x, ksize, strides, paddings=[0, 0], global_pool=0):

    N, C, H, W = x.shape
    if global_pool == 1:
        ksize = [H, W]
    H_out = (H - ksize[0] + 2 * paddings[0]) / strides[0] + 1
    W_out = (W - ksize[1] + 2 * paddings[1]) / strides[1] + 1
    out = np.zeros((N, C, H_out, W_out))
    for i in xrange(H_out):
        for j in xrange(W_out):
            r_start = np.max((i * strides[0] - paddings[0], 0))
            r_end = np.min((i * strides[0] + ksize[0] - paddings[0], H))
            c_start = np.max((j * strides[1] - paddings[1], 0))
            c_end = np.min((j * strides[1] + ksize[1] - paddings[1], W))
            x_masked = x[:, :, r_start:r_end, c_start:c_end]

            out[:, :, i, j] = np.sum(x_masked, axis=(2, 3)) / (
                (r_end - r_start) * (c_end - c_start))
    return out


class TestPool2d_Op(OpTest):
    def setUp(self):
        self.initTestCase()
        input = np.random.random(self.shape).astype("float32")
        output = self.pool2D_forward_naive(input, self.ksize, self.strides,
                                           self.paddings, self.global_pool)
        self.inputs = {'X': input}

        self.attrs = {
            'strides': self.strides,
            'paddings': self.paddings,
            'ksize': self.ksize,
            'poolingType': self.pool_type,
            'globalPooling': self.global_pool,
        }

        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.pool_type != "max":
            self.check_grad(set(['X']), 'Out', max_relative_error=0.07)

    def initTestCase(self):
        self.global_pool = True
        self.op_type = "pool2d"
        self.pool_type = "avg"
        self.pool2D_forward_naive = avg_pool2D_forward_naive
        self.shape = [2, 3, 5, 5]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [0, 0]


class TestCase1(TestPool2d_Op):
    def initTestCase(self):
        self.global_pool = False
        self.op_type = "pool2d"
        self.pool_type = "avg"
        self.pool2D_forward_naive = avg_pool2D_forward_naive
        self.shape = [2, 3, 7, 7]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [0, 0]


class TestCase2(TestPool2d_Op):
    def initTestCase(self):
        self.global_pool = False
        self.op_type = "pool2d"
        self.pool_type = "avg"
        self.pool2D_forward_naive = avg_pool2D_forward_naive
        self.shape = [2, 3, 7, 7]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 1]


class TestCase3(TestPool2d_Op):
    def initTestCase(self):
        self.global_pool = True
        self.op_type = "pool2d"
        self.pool_type = "max"
        self.pool2D_forward_naive = max_pool2D_forward_naive
        self.shape = [2, 3, 5, 5]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [0, 0]


class TestCase4(TestPool2d_Op):
    def initTestCase(self):
        self.global_pool = False
        self.op_type = "pool2d"
        self.pool_type = "max"
        self.pool2D_forward_naive = max_pool2D_forward_naive
        self.shape = [2, 3, 7, 7]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [0, 0]


class TestCase5(TestPool2d_Op):
    def initTestCase(self):
        self.global_pool = False
        self.op_type = "pool2d"
        self.pool_type = "max"
        self.pool2D_forward_naive = max_pool2D_forward_naive
        self.shape = [2, 3, 7, 7]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 1]


if __name__ == '__main__':
    unittest.main()
