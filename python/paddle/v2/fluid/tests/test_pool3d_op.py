import unittest
import numpy as np
from op_test import OpTest


def max_pool3D_forward_naive(x, ksize, strides, paddings=[0, 0], global_pool=0):

    N, C, D, H, W = x.shape
    if global_pool == 1:
        ksize = [D, H, W]
    D_out = (D - ksize[0] + 2 * paddings[0]) / strides[0] + 1
    H_out = (H - ksize[1] + 2 * paddings[1]) / strides[1] + 1
    W_out = (W - ksize[2] + 2 * paddings[2]) / strides[2] + 1
    out = np.zeros((N, C, D_out, H_out, W_out))
    for k in xrange(D_out):
        d_start = np.max((k * strides[0] - paddings[0], 0))
        d_end = np.min((k * strides[0] + ksize[0] - paddings[0], D))
        for i in xrange(H_out):
            h_start = np.max((i * strides[0] - paddings[0], 0))
            h_end = np.min((i * strides[0] + ksize[0] - paddings[0], H))
            for j in xrange(W_out):
                w_start = np.max((j * strides[1] - paddings[1], 0))
                w_end = np.min((j * strides[1] + ksize[1] - paddings[1], W))
                x_masked = x[:, :, d_start:d_end, h_start:h_end, w_start:w_end]

                out[:, :, k, i, j] = np.max(x_masked, axis=(2, 3, 4))
    return out


def avg_pool3D_forward_naive(x, ksize, strides, paddings=[0, 0], global_pool=0):

    N, C, D, H, W = x.shape
    if global_pool == 1:
        ksize = [D, H, W]
    D_out = (D - ksize[0] + 2 * paddings[0]) / strides[0] + 1
    H_out = (H - ksize[1] + 2 * paddings[1]) / strides[1] + 1
    W_out = (W - ksize[2] + 2 * paddings[2]) / strides[2] + 1
    out = np.zeros((N, C, D_out, H_out, W_out))
    for k in xrange(D_out):
        d_start = np.max((k * strides[0] - paddings[0], 0))
        d_end = np.min((k * strides[0] + ksize[0] - paddings[0], D))
        for i in xrange(H_out):
            h_start = np.max((i * strides[0] - paddings[0], 0))
            h_end = np.min((i * strides[0] + ksize[0] - paddings[0], H))
            for j in xrange(W_out):
                w_start = np.max((j * strides[1] - paddings[1], 0))
                w_end = np.min((j * strides[1] + ksize[1] - paddings[1], W))
                x_masked = x[:, :, d_start:d_end, h_start:h_end, w_start:w_end]

                out[:, :, k, i, j] = np.sum(x_masked, axis=(2, 3, 4)) / (
                    (d_end - d_start) * (h_end - h_start) * (w_end - w_start))
    return out


class TestPool3d_Op(OpTest):
    def setUp(self):
        self.init_test_case()
        if self.global_pool:
            self.paddings = [0 for _ in range(len(self.paddings))]
        input = np.random.random(self.shape).astype("float32")
        output = self.pool3D_forward_naive(input, self.ksize, self.strides,
                                           self.paddings,
                                           self.global_pool).astype("float32")
        self.inputs = {'X': input}

        self.attrs = {
            'strides': self.strides,
            'paddings': self.paddings,
            'ksize': self.ksize,
            'pooling_type': self.pool_type,
            'global_pooling': self.global_pool,
        }

        self.outputs = {'Out': output.astype('float32')}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.pool_type != "max":
            self.check_grad(set(['X']), 'Out', max_relative_error=0.07)

    def init_test_case(self):
        self.global_pool = True
        self.op_type = "pool3d"
        self.pool_type = "avg"
        self.pool3D_forward_naive = avg_pool3D_forward_naive
        self.shape = [2, 3, 5, 5, 5]
        self.ksize = [3, 3, 3]
        self.strides = [1, 1, 1]
        self.paddings = [0, 0, 0]


class TestCase1(TestPool3d_Op):
    def init_test_case(self):
        self.global_pool = False
        self.op_type = "pool3d"
        self.pool_type = "avg"
        self.pool3D_forward_naive = avg_pool3D_forward_naive
        self.shape = [2, 3, 7, 7, 7]
        self.ksize = [3, 3, 3]
        self.strides = [1, 1, 1]
        self.paddings = [0, 0, 0]


class TestCase2(TestPool3d_Op):
    def init_test_case(self):
        self.global_pool = False
        self.op_type = "pool3d"
        self.pool_type = "avg"
        self.pool3D_forward_naive = avg_pool3D_forward_naive
        self.shape = [2, 3, 7, 7, 7]
        self.ksize = [3, 3, 3]
        self.strides = [1, 1, 1]
        self.paddings = [1, 1, 1]


class TestCase3(TestPool3d_Op):
    def init_test_case(self):
        self.global_pool = True
        self.op_type = "pool3d"
        self.pool_type = "max"
        self.pool3D_forward_naive = max_pool3D_forward_naive
        self.shape = [2, 3, 5, 5, 5]
        self.ksize = [3, 3, 3]
        self.strides = [1, 1, 1]
        self.paddings = [0, 0, 0]


class TestCase4(TestPool3d_Op):
    def init_test_case(self):
        self.global_pool = False
        self.op_type = "pool3d"
        self.pool_type = "max"
        self.pool3D_forward_naive = max_pool3D_forward_naive
        self.shape = [2, 3, 7, 7, 7]
        self.ksize = [3, 3, 3]
        self.strides = [1, 1, 1]
        self.paddings = [0, 0, 0]


class TestCase5(TestPool3d_Op):
    def init_test_case(self):
        self.global_pool = False
        self.op_type = "pool3d"
        self.pool_type = "max"
        self.pool3D_forward_naive = max_pool3D_forward_naive
        self.shape = [2, 3, 7, 7, 7]
        self.ksize = [3, 3, 3]
        self.strides = [1, 1, 1]
        self.paddings = [1, 1, 1]


if __name__ == '__main__':
    unittest.main()
