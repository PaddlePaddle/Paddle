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
        self.init_test_case()
        self.init_op_type()
        self.init_pool_type()
        if self.global_pool:
            self.paddings = [0 for _ in range(len(self.paddings))]
        input = np.random.random(self.shape).astype("float32")
        output = self.pool2D_forward_naive(input, self.ksize, self.strides,
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
        self.pool2D_forward_naive = avg_pool2D_forward_naive
        self.shape = [2, 3, 5, 5]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [0, 0]

    def init_op_type(self):
        self.op_type = "pool2d"

    def init_pool_type(self):
        self.pool_type = "avg"


class TestCase1(TestPool2d_Op):
    def init_test_case(self):
        self.global_pool = False
        self.pool2D_forward_naive = avg_pool2D_forward_naive
        self.shape = [2, 3, 7, 7]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [0, 0]

    def init_op_type(self):
        self.op_type = "pool2d"

    def init_pool_type(self):
        self.pool_type = "avg"


class TestCase2(TestPool2d_Op):
    def init_test_case(self):
        self.global_pool = False
        self.pool2D_forward_naive = avg_pool2D_forward_naive
        self.shape = [2, 3, 7, 7]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 1]

    def init_op_type(self):
        self.op_type = "pool2d"

    def init_pool_type(self):
        self.pool_type = "avg"


class TestCase3(TestPool2d_Op):
    def init_test_case(self):
        self.global_pool = True
        self.pool2D_forward_naive = max_pool2D_forward_naive
        self.shape = [2, 3, 5, 5]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [0, 0]

    def init_op_type(self):
        self.op_type = "pool2d"

    def init_pool_type(self):
        self.pool_type = "max"


class TestCase4(TestPool2d_Op):
    def init_test_case(self):
        self.global_pool = False
        self.pool2D_forward_naive = max_pool2D_forward_naive
        self.shape = [2, 3, 7, 7]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [0, 0]

    def init_op_type(self):
        self.op_type = "pool2d"

    def init_pool_type(self):
        self.pool_type = "max"


class TestCase5(TestPool2d_Op):
    def init_test_case(self):
        self.global_pool = False
        self.pool2D_forward_naive = max_pool2D_forward_naive
        self.shape = [2, 3, 7, 7]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 1]

    def init_op_type(self):
        self.op_type = "pool2d"

    def init_pool_type(self):
        self.pool_type = "max"


#--------------------test pool2d_cudnn--------------------
class TestCaseCudnn1(TestPool2d_Op):
    def init_test_case(self):
        self.global_pool = True
        self.pool2D_forward_naive = avg_pool2D_forward_naive
        self.shape = [2, 3, 5, 5]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [0, 0]

    def init_op_type(self):
        self.op_type = "pool2d_cudnn"

    def init_pool_type(self):
        self.pool_type = "avg"


class TestCaseCudnn2(TestPool2d_Op):
    def init_test_case(self):
        self.global_pool = False
        self.pool2D_forward_naive = avg_pool2D_forward_naive
        self.shape = [2, 3, 7, 7]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [0, 0]

    def init_op_type(self):
        self.op_type = "pool2d_cudnn"

    def init_pool_type(self):
        self.pool_type = "avg"


class TestCaseCudnn3(TestPool2d_Op):
    def init_test_case(self):
        self.global_pool = False
        self.pool2D_forward_naive = avg_pool2D_forward_naive
        self.shape = [2, 3, 7, 7]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 1]

    def init_op_type(self):
        self.op_type = "pool2d_cudnn"

    def init_pool_type(self):
        self.pool_type = "avg"


class TestCaseCudnn4(TestPool2d_Op):
    def init_test_case(self):
        self.global_pool = True
        self.pool2D_forward_naive = max_pool2D_forward_naive
        self.shape = [2, 3, 5, 5]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [0, 0]

    def init_op_type(self):
        self.op_type = "pool2d_cudnn"

    def init_pool_type(self):
        self.pool_type = "max"


class TestCaseCudnn5(TestPool2d_Op):
    def init_test_case(self):
        self.global_pool = False
        self.pool2D_forward_naive = max_pool2D_forward_naive
        self.shape = [2, 3, 7, 7]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [0, 0]

    def init_op_type(self):
        self.op_type = "pool2d_cudnn"

    def init_pool_type(self):
        self.pool_type = "max"


class TestCaseCudnn6(TestPool2d_Op):
    def init_test_case(self):
        self.global_pool = False
        self.pool2D_forward_naive = max_pool2D_forward_naive
        self.shape = [2, 3, 7, 7]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 1]

    def init_op_type(self):
        self.op_type = "pool2d_cudnn"

    def init_pool_type(self):
        self.pool_type = "max"


if __name__ == '__main__':
    unittest.main()
