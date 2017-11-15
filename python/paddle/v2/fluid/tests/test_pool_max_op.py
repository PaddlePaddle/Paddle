import unittest
import numpy as np
from op_test import OpTest


def max_pool3D_forward_naive(x, ksize, strides, paddings, global_pool=0):

    N, C, D, H, W = x.shape
    if global_pool == 1:
        ksize = [D, H, W]
    D_out = (D - ksize[0] + 2 * paddings[0]) / strides[0] + 1
    H_out = (H - ksize[1] + 2 * paddings[1]) / strides[1] + 1
    W_out = (W - ksize[2] + 2 * paddings[2]) / strides[2] + 1
    out = np.zeros((N, C, D_out, H_out, W_out))
    mask = np.zeros((N, C, D_out, H_out, W_out))
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

                for n in xrange(N):
                    for c in xrange(C):
                        arr = x_masked[n, c, :, :, :]
                        index = np.where(arr == np.max(arr))
                        sub_deep = index[0][0]
                        sub_row = index[1][0]
                        sub_col = index[2][0]
                        index = ((d_start + sub_deep) * H +
                                 (h_start + sub_row)) * W + w_start + sub_col
                        mask[n, c, k, i, j] = index

    return out, mask


def max_pool2D_forward_naive(x, ksize, strides, paddings, global_pool=0):

    N, C, H, W = x.shape
    if global_pool == 1:
        ksize = [H, W]
    H_out = (H - ksize[0] + 2 * paddings[0]) / strides[0] + 1
    W_out = (W - ksize[1] + 2 * paddings[1]) / strides[1] + 1
    out = np.zeros((N, C, H_out, W_out))
    mask = np.zeros((N, C, H_out, W_out))
    for i in xrange(H_out):
        for j in xrange(W_out):
            r_start = np.max((i * strides[0] - paddings[0], 0))
            r_end = np.min((i * strides[0] + ksize[0] - paddings[0], H))
            c_start = np.max((j * strides[1] - paddings[1], 0))
            c_end = np.min((j * strides[1] + ksize[1] - paddings[1], W))
            x_masked = x[:, :, r_start:r_end, c_start:c_end]

            out[:, :, i, j] = np.max(x_masked, axis=(2, 3))

            for n in xrange(N):
                for c in xrange(C):
                    arr = x_masked[n, c, :, :]
                    index = np.where(arr == np.max(arr))
                    sub_row = index[0][0]
                    sub_col = index[1][0]
                    index = (r_start + sub_row) * W + c_start + sub_col
                    mask[n, c, i, j] = index

    return out, mask


class TestMaxPoolWithIndex_Op(OpTest):
    def setUp(self):
        self.init_test_case()
        if self.global_pool:
            self.paddings = [0 for _ in range(len(self.paddings))]
        input = np.random.random(self.shape).astype("float32")
        output, mask = self.pool_forward_naive(input, self.ksize, self.strides,
                                               self.paddings, self.global_pool)
        output = output.astype("float32")
        mask = mask.astype("float32")

        self.attrs = {
            'strides': self.strides,
            'paddings': self.paddings,
            'ksize': self.ksize,
            'global_pooling': self.global_pool,
        }

        self.inputs = {'X': input}
        self.outputs = {'Out': output, "Mask": mask}

    def test_check_output(self):
        self.check_output()

    # def test_check_grad(self):
    #     self.check_grad(set(['X']), ['Out'], max_relative_error=0.07)

    def init_test_case(self):
        self.global_pool = True
        self.index = "max_pool3d_with_index"
        self.op_type = "%s" % self.index
        self.pool_forward_naive = max_pool3D_forward_naive
        self.shape = [2, 3, 5, 5, 5]
        self.ksize = [3, 3, 3]
        self.strides = [1, 1, 1]
        self.paddings = [1, 1, 1]


class TestCase1(TestMaxPoolWithIndex_Op):
    def init_test_case(self):
        self.global_pool = True
        self.op_type = "max_pool3d_with_index"
        self.pool_forward_naive = max_pool3D_forward_naive
        self.shape = [2, 3, 5, 5, 5]
        self.ksize = [3, 3, 3]
        self.strides = [1, 1, 1]
        self.paddings = [1, 1, 1]


class TestCase2(TestMaxPoolWithIndex_Op):
    def init_test_case(self):
        self.global_pool = False
        self.op_type = "max_pool3d_with_index"
        self.pool_forward_naive = max_pool3D_forward_naive
        self.shape = [2, 3, 7, 7, 7]
        self.ksize = [3, 3, 3]
        self.strides = [1, 1, 1]
        self.paddings = [1, 1, 1]


class TestCase3(TestMaxPoolWithIndex_Op):
    def init_test_case(self):
        self.global_pool = False
        self.op_type = "max_pool3d_with_index"
        self.pool_forward_naive = max_pool3D_forward_naive
        self.shape = [2, 3, 7, 7, 7]
        self.ksize = [3, 3, 3]
        self.strides = [2, 2, 2]
        self.paddings = [0, 0, 0]


class TestCase4(TestMaxPoolWithIndex_Op):
    def init_test_case(self):
        self.global_pool = True
        self.op_type = "max_pool3d_with_index"
        self.pool_forward_naive = max_pool3D_forward_naive
        self.shape = [2, 3, 5, 5, 5]
        self.ksize = [3, 3, 3]
        self.strides = [1, 1, 1]
        self.paddings = [1, 1, 1]


class TestCase5(TestMaxPoolWithIndex_Op):
    def init_test_case(self):
        self.global_pool = True
        self.op_type = "max_pool3d_with_index"
        self.pool_forward_naive = max_pool3D_forward_naive
        self.shape = [2, 3, 5, 5, 5]
        self.ksize = [3, 3, 3]
        self.strides = [2, 2, 2]
        self.paddings = [0, 0, 0]


class TestCase6(TestMaxPoolWithIndex_Op):
    def init_test_case(self):
        self.global_pool = False
        self.op_type = "max_pool2d_with_index"
        self.pool_forward_naive = max_pool2D_forward_naive
        self.shape = [2, 3, 7, 7]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 1]


class TestCase7(TestMaxPoolWithIndex_Op):
    def init_test_case(self):
        self.global_pool = False
        self.op_type = "max_pool2d_with_index"
        self.pool_forward_naive = max_pool2D_forward_naive
        self.shape = [2, 3, 7, 7]
        self.ksize = [3, 3]
        self.strides = [2, 2]
        self.paddings = [0, 0]


class TestCase8(TestMaxPoolWithIndex_Op):
    def init_test_case(self):
        self.global_pool = True
        self.op_type = "max_pool2d_with_index"
        self.pool_forward_naive = max_pool2D_forward_naive
        self.shape = [2, 3, 5, 5]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 1]


class TestCase9(TestMaxPoolWithIndex_Op):
    def init_test_case(self):
        self.global_pool = True
        self.op_type = "max_pool2d_with_index"
        self.pool_forward_naive = max_pool2D_forward_naive
        self.shape = [2, 3, 5, 5]
        self.ksize = [3, 3]
        self.strides = [2, 2]
        self.paddings = [0, 0]


if __name__ == '__main__':
    unittest.main()
