import unittest
import numpy as np
from op_test import OpTest


def conv2dtranspose_forward_naive(input_, filter_, conv2dtranspose_param):
    # [2, 3, 5, 5]
    in_n, in_c, in_h, in_w = input_.shape
    # [3, 6, 3, 3]
    f_c, out_c, f_h, f_w = filter_.shape
    assert in_c == f_c

    stride, pad = conv2dtranspose_param['stride'], conv2dtranspose_param['pad']
    out_h = (in_h - 1) * stride[0] + f_h
    out_w = (in_w - 1) * stride[1] + f_w

    out = np.zeros((in_n, out_c, out_h, out_w))

    for n in range(in_n):
        for i in range(in_h):
            for j in range(in_w):
                input_masked = input_[n, :, i, j]  # (c)
                input_masked = np.reshape(input_masked, (in_c, 1, 1))
                input_masked = np.tile(input_masked, (1, f_h, f_w))

                for k in range(out_c):
                    tmp_out = np.sum(input_masked * filter_[:, k, :, :], axis=0)
                    i1, i2 = i * stride[0], i * stride[0] + f_h
                    j1, j2 = j * stride[0], j * stride[0] + f_w
                    out[n, k, i1:i2, j1:j2] += tmp_out

    return out


class TestConv2dTransposeOp(OpTest):
    def setUp(self):
        # init as conv transpose
        self.init_op_type()

        # [2, 3, 5, 5] -> kernel [3, 6, 3, 3] -> output [2, 6, 7, 7]
        self.init_test_case()

        conv2dtranspose_param = {'stride': self.stride, 'pad': self.pad}
        input_ = np.random.random(self.input_size).astype("float32")
        filter_ = np.random.random(self.filter_size).astype("float32")
        output = conv2dtranspose_forward_naive(
            input_, filter_, conv2dtranspose_param).astype('float32')

        self.inputs = {'Input': input_, 'Filter': filter_}
        self.attrs = {
            'strides': self.stride,
            'paddings': self.pad,
            'dilations': self.dilations
        }
        self.outputs = {'Output': output}

    def test_check_output(self):
        print 'check output here for', self.op_type
        self.check_output()

    def test_check_grad_no_input(self):
        self.check_grad(
            ['Filter'],
            'Output',
            max_relative_error=0.02,
            no_grad_set=set(['Input']))

    def test_check_grad_no_filter(self):
        self.check_grad(
            ['Input'],
            'Output',
            max_relative_error=0.02,
            no_grad_set=set(['Filter']))

    def test_check_grad(self):
        self.check_grad(
            set(['Input', 'Filter']), 'Output', max_relative_error=0.02)

    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.dilations = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 3]

    def init_op_type(self):
        self.op_type = "conv2d_transpose"


# ------------ test_cudnn ------------
class TestCudnn(TestConv2dTransposeOp):
    def init_op_type(self):
        self.op_type = "conv2d_transpose_cudnn"


if __name__ == '__main__':
    unittest.main()
