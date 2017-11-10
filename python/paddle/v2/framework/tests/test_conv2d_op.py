import unittest
import numpy as np
from op_test import OpTest


def conv2d_forward_naive(input, filter, group, conv_param):
    in_n, in_c, in_h, in_w = input.shape
    out_c, f_c, f_h, f_w = filter.shape
    assert f_c * group == in_c
    assert np.mod(out_c, group) == 0
    sub_out_c = out_c / group

    stride, pad = conv_param['stride'], conv_param['pad']
    out_h = 1 + (in_h + 2 * pad[0] - f_h) / stride[0]
    out_w = 1 + (in_w + 2 * pad[1] - f_w) / stride[1]
    out = np.zeros((in_n, out_c, out_h, out_w))

    input_pad = np.pad(input, ((0, ), (0, ), (pad[0], ), (pad[1], )),
                       mode='constant',
                       constant_values=0)
    for i in range(out_h):
        for j in range(out_w):
            for g in range(group):
                input_pad_masked = \
                    input_pad[:, g * f_c:(g + 1) * f_c,
                    i * stride[0]:i * stride[0] + f_h,
                    j * stride[1]:j * stride[1] + f_w]

                f_sub = filter[g * sub_out_c:(g + 1) * sub_out_c, :, :, :]
                for k in range(sub_out_c):
                    out[:, g * sub_out_c + k, i, j] = \
                        np.sum(input_pad_masked * f_sub[k, :, :, :],
                               axis=(1, 2, 3))

    return out


class TestConv2dOp(OpTest):
    def setUp(self):
        self.init_op_type()
        self.init_group()
        self.init_test_case()

        conv2d_param = {'stride': self.stride, 'pad': self.pad}
        input = np.random.random(self.input_size).astype("float32")
        filter = np.random.random(self.filter_size).astype("float32")
        output = conv2d_forward_naive(input, filter, self.groups,
                                      conv2d_param).astype('float32')

        self.inputs = {'Input': input, 'Filter': filter}
        self.attrs = {
            'strides': self.stride,
            'paddings': self.pad,
            'groups': self.groups,
            'dilations': self.dilations
        }
        self.outputs = {'Output': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(
            set(['Input', 'Filter']), 'Output', max_relative_error=0.02)

    def test_check_grad_no_filter(self):
        self.check_grad(
            ['Input'],
            'Output',
            max_relative_error=0.02,
            no_grad_set=set(['Filter']))

    def test_check_grad_no_input(self):
        self.check_grad(
            ['Filter'],
            'Output',
            max_relative_error=0.02,
            no_grad_set=set(['Input']))

    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.dilations = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] / self.groups
        self.filter_size = [6, f_c, 3, 3]

    def init_group(self):
        self.groups = 1

    def init_op_type(self):
        self.op_type = "conv2d"


class TestWithGroup(TestConv2dOp):
    def init_group(self):
        self.groups = 3

    def init_op_type(self):
        self.op_type = "conv2d"


#----------------Conv2dCudnn----------------


class TestCudnn(TestConv2dOp):
    def init_group(self):
        self.groups = 1

    def init_op_type(self):
        self.op_type = "conv_cudnn"


class TestCudnnWithGroup(TestConv2dOp):
    def init_group(self):
        self.groups = 3

    def init_op_type(self):
        self.op_type = "conv_cudnn"


if __name__ == '__main__':
    unittest.main()
