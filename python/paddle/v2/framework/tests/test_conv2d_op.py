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
                input_pad_masked = input_pad[:, g * f_c:(
                    g + 1) * f_c, i * stride[0]:i * stride[0] + f_h, j * stride[
                        1]:j * stride[1] + f_w]
                f_sub = filter[g * sub_out_c:(g + 1) * sub_out_c, :, :, :]
                for k in range(sub_out_c):
                    out[:, g * sub_out_c + k, i, j] = np.sum(input_pad_masked *
                                                             f_sub[k, :, :, :],
                                                             axis=(1, 2, 3))

    return out


class TestConv2dOp(OpTest):
    def setUp(self):
        self.init_groups()
        self.op_type = "conv2d"
        pad = [0, 0]
        stride = [1, 1]
        input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(input_size[1], self.groups) == 0
        f_c = input_size[1] / self.groups
        filter_size = [6, f_c, 3, 3]

        conv2d_param = {'stride': stride, 'pad': pad}

        input = np.random.random(input_size).astype("float32")
        filter = np.random.random(filter_size).astype("float32")

        output = conv2d_forward_naive(input, filter, self.groups, conv2d_param)

        self.inputs = {'Input': input, 'Filter': filter}
        self.attrs = {'strides': stride, 'paddings': pad, 'groups': self.groups}
        self.outputs = {'Output': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(
            set(['Input', 'Filter']), 'Output', max_relative_error=0.05)

    def test_check_grad_no_filter(self):
        self.check_grad(
            ['Input'],
            'Output',
            max_relative_error=0.05,
            no_grad_set=set(['Filter']))

    def test_check_grad_no_input(self):
        self.check_grad(
            ['Filter'],
            'Output',
            max_relative_error=0.05,
            no_grad_set=set(['Input']))

    def init_groups(self):
        self.groups = 1


class TestWithGroup(TestConv2dOp):
    def init_groups(self):
        self.groups = 3


if __name__ == '__main__':
    unittest.main()
