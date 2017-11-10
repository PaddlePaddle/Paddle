import unittest
import numpy as np
from op_test import OpTest


def conv3d_forward_naive(input, filter, group, conv_param):
    in_n, in_c, in_d, in_h, in_w = input.shape
    out_c, f_c, f_d, f_h, f_w = filter.shape
    assert f_c * group == in_c
    assert np.mod(out_c, group) == 0
    sub_out_c = out_c / group

    stride, pad, dilation = conv_param['stride'], conv_param['pad'], conv_param[
        'dilations']

    out_d = 1 + (in_d + 2 * pad[0] - (dilation[0] * (f_d - 1) + 1)) / stride[0]
    out_h = 1 + (in_h + 2 * pad[1] - (dilation[1] * (f_h - 1) + 1)) / stride[1]
    out_w = 1 + (in_w + 2 * pad[2] - (dilation[2] * (f_w - 1) + 1)) / stride[2]

    out = np.zeros((in_n, out_c, out_d, out_h, out_w))

    d_bolck_d = (dilation[0] * (f_d - 1) + 1)
    d_bolck_h = (dilation[1] * (f_h - 1) + 1)
    d_bolck_w = (dilation[2] * (f_w - 1) + 1)

    input_pad = np.pad(input, ((0, ), (0, ), (pad[0], ), (pad[1], ),
                               (pad[2], )),
                       mode='constant',
                       constant_values=0)

    filter_dilation = np.zeros((out_c, f_c, d_bolck_d, d_bolck_h, d_bolck_w))
    filter_dilation[:, :, 0:d_bolck_d:dilation[0], 0:d_bolck_h:dilation[1], 0:
                    d_bolck_w:dilation[2]] = filter

    for d in range(out_d):
        for i in range(out_h):
            for j in range(out_w):
                for g in range(group):
                    input_pad_masked = \
                        input_pad[:, g * f_c:(g + 1) * f_c,
                        d * stride[0]:d * stride[0] + d_bolck_d,
                        i * stride[1]:i * stride[1] + d_bolck_h,
                        j * stride[2]:j * stride[2] + d_bolck_w]

                    f_sub = filter_dilation[g * sub_out_c:(g + 1) *
                                            sub_out_c, :, :, :, :]
                    for k in range(sub_out_c):
                        out[:, g * sub_out_c + k, d, i, j] = \
                            np.sum(input_pad_masked * f_sub[k, :, :, :, :],
                                   axis=(1, 2, 3, 4))

    return out


class TestConv3dOp(OpTest):
    def setUp(self):
        self.init_group()
        self.init_op_type()
        self.init_dilation()
        self.init_test_case()

        conv3d_param = {
            'stride': self.stride,
            'pad': self.pad,
            'dilations': self.dilations
        }
        input = np.random.random(self.input_size).astype("float32")
        filter = np.random.random(self.filter_size).astype("float32")
        output = conv3d_forward_naive(input, filter, self.groups,
                                      conv3d_param).astype("float32")

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
            set(['Input', 'Filter']), 'Output', max_relative_error=0.03)

    def test_check_grad_no_filter(self):
        self.check_grad(
            ['Input'],
            'Output',
            max_relative_error=0.03,
            no_grad_set=set(['Filter']))

    def test_check_grad_no_input(self):
        self.check_grad(
            ['Filter'],
            'Output',
            max_relative_error=0.03,
            no_grad_set=set(['Input']))

    def init_test_case(self):
        self.pad = [0, 0, 0]
        self.stride = [1, 1, 1]
        self.input_size = [2, 3, 4, 4, 4]  # NCDHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] / self.groups
        self.filter_size = [6, f_c, 3, 3, 3]

    def init_dilation(self):
        self.dilations = [1, 1, 1]

    def init_group(self):
        self.groups = 1

    def init_op_type(self):
        self.op_type = "conv3d"


class TestCase1(TestConv3dOp):
    def init_test_case(self):
        self.pad = [1, 1, 1]
        self.stride = [1, 1, 1]
        self.input_size = [2, 3, 4, 4, 4]  # NCDHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] / self.groups
        self.filter_size = [6, f_c, 3, 3, 3]


class TestWithGroup1(TestConv3dOp):
    def init_group(self):
        self.groups = 3


class TestWithGroup2(TestCase1):
    def init_group(self):
        self.groups = 3


class TestWith1x1(TestConv3dOp):
    def init_test_case(self):
        self.pad = [0, 0, 0]
        self.stride = [1, 1, 1]
        self.input_size = [2, 3, 4, 4, 4]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] / self.groups
        self.filter_size = [6, f_c, 1, 1, 1]

    def init_dilation(self):
        self.dilations = [1, 1, 1]

    def init_group(self):
        self.groups = 3


class TestWithDilation(TestConv3dOp):
    def init_test_case(self):
        self.pad = [0, 0, 0]
        self.stride = [1, 1, 1]
        self.input_size = [2, 3, 6, 6, 6]  # NCDHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] / self.groups
        self.filter_size = [6, f_c, 2, 2, 2]

    def init_dilation(self):
        self.dilations = [2, 2, 2]

    def init_group(self):
        self.groups = 3


if __name__ == '__main__':
    unittest.main()
