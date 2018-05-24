import unittest
import numpy as np
from op_test import OpTest


def conv3dtranspose_forward_naive(input_, filter_, attrs):
    in_n, in_c, in_d, in_h, in_w = input_.shape
    f_c, out_c, f_d, f_h, f_w = filter_.shape
    assert in_c == f_c

    stride, pad, dilations = attrs['strides'], attrs['paddings'], attrs[
        'dilations']

    d_bolck_d = dilations[0] * (f_d - 1) + 1
    d_bolck_h = dilations[1] * (f_h - 1) + 1
    d_bolck_w = dilations[2] * (f_w - 1) + 1
    out_d = (in_d - 1) * stride[0] + d_bolck_d
    out_h = (in_h - 1) * stride[1] + d_bolck_h
    out_w = (in_w - 1) * stride[2] + d_bolck_w
    out = np.zeros((in_n, out_c, out_d, out_h, out_w))

    for n in range(in_n):
        for d in range(in_d):
            for i in range(in_h):
                for j in range(in_w):
                    input_masked = input_[n, :, d, i, j]  # (c)
                    input_masked = np.reshape(input_masked, (in_c, 1, 1, 1))
                    input_masked = np.tile(input_masked, (1, f_d, f_h, f_w))

                    for k in range(out_c):
                        tmp_out = np.sum(input_masked * filter_[:, k, :, :, :],
                                         axis=0)
                        d1, d2 = d * stride[0], d * stride[0] + d_bolck_d
                        i1, i2 = i * stride[1], i * stride[1] + d_bolck_h
                        j1, j2 = j * stride[2], j * stride[2] + d_bolck_w
                        out[n, k, d1:d2:dilations[0], i1:i2:dilations[1], j1:j2:
                            dilations[2]] += tmp_out

    out = out[:, :, pad[0]:out_d - pad[0], pad[1]:out_h - pad[1], pad[2]:out_w -
              pad[2]]
    return out


class TestConv3dTransposeOp(OpTest):
    def setUp(self):
        # init as conv transpose
        self.init_op_type()
        self.init_test_case()

        input_ = np.random.random(self.input_size).astype("float32")
        filter_ = np.random.random(self.filter_size).astype("float32")

        self.inputs = {'Input': input_, 'Filter': filter_}
        self.attrs = {
            'strides': self.stride,
            'paddings': self.pad,
            'dilations': self.dilations
        }

        output = conv3dtranspose_forward_naive(input_, filter_,
                                               self.attrs).astype("float32")

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
        self.pad = [0, 0, 0]
        self.stride = [1, 1, 1]
        self.dilations = [1, 1, 1]
        self.input_size = [2, 3, 5, 5, 5]  # NCDHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 3, 3]

    def init_op_type(self):
        self.op_type = "conv3d_transpose"


class TestWithPad(TestConv3dTransposeOp):
    def init_test_case(self):
        self.pad = [1, 1, 1]
        self.stride = [1, 1, 1]
        self.dilations = [1, 1, 1]
        self.input_size = [2, 3, 5, 5, 5]  # NCDHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 3, 3]


class TestWithStride(TestConv3dTransposeOp):
    def init_test_case(self):
        self.pad = [1, 1, 1]
        self.stride = [2, 2, 2]
        self.dilations = [1, 1, 1]
        self.input_size = [2, 3, 5, 5, 5]  # NCDHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 3, 3]


class TestWithDilation(TestConv3dTransposeOp):
    def init_test_case(self):
        self.pad = [1, 1, 1]
        self.stride = [1, 1, 1]
        self.dilations = [2, 2, 2]
        self.input_size = [2, 3, 5, 5, 5]  # NCDHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 3, 3]


# ------------ test_cudnn ------------
class TestCudnn(TestConv3dTransposeOp):
    def init_op_type(self):
        self.op_type = "conv3d_transpose_cudnn"


class TestCudnnWithPad(TestWithPad):
    def init_test_case(self):
        self.pad = [1, 1, 1]
        self.stride = [1, 1, 1]
        self.dilations = [1, 1, 1]
        self.input_size = [2, 3, 5, 5, 5]  # NCDHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 3, 3]

    def init_op_type(self):
        self.op_type = "conv3d_transpose_cudnn"


class TestCudnnWithStride(TestWithStride):
    def init_test_case(self):
        self.pad = [1, 1, 1]
        self.stride = [2, 2, 2]
        self.dilations = [1, 1, 1]
        self.input_size = [2, 3, 5, 5, 5]  # NCDHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 3, 3]

    def init_op_type(self):
        self.op_type = "conv3d_transpose_cudnn"


# #cudnn v5 does not support dilation conv.
# class TestCudnnWithDilation(TestWithDilation):
#     def init_test_case(self):
#         self.pad = [1, 1, 1]
#         self.stride = [2, 2, 2]
#         self.dilations = [2, 2, 2]
#         self.input_size = [2, 3, 5, 5, 5]  # NCDHW
#         f_c = self.input_size[1]
#         self.filter_size = [f_c, 6, 3, 3, 3]
#
#     def init_op_type(self):
#         self.op_type = "conv3d_transpose_cudnn"

if __name__ == '__main__':
    unittest.main()
