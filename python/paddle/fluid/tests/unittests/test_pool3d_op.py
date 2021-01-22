#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
from __future__ import division

import unittest
import numpy as np

import paddle
import paddle.fluid.core as core
from op_test import OpTest
import paddle.fluid as fluid


def adaptive_start_index(index, input_size, output_size):
    return int(np.floor(index * input_size / output_size))


def adaptive_end_index(index, input_size, output_size):
    return int(np.ceil((index + 1) * input_size / output_size))


def pool3D_forward_naive(x,
                         ksize,
                         strides,
                         paddings,
                         global_pool=0,
                         ceil_mode=False,
                         exclusive=True,
                         adaptive=False,
                         data_format='NCDHW',
                         pool_type='max',
                         padding_algorithm="EXPLICIT"):
    # update paddings
    def _get_padding_with_SAME(input_shape, pool_size, pool_stride):
        padding = []
        for input_size, filter_size, stride_size in zip(input_shape, pool_size,
                                                        pool_stride):
            out_size = int((input_size + stride_size - 1) / stride_size)
            pad_sum = np.max((
                (out_size - 1) * stride_size + filter_size - input_size, 0))
            pad_0 = int(pad_sum / 2)
            pad_1 = int(pad_sum - pad_0)
            padding.append(pad_0)
            padding.append(pad_1)
        return padding

    if isinstance(padding_algorithm, str):
        padding_algorithm = padding_algorithm.upper()
        if padding_algorithm not in ["SAME", "VALID", "EXPLICIT"]:
            raise ValueError("Unknown Attr(padding_algorithm): '%s'. "
                             "It can only be 'SAME' or 'VALID'." %
                             str(padding_algorithm))

        if padding_algorithm == "VALID":
            paddings = [0, 0, 0, 0, 0, 0]
            if ceil_mode != False:
                raise ValueError(
                    "When Attr(pool_padding) is \"VALID\", Attr(ceil_mode)"
                    " must be False. "
                    "Received ceil_mode: True.")
        elif padding_algorithm == "SAME":
            input_data_shape = []
            if data_format == "NCDHW":
                input_data_shape = x.shape[2:5]
            elif data_format == "NDHWC":
                input_data_shape = x.shape[1:4]
            paddings = _get_padding_with_SAME(input_data_shape, ksize, strides)

    assert len(paddings) == 3 or len(paddings) == 6
    is_sys = True if len(paddings) == 3 else False

    N = x.shape[0]
    C,D, H, W = [x.shape[1], x.shape[2], x.shape[3], x.shape[4]] \
        if data_format == 'NCDHW' else [x.shape[4], x.shape[1], x.shape[2],x.shape[3]]

    if global_pool == 1:
        ksize = [D, H, W]
        paddings = [0 for _ in range(len(paddings))]

    pad_d_forth = paddings[0] if is_sys else paddings[0]
    pad_d_back = paddings[0] if is_sys else paddings[1]
    pad_h_up = paddings[1] if is_sys else paddings[2]
    pad_h_down = paddings[1] if is_sys else paddings[3]
    pad_w_left = paddings[2] if is_sys else paddings[4]
    pad_w_right = paddings[2] if is_sys else paddings[5]

    if adaptive:
        D_out, H_out, W_out = ksize
    else:

        D_out = (D - ksize[0] + pad_d_forth+pad_d_back + strides[0] - 1) // strides[0] + 1 \
            if ceil_mode  else (D - ksize[0] + pad_d_forth+pad_d_back) // strides[0] + 1

        H_out = (H - ksize[1] + pad_h_up + pad_h_down + strides[1] - 1) // strides[1] + 1 \
            if ceil_mode else (H - ksize[1] + pad_h_up + pad_h_down) // strides[1] + 1

        W_out = (W - ksize[2] + pad_w_left + pad_w_right + strides[2] - 1) // strides[2] + 1 \
            if ceil_mode else (W - ksize[2] + pad_w_left + pad_w_right) // strides[2] + 1


    out = np.zeros((N, C, D_out, H_out, W_out)) if data_format=='NCDHW' \
        else np.zeros((N, D_out, H_out, W_out, C))
    for k in range(D_out):
        if adaptive:
            d_start = adaptive_start_index(k, D, ksize[0])
            d_end = adaptive_end_index(k, D, ksize[0])

        for i in range(H_out):
            if adaptive:
                h_start = adaptive_start_index(i, H, ksize[1])
                h_end = adaptive_end_index(i, H, ksize[1])

            for j in range(W_out):
                if adaptive:
                    w_start = adaptive_start_index(j, W, ksize[2])
                    w_end = adaptive_end_index(j, W, ksize[2])
                else:

                    d_start = k * strides[0] - pad_d_forth
                    d_end = np.min((k * strides[0] + ksize[0] - pad_d_forth,
                                    D + pad_d_back))
                    h_start = i * strides[1] - pad_h_up
                    h_end = np.min(
                        (i * strides[1] + ksize[1] - pad_h_up, H + pad_h_down))
                    w_start = j * strides[2] - pad_w_left
                    w_end = np.min((j * strides[2] + ksize[2] - pad_w_left,
                                    W + pad_w_right))

                    field_size = (d_end - d_start) * (h_end - h_start) * (
                        w_end - w_start)
                    w_start = np.max((w_start, 0))
                    d_start = np.max((d_start, 0))
                    h_start = np.max((h_start, 0))
                    w_end = np.min((w_end, W))
                    d_end = np.min((d_end, D))
                    h_end = np.min((h_end, H))
                if data_format == 'NCDHW':
                    x_masked = x[:, :, d_start:d_end, h_start:h_end, w_start:
                                 w_end]
                    if pool_type == 'avg':
                        if (exclusive or adaptive):
                            field_size = (d_end - d_start) * (
                                h_end - h_start) * (w_end - w_start)

                        out[:, :, k, i, j] = np.sum(x_masked,
                                                    axis=(2, 3, 4)) / field_size
                    elif pool_type == 'max':
                        out[:, :, k, i, j] = np.max(x_masked, axis=(2, 3, 4))

                elif data_format == 'NDHWC':
                    x_masked = x[:, d_start:d_end, h_start:h_end, w_start:
                                 w_end, :]
                    if pool_type == 'avg':
                        if (exclusive or adaptive):
                            field_size = (d_end - d_start) * (
                                h_end - h_start) * (w_end - w_start)

                        out[:, k, i, j, :] = np.sum(x_masked,
                                                    axis=(1, 2, 3)) / field_size
                    elif pool_type == 'max':
                        out[:, k, i, j, :] = np.max(x_masked, axis=(1, 2, 3))

    return out


def max_pool3D_forward_naive(x,
                             ksize,
                             strides,
                             paddings,
                             global_pool=0,
                             ceil_mode=False,
                             exclusive=True,
                             adaptive=False):
    out = pool3D_forward_naive(
        x=x,
        ksize=ksize,
        strides=strides,
        paddings=paddings,
        global_pool=global_pool,
        ceil_mode=ceil_mode,
        exclusive=exclusive,
        adaptive=adaptive,
        data_format='NCDHW',
        pool_type="max")
    return out


def avg_pool3D_forward_naive(x,
                             ksize,
                             strides,
                             paddings,
                             global_pool=0,
                             ceil_mode=False,
                             exclusive=True,
                             adaptive=False):
    out = pool3D_forward_naive(
        x=x,
        ksize=ksize,
        strides=strides,
        paddings=paddings,
        global_pool=global_pool,
        ceil_mode=ceil_mode,
        exclusive=exclusive,
        adaptive=adaptive,
        data_format='NCDHW',
        pool_type="avg")
    return out


class TestPool3D_Op(OpTest):
    def setUp(self):
        self.op_type = "pool3d"
        self.init_kernel_type()
        self.dtype = np.float64
        self.init_test_case()
        self.padding_algorithm = "EXPLICIT"
        self.init_paddings()
        self.init_global_pool()
        self.init_kernel_type()
        self.init_pool_type()
        self.init_ceil_mode()
        self.init_exclusive()
        self.init_adaptive()
        self.init_data_format()
        self.init_shape()
        paddle.enable_static()

        input = np.random.random(self.shape).astype(self.dtype)
        output = pool3D_forward_naive(
            input, self.ksize, self.strides, self.paddings, self.global_pool,
            self.ceil_mode, self.exclusive, self.adaptive, self.data_format,
            self.pool_type, self.padding_algorithm).astype(self.dtype)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(input)}

        self.attrs = {
            'strides': self.strides,
            'paddings': self.paddings,
            'ksize': self.ksize,
            'pooling_type': self.pool_type,
            'global_pooling': self.global_pool,
            'use_cudnn': self.use_cudnn,
            'ceil_mode': self.ceil_mode,
            'data_format': self.data_format,
            'exclusive': self.exclusive,
            'adaptive': self.adaptive,
            "padding_algorithm": self.padding_algorithm,
        }

        self.outputs = {'Out': output}

    def has_cudnn(self):
        return core.is_compiled_with_cuda() and self.use_cudnn

    def test_check_output(self):
        if self.has_cudnn():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-5)
        else:
            self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        if self.has_cudnn() and self.pool_type != "max":
            place = core.CUDAPlace(0)
            self.check_grad_with_place(place, set(['X']), 'Out')
        elif self.pool_type != "max":
            self.check_grad(set(['X']), 'Out')

    def init_data_format(self):
        self.data_format = "NCDHW"

    def init_shape(self):
        self.shape = [1, 3, 5, 6, 5]

    def init_test_case(self):
        self.ksize = [2, 3, 1]
        self.strides = [2, 2, 3]

    def init_paddings(self):
        self.paddings = [0, 0, 0]
        self.padding_algorithm = "EXPLICIT"

    def init_kernel_type(self):
        self.use_cudnn = False

    def init_pool_type(self):
        self.pool_type = "avg"

    def init_global_pool(self):
        self.global_pool = True

    def init_ceil_mode(self):
        self.ceil_mode = False

    def init_exclusive(self):
        self.exclusive = True

    def init_adaptive(self):
        self.adaptive = False


class TestCase1(TestPool3D_Op):
    def init_shape(self):
        self.shape = [1, 3, 7, 7, 7]

    def init_test_case(self):
        self.ksize = [3, 3, 3]
        self.strides = [1, 1, 1]

    def init_paddings(self):
        self.paddings = [0, 0, 0]

    def init_pool_type(self):
        self.pool_type = "avg"

    def init_global_pool(self):
        self.global_pool = False


class TestCase2(TestPool3D_Op):
    def init_shape(self):
        self.shape = [1, 3, 6, 7, 7]

    def init_test_case(self):
        self.ksize = [3, 3, 4]
        self.strides = [1, 3, 2]

    def init_paddings(self):
        self.paddings = [1, 1, 1]

    def init_pool_type(self):
        self.pool_type = "avg"

    def init_global_pool(self):
        self.global_pool = False


class TestCase3(TestPool3D_Op):
    def init_pool_type(self):
        self.pool_type = "max"


class TestCase4(TestCase1):
    def init_pool_type(self):
        self.pool_type = "max"


class TestCase5(TestCase2):
    def init_pool_type(self):
        self.pool_type = "max"


#--------------------test pool3d cudnn--------------------


def create_test_cudnn_class(parent):
    @unittest.skipIf(not core.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    class TestCUDNNCase(parent):
        def init_kernel_type(self):
            self.use_cudnn = True

    cls_name = "{0}_{1}".format(parent.__name__, "CUDNNOp")
    TestCUDNNCase.__name__ = cls_name
    globals()[cls_name] = TestCUDNNCase


create_test_cudnn_class(TestPool3D_Op)
create_test_cudnn_class(TestCase1)
create_test_cudnn_class(TestCase2)
create_test_cudnn_class(TestCase3)
create_test_cudnn_class(TestCase4)
create_test_cudnn_class(TestCase5)


def create_test_cudnn_fp16_class(parent):
    @unittest.skipIf(not core.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    class TestCUDNNFp16Case(parent):
        def init_kernel_type(self):
            self.use_cudnn = True
            self.dtype = np.float16

        def test_check_output(self):
            if core.is_compiled_with_cuda():
                place = core.CUDAPlace(0)
                if core.is_float16_supported(place):
                    self.check_output_with_place(place, atol=1e-3)

    cls_name = "{0}_{1}".format(parent.__name__, "CUDNNFp16Op")
    TestCUDNNFp16Case.__name__ = cls_name
    globals()[cls_name] = TestCUDNNFp16Case


def create_test_fp16_class(parent):
    @unittest.skipIf(not core.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    class TestFp16Case(parent):
        def init_kernel_type(self):
            self.use_cudnn = False
            self.dtype = np.float16

        def test_check_output(self):
            if core.is_compiled_with_cuda():
                place = core.CUDAPlace(0)
                if core.is_float16_supported(place):
                    self.check_output_with_place(place, atol=1e-2)

    cls_name = "{0}_{1}".format(parent.__name__, "Fp16Op")
    TestFp16Case.__name__ = cls_name
    globals()[cls_name] = TestFp16Case


create_test_cudnn_fp16_class(TestPool3D_Op)
create_test_cudnn_fp16_class(TestCase1)
create_test_cudnn_fp16_class(TestCase2)
create_test_cudnn_fp16_class(TestCase3)
create_test_cudnn_fp16_class(TestCase4)
create_test_cudnn_fp16_class(TestCase5)

create_test_fp16_class(TestPool3D_Op)
create_test_fp16_class(TestCase1)
create_test_fp16_class(TestCase2)
create_test_fp16_class(TestCase3)
create_test_fp16_class(TestCase4)
create_test_fp16_class(TestCase5)


# ---- test ceil mode ------
def create_test_cudnn_use_ceil_class(parent):
    @unittest.skipIf(not core.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    class TestPool3DUseCeilCase(parent):
        def init_kernel_type(self):
            self.use_cudnn = True

        def init_ceil_mode(self):
            self.ceil_mode = True

    cls_name = "{0}_{1}".format(parent.__name__, "CUDNNOpCeilMode")
    TestPool3DUseCeilCase.__name__ = cls_name
    globals()[cls_name] = TestPool3DUseCeilCase


create_test_cudnn_use_ceil_class(TestPool3D_Op)
create_test_cudnn_use_ceil_class(TestCase1)


def create_test_use_ceil_class(parent):
    class TestPool3DUseCeilCase(parent):
        def init_ceil_mode(self):
            self.ceil_mode = True

    cls_name = "{0}_{1}".format(parent.__name__, "CeilModeCast")
    TestPool3DUseCeilCase.__name__ = cls_name
    globals()[cls_name] = TestPool3DUseCeilCase


create_test_use_ceil_class(TestCase1)
create_test_use_ceil_class(TestCase2)


class TestAvgInclude(TestCase2):
    def init_exclusive(self):
        self.exclusive = False


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestCUDNNAvgInclude(TestCase2):
    def init_kernel_type(self):
        self.use_cudnn = True

    def init_exclusive(self):
        self.exclusive = False


class TestAvgPoolAdaptive(TestCase1):
    def init_adaptive(self):
        self.adaptive = True


class TestAvgPoolAdaptiveAsyOutSize(TestCase1):
    def init_adaptive(self):
        self.adaptive = True

    def init_shape(self):
        self.shape = [1, 3, 3, 4, 4]

    def init_test_case(self):
        self.ksize = [2, 2, 3]
        self.strides = [1, 1, 1]


#-------test pool3d with asymmetric padding------
class TestPool3D_Op_AsyPadding(TestPool3D_Op):
    def init_test_case(self):
        self.ksize = [3, 4, 3]
        self.strides = [1, 1, 2]

    def init_paddings(self):
        self.paddings = [0, 0, 0, 2, 3, 0]

    def init_shape(self):
        self.shape = [1, 3, 5, 5, 6]


class TestCase1_AsyPadding(TestCase1):
    def init_test_case(self):
        self.ksize = [3, 3, 4]
        self.strides = [1, 1, 2]

    def init_paddings(self):
        self.paddings = [1, 0, 2, 1, 2, 1]

    def init_shape(self):
        self.shape = [1, 3, 7, 7, 6]


class TestCase2_AsyPadding(TestCase2):
    def init_test_case(self):
        self.ksize = [3, 3, 3]
        self.strides = [1, 1, 1]

    def init_paddings(self):
        self.paddings = [1, 2, 1, 1, 1, 0]

    def init_shape(self):
        self.shape = [1, 3, 7, 7, 7]


class TestCase3_AsyPadding(TestCase3):
    def init_test_case(self):
        self.ksize = [3, 3, 3]
        self.strides = [1, 1, 1]

    def init_paddings(self):
        self.paddings = [1, 0, 0, 0, 1, 0]

    def init_shape(self):
        self.shape = [1, 3, 5, 5, 5]


class TestCase4_AsyPadding(TestCase4):
    def init_test_case(self):
        self.ksize = [3, 3, 3]
        self.strides = [1, 1, 1]

    def init_paddings(self):
        self.paddings = [1, 0, 2, 1, 2, 1]

    def init_shape(self):
        self.shape = [1, 3, 7, 7, 7]


class TestCase5_AsyPadding(TestCase5):
    def init_test_case(self):
        self.ksize = [3, 3, 3]
        self.strides = [1, 1, 1]

    def init_paddings(self):
        self.paddings = [1, 2, 1, 1, 1, 0]

    def init_shape(self):
        self.shape = [1, 3, 7, 7, 7]


create_test_cudnn_class(TestPool3D_Op_AsyPadding)
create_test_cudnn_class(TestCase1_AsyPadding)
create_test_cudnn_class(TestCase2_AsyPadding)
create_test_cudnn_class(TestCase3_AsyPadding)
create_test_cudnn_class(TestCase4_AsyPadding)
create_test_cudnn_class(TestCase5_AsyPadding)

create_test_cudnn_fp16_class(TestPool3D_Op_AsyPadding)
create_test_cudnn_fp16_class(TestCase1_AsyPadding)
create_test_cudnn_fp16_class(TestCase2_AsyPadding)
create_test_cudnn_fp16_class(TestCase3_AsyPadding)
create_test_cudnn_fp16_class(TestCase4_AsyPadding)
create_test_cudnn_fp16_class(TestCase5_AsyPadding)

create_test_cudnn_use_ceil_class(TestPool3D_Op_AsyPadding)
create_test_cudnn_use_ceil_class(TestCase1_AsyPadding)

create_test_use_ceil_class(TestCase1_AsyPadding)
create_test_use_ceil_class(TestCase2_AsyPadding)


class TestAvgInclude_AsyPadding(TestCase2):
    def init_exclusive(self):
        self.exclusive = False

    def init_paddings(self):
        self.paddings = [2, 2, 1, 1, 0, 0]


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestCUDNNAvgInclude_AsyPadding(TestCase2):
    def init_kernel_type(self):
        self.use_cudnn = True

    def init_exclusive(self):
        self.exclusive = False

    def init_paddings(self):
        self.paddings = [1, 0, 0, 0, 0, 0]

    def init_shape(self):
        self.shape = [1, 3, 5, 5, 5]


class TestAvgPoolAdaptive_AsyPadding(TestCase1):
    def init_adaptive(self):
        self.adaptive = True

    def init_paddings(self):
        self.paddings = [1, 0, 2, 1, 2, 1]


# ------------ test channel_last --------------
class TestPool3D_channel_last(TestPool3D_Op):
    def init_data_format(self):
        self.data_format = "NDHWC"

    def init_shape(self):
        self.shape = [1, 5, 5, 6, 3]


class TestCase1_channel_last(TestCase1):
    def init_data_format(self):
        self.data_format = "NDHWC"

    def init_shape(self):
        self.shape = [1, 7, 7, 7, 3]


class TestCase2_channel_last(TestCase2):
    def init_data_format(self):
        self.data_format = "NDHWC"

    def init_shape(self):
        self.shape = [1, 7, 7, 5, 3]


class TestCase3_channel_last(TestCase3):
    def init_data_format(self):
        self.data_format = "NDHWC"

    def init_shape(self):
        self.shape = [1, 5, 6, 5, 3]


class TestCase4_channel_last(TestCase4):
    def init_data_format(self):
        self.data_format = "NDHWC"

    def init_shape(self):
        self.shape = [1, 7, 6, 7, 3]


class TestCase5_channel_last(TestCase5):
    def init_data_format(self):
        self.data_format = "NDHWC"

    def init_shape(self):
        self.shape = [1, 7, 7, 7, 3]


create_test_cudnn_class(TestPool3D_channel_last)
create_test_cudnn_class(TestCase1_channel_last)
create_test_cudnn_class(TestCase2_channel_last)
create_test_cudnn_class(TestCase3_channel_last)
create_test_cudnn_class(TestCase4_channel_last)
create_test_cudnn_class(TestCase5_channel_last)

create_test_cudnn_use_ceil_class(TestPool3D_channel_last)
create_test_cudnn_use_ceil_class(TestCase1_channel_last)

create_test_use_ceil_class(TestCase1_channel_last)
create_test_use_ceil_class(TestCase2_channel_last)


class TestCase5_Max(TestCase2):
    def init_pool_type(self):
        self.pool_type = "max"

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        if self.has_cudnn() and self.pool_type == "max":
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place, set(['X']), 'Out', max_relative_error=1.00)
        elif self.pool_type == "max":
            self.check_grad(set(['X']), 'Out', max_relative_error=1.00)


class TestCase5_channel_last_Max(TestCase5_Max):
    def init_data_format(self):
        self.data_format = "NDHWC"

    def init_shape(self):
        self.shape = [1, 7, 7, 7, 3]


create_test_cudnn_class(TestCase5_Max)
create_test_cudnn_class(TestCase5_channel_last_Max)


class TestAvgInclude_channel_last(TestCase2_channel_last):
    def init_exclusive(self):
        self.exclusive = False


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestCUDNNAvgInclude_channel_last(TestCase2_channel_last):
    def init_kernel_type(self):
        self.use_cudnn = True

    def init_exclusive(self):
        self.exclusive = False


class TestAvgPoolAdaptive_channel_last(TestCase1_channel_last):
    def init_adaptive(self):
        self.adaptive = True


# --- asy padding
class TestPool3D_Op_AsyPadding_channel_last(TestPool3D_Op_AsyPadding):
    def init_data_format(self):
        self.data_format = "NDHWC"

    def init_shape(self):
        self.shape = [1, 5, 5, 6, 3]


class TestCase1_AsyPadding_channel_last(TestCase1_AsyPadding):
    def init_data_format(self):
        self.data_format = "NDHWC"

    def init_shape(self):
        self.shape = [1, 7, 6, 8, 3]


class TestCase2_AsyPadding_channel_last(TestCase2_AsyPadding):
    def init_data_format(self):
        self.data_format = "NDHWC"

    def init_shape(self):
        self.shape = [1, 6, 8, 7, 3]


class TestCase3_AsyPadding_channel_last(TestCase3_AsyPadding):
    def init_data_format(self):
        self.data_format = "NDHWC"

    def init_shape(self):
        self.shape = [1, 5, 7, 5, 3]


class TestCase4_AsyPadding_channel_last(TestCase4_AsyPadding):
    def init_data_format(self):
        self.data_format = "NDHWC"

    def init_shape(self):
        self.shape = [1, 6, 7, 7, 3]


class TestCase5_AsyPadding_channel_last(TestCase5_AsyPadding):
    def init_data_format(self):
        self.data_format = "NDHWC"

    def init_shape(self):
        self.shape = [1, 7, 8, 6, 3]


create_test_cudnn_class(TestPool3D_Op_AsyPadding_channel_last)
create_test_cudnn_class(TestCase1_AsyPadding_channel_last)
create_test_cudnn_class(TestCase2_AsyPadding_channel_last)
create_test_cudnn_class(TestCase3_AsyPadding_channel_last)
create_test_cudnn_class(TestCase4_AsyPadding_channel_last)
create_test_cudnn_class(TestCase5_AsyPadding_channel_last)

create_test_cudnn_use_ceil_class(TestPool3D_Op_AsyPadding_channel_last)
create_test_cudnn_use_ceil_class(TestCase1_AsyPadding_channel_last)

create_test_use_ceil_class(TestCase1_AsyPadding_channel_last)
create_test_use_ceil_class(TestCase2_AsyPadding_channel_last)


class TestAvgInclude_AsyPadding_channel_last(TestAvgInclude_AsyPadding):
    def init_data_format(self):
        self.data_format = "NDHWC"


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestCUDNNAvgInclude_AsyPadding_channel_last(
        TestCUDNNAvgInclude_AsyPadding):
    def init_data_format(self):
        self.data_format = "NDHWC"


class TestAvgPoolAdaptive_AsyPadding_channel_last(
        TestAvgPoolAdaptive_AsyPadding):
    def init_data_format(self):
        self.data_format = "NDHWC"

    def init_shape(self):
        self.shape = [1, 7, 7, 7, 3]


#test padding = SAME VALID
def create_test_padding_SAME_class(parent):
    class TestPaddingSMAECase(parent):
        def init_paddings(self):
            self.paddings = [0, 0, 0]
            self.padding_algorithm = "SAME"

    cls_name = "{0}_{1}".format(parent.__name__, "PaddingSAMEOp")
    TestPaddingSMAECase.__name__ = cls_name
    globals()[cls_name] = TestPaddingSMAECase


create_test_padding_SAME_class(TestPool3D_Op)
create_test_padding_SAME_class(TestCase1)
create_test_padding_SAME_class(TestCase2)
create_test_padding_SAME_class(TestCase3)
create_test_padding_SAME_class(TestCase4)
create_test_padding_SAME_class(TestCase5)

create_test_padding_SAME_class(TestPool3D_channel_last)
create_test_padding_SAME_class(TestCase1_channel_last)
create_test_padding_SAME_class(TestCase2_channel_last)
create_test_padding_SAME_class(TestCase3_channel_last)
create_test_padding_SAME_class(TestCase4_channel_last)
create_test_padding_SAME_class(TestCase5_channel_last)


def create_test_cudnn_padding_SAME_class(parent):
    @unittest.skipIf(not core.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    class TestCUDNNPaddingSMAECase(parent):
        def init_kernel_type(self):
            self.use_cudnn = True

        def init_paddings(self):
            self.paddings = [1, 1, 1]
            self.padding_algorithm = "SAME"

    cls_name = "{0}_{1}".format(parent.__name__, "CudnnPaddingSAMEOp")
    TestCUDNNPaddingSMAECase.__name__ = cls_name
    globals()[cls_name] = TestCUDNNPaddingSMAECase


create_test_cudnn_padding_SAME_class(TestPool3D_Op)
create_test_cudnn_padding_SAME_class(TestCase1)
create_test_cudnn_padding_SAME_class(TestCase2)
create_test_cudnn_padding_SAME_class(TestCase3)
create_test_cudnn_padding_SAME_class(TestCase4)
create_test_cudnn_padding_SAME_class(TestCase5)

create_test_cudnn_padding_SAME_class(TestPool3D_channel_last)
create_test_cudnn_padding_SAME_class(TestCase1_channel_last)
create_test_cudnn_padding_SAME_class(TestCase2_channel_last)
create_test_cudnn_padding_SAME_class(TestCase3_channel_last)
create_test_cudnn_padding_SAME_class(TestCase4_channel_last)
create_test_cudnn_padding_SAME_class(TestCase5_channel_last)


def create_test_padding_VALID_class(parent):
    class TestPaddingVALIDCase(parent):
        def init_paddings(self):
            self.paddings = [1, 1, 1]
            self.padding_algorithm = "VALID"

    cls_name = "{0}_{1}".format(parent.__name__, "PaddingVALIDOp")
    TestPaddingVALIDCase.__name__ = cls_name
    globals()[cls_name] = TestPaddingVALIDCase


create_test_padding_VALID_class(TestPool3D_Op)
create_test_padding_VALID_class(TestCase1)
create_test_padding_VALID_class(TestCase2)
create_test_padding_VALID_class(TestCase3)
create_test_padding_VALID_class(TestCase4)
create_test_padding_VALID_class(TestCase5)

create_test_padding_VALID_class(TestPool3D_channel_last)
create_test_padding_VALID_class(TestCase1_channel_last)
create_test_padding_VALID_class(TestCase2_channel_last)
create_test_padding_VALID_class(TestCase3_channel_last)
create_test_padding_VALID_class(TestCase4_channel_last)
create_test_padding_VALID_class(TestCase5_channel_last)


def create_test_cudnn_padding_VALID_class(parent):
    @unittest.skipIf(not core.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    class TestCUDNNPaddingVALIDCase(parent):
        def init_kernel_type(self):
            self.use_cudnn = True

        def init_paddings(self):
            self.paddings = [1, 1, 1]
            self.padding_algorithm = "VALID"

    cls_name = "{0}_{1}".format(parent.__name__, "CudnnPaddingVALIDOp")
    TestCUDNNPaddingVALIDCase.__name__ = cls_name
    globals()[cls_name] = TestCUDNNPaddingVALIDCase


create_test_cudnn_padding_VALID_class(TestPool3D_Op)
create_test_cudnn_padding_VALID_class(TestCase1)
create_test_cudnn_padding_VALID_class(TestCase2)
create_test_cudnn_padding_VALID_class(TestCase3)
create_test_cudnn_padding_VALID_class(TestCase4)
create_test_cudnn_padding_VALID_class(TestCase5)

create_test_cudnn_padding_VALID_class(TestPool3D_channel_last)
create_test_cudnn_padding_VALID_class(TestCase1_channel_last)
create_test_cudnn_padding_VALID_class(TestCase2_channel_last)
create_test_cudnn_padding_VALID_class(TestCase3_channel_last)
create_test_cudnn_padding_VALID_class(TestCase4_channel_last)
create_test_cudnn_padding_VALID_class(TestCase5_channel_last)


#test API
class TestPool3DAPI(unittest.TestCase):
    def test_api(self):
        x_NDHWC = np.random.random([2, 5, 5, 5, 3]).astype("float32")
        x_NCDHW = np.random.random([2, 3, 5, 5, 5]).astype("float32")

        input_NDHWC = fluid.layers.data(
            name="input_NDHWC",
            shape=[2, 5, 5, 5, 3],
            append_batch_size=False,
            dtype="float32")

        input_NCDHW = fluid.layers.data(
            name="input_NCDHW",
            shape=[2, 3, 5, 5, 5],
            append_batch_size=False,
            dtype="float32")

        ksize = [3, 3, 3]
        out_1 = fluid.layers.pool3d(
            input=input_NDHWC,
            pool_size=ksize,
            pool_type="max",
            pool_padding=[1, 1, 1],
            use_cudnn=False,
            data_format="NDHWC")

        out_2 = fluid.layers.pool3d(
            input=input_NDHWC,
            pool_size=ksize,
            pool_type="avg",
            pool_padding=[[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]],
            use_cudnn=False,
            data_format="NDHWC")

        out_3 = fluid.layers.pool3d(
            input=input_NCDHW,
            pool_size=ksize,
            pool_type="avg",
            pool_padding=[[0, 0], [0, 0], [1, 1], [1, 1], [1, 1]],
            use_cudnn=False,
            data_format="NCDHW")

        out_4 = fluid.layers.pool3d(
            input=input_NCDHW,
            pool_size=ksize,
            pool_type="avg",
            pool_padding=[1, 2, 1, 0, 0, 1],
            use_cudnn=False,
            data_format="NCDHW")
        # test VALID
        out_5 = fluid.layers.pool3d(
            input=input_NDHWC,
            pool_size=ksize,
            pool_type="avg",
            pool_padding="VALID",
            use_cudnn=False,
            data_format="NDHWC")

        out_6 = fluid.layers.pool3d(
            input=input_NCDHW,
            pool_size=ksize,
            pool_type="avg",
            pool_padding="VALID",
            use_cudnn=False,
            data_format="NCDHW")

        # test SAME
        out_7 = fluid.layers.pool3d(
            input=input_NDHWC,
            pool_size=ksize,
            pool_stride=[1, 1, 2],
            pool_type="avg",
            pool_padding="SAME",
            use_cudnn=False,
            data_format="NDHWC")

        out_8 = fluid.layers.pool3d(
            input=input_NCDHW,
            pool_size=[4, 4, 4],
            pool_type="avg",
            pool_padding="SAME",
            use_cudnn=False,
            data_format="NCDHW")

        exe = fluid.Executor(place=fluid.CPUPlace())
        [res_1, res_2, res_3, res_4, res_5, res_6, res_7, res_8] = exe.run(
            fluid.default_main_program(),
            feed={"input_NDHWC": x_NDHWC,
                  "input_NCDHW": x_NCDHW},
            fetch_list=[
                out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8
            ])

        assert np.allclose(
            res_1,
            pool3D_forward_naive(
                x=x_NDHWC,
                ksize=ksize,
                pool_type="max",
                strides=[1, 1, 1],
                paddings=[1, 1, 1],
                data_format="NDHWC"))

        assert np.allclose(
            res_2,
            pool3D_forward_naive(
                x=x_NDHWC,
                ksize=ksize,
                pool_type="avg",
                strides=[1, 1, 1],
                paddings=[1, 1, 1, 1, 1, 1],
                data_format="NDHWC"))
        assert np.allclose(
            res_3,
            pool3D_forward_naive(
                x=x_NCDHW,
                ksize=ksize,
                pool_type="avg",
                strides=[1, 1, 1],
                paddings=[1, 1, 1, 1, 1, 1],
                data_format="NCDHW"),
            rtol=0.07,
            atol=1e-05)

        assert np.allclose(
            res_4,
            pool3D_forward_naive(
                x=x_NCDHW,
                ksize=ksize,
                pool_type="avg",
                strides=[1, 1, 1],
                paddings=[1, 2, 1, 0, 0, 1],
                data_format="NCDHW"),
            rtol=0.07,
            atol=1e-05)
        # VALID
        assert np.allclose(
            res_5,
            pool3D_forward_naive(
                x=x_NDHWC,
                ksize=ksize,
                pool_type="avg",
                strides=[1, 1, 1],
                paddings=[10, 20],
                padding_algorithm="VALID",
                data_format="NDHWC"))

        assert np.allclose(
            res_6,
            pool3D_forward_naive(
                x=x_NCDHW,
                ksize=ksize,
                pool_type="avg",
                strides=[1, 1, 1],
                paddings=[10, 20],
                padding_algorithm="VALID",
                data_format="NCDHW"),
            rtol=0.07,
            atol=1e-05)
        # SAME
        assert np.allclose(
            res_7,
            pool3D_forward_naive(
                x=x_NDHWC,
                ksize=ksize,
                pool_type="avg",
                strides=[1, 1, 2],
                paddings=[10, 20],
                padding_algorithm="SAME",
                data_format="NDHWC"))

        assert np.allclose(
            res_8,
            pool3D_forward_naive(
                x=x_NCDHW,
                ksize=[4, 4, 4],
                pool_type="avg",
                strides=[1, 1, 1],
                paddings=[10, 20],
                padding_algorithm="SAME",
                data_format="NCDHW"),
            rtol=0.07,
            atol=1e-05)


class TestPool3DAPI_Error(unittest.TestCase):
    def test_api(self):
        input_NDHWC = fluid.layers.data(
            name="input_NDHWC",
            shape=[2, 5, 5, 5, 3],
            append_batch_size=False,
            dtype="float32")
        ksize = [3, 3, 3]

        # cudnn type error
        def run_1():
            out_1 = fluid.layers.pool3d(
                input=input_NDHWC,
                pool_size=ksize,
                pool_type="max",
                pool_padding=[1, 1, 1],
                use_cudnn=[0],
                data_format="NDHWC")

        self.assertRaises(TypeError, run_1)

        # data_format value error
        def run_2():
            out_2 = fluid.layers.pool3d(
                input=input_NDHWC,
                pool_size=ksize,
                pool_type="max",
                pool_padding=[1, 1, 1],
                use_cudnn=False,
                data_format="NDHWCC")

        self.assertRaises(ValueError, run_2)

        # padding str value error
        def run_3():
            out_3 = fluid.layers.pool3d(
                input=input_NDHWC,
                pool_size=ksize,
                pool_type="max",
                pool_padding="VALIDSAME",
                use_cudnn=False,
                data_format="NDHWC")

        self.assertRaises(ValueError, run_3)

        # padding str valid and ceil_mode value error
        def run_4():
            out_4 = fluid.layers.pool3d(
                input=input_NDHWC,
                pool_size=ksize,
                pool_type="max",
                pool_padding="VALID",
                use_cudnn=False,
                ceil_mode=True,
                data_format="NDHWC")

        self.assertRaises(ValueError, run_4)

        # padding with 8 ele. value error
        def run_5():
            out_5 = fluid.layers.pool3d(
                input=input_NDHWC,
                pool_size=ksize,
                pool_type="max",
                pool_padding=[[1, 1], [0, 0], [0, 0], [1, 1], [1, 1]],
                use_cudnn=False,
                data_format="NDHWC")

        self.assertRaises(ValueError, run_5)


if __name__ == '__main__':
    unittest.main()
