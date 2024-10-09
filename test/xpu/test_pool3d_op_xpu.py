#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


def adaptive_start_index(index, input_size, output_size):
    return int(np.floor(index * input_size / output_size))


def adaptive_end_index(index, input_size, output_size):
    return int(np.ceil((index + 1) * input_size / output_size))


def pool3D_forward_naive(
    x,
    ksize,
    strides,
    paddings,
    global_pool=0,
    ceil_mode=False,
    exclusive=True,
    adaptive=False,
    data_format='NCDHW',
    pool_type='max',
    padding_algorithm="EXPLICIT",
):
    # update paddings
    def _get_padding_with_SAME(input_shape, pool_size, pool_stride):
        padding = []
        for input_size, filter_size, stride_size in zip(
            input_shape, pool_size, pool_stride
        ):
            out_size = int((input_size + stride_size - 1) / stride_size)
            pad_sum = np.max(
                ((out_size - 1) * stride_size + filter_size - input_size, 0)
            )
            pad_0 = int(pad_sum / 2)
            pad_1 = int(pad_sum - pad_0)
            padding.append(pad_0)
            padding.append(pad_1)
        return padding

    if isinstance(padding_algorithm, str):
        padding_algorithm = padding_algorithm.upper()
        if padding_algorithm not in ["SAME", "VALID", "EXPLICIT"]:
            raise ValueError(
                f"Unknown Attr(padding_algorithm): '{padding_algorithm}'. "
                "It can only be 'SAME' or 'VALID'."
            )

        if padding_algorithm == "VALID":
            paddings = [0, 0, 0, 0, 0, 0]
            if ceil_mode is not False:
                raise ValueError(
                    'When Attr(pool_padding) is "VALID", Attr(ceil_mode)'
                    " must be False. "
                    "Received ceil_mode: True."
                )
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
    C, D, H, W = (
        [x.shape[1], x.shape[2], x.shape[3], x.shape[4]]
        if data_format == 'NCDHW'
        else [x.shape[4], x.shape[1], x.shape[2], x.shape[3]]
    )

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
        D_out = (
            (D - ksize[0] + pad_d_forth + pad_d_back + strides[0] - 1)
            // strides[0]
            + 1
            if ceil_mode
            else (D - ksize[0] + pad_d_forth + pad_d_back) // strides[0] + 1
        )

        H_out = (
            (H - ksize[1] + pad_h_up + pad_h_down + strides[1] - 1)
            // strides[1]
            + 1
            if ceil_mode
            else (H - ksize[1] + pad_h_up + pad_h_down) // strides[1] + 1
        )

        W_out = (
            (W - ksize[2] + pad_w_left + pad_w_right + strides[2] - 1)
            // strides[2]
            + 1
            if ceil_mode
            else (W - ksize[2] + pad_w_left + pad_w_right) // strides[2] + 1
        )

    out = (
        np.zeros((N, C, D_out, H_out, W_out))
        if data_format == 'NCDHW'
        else np.zeros((N, D_out, H_out, W_out, C))
    )
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
                    d_end = np.min(
                        (
                            k * strides[0] + ksize[0] - pad_d_forth,
                            D + pad_d_back,
                        )
                    )
                    h_start = i * strides[1] - pad_h_up
                    h_end = np.min(
                        (i * strides[1] + ksize[1] - pad_h_up, H + pad_h_down)
                    )
                    w_start = j * strides[2] - pad_w_left
                    w_end = np.min(
                        (
                            j * strides[2] + ksize[2] - pad_w_left,
                            W + pad_w_right,
                        )
                    )

                    field_size = (
                        (d_end - d_start)
                        * (h_end - h_start)
                        * (w_end - w_start)
                    )
                    w_start = np.max((w_start, 0))
                    d_start = np.max((d_start, 0))
                    h_start = np.max((h_start, 0))
                    w_end = np.min((w_end, W))
                    d_end = np.min((d_end, D))
                    h_end = np.min((h_end, H))
                if data_format == 'NCDHW':
                    x_masked = x[
                        :, :, d_start:d_end, h_start:h_end, w_start:w_end
                    ]
                    if pool_type == 'avg':
                        if exclusive or adaptive:
                            field_size = (
                                (d_end - d_start)
                                * (h_end - h_start)
                                * (w_end - w_start)
                            )

                        out[:, :, k, i, j] = (
                            np.sum(x_masked, axis=(2, 3, 4)) / field_size
                        )
                    elif pool_type == 'max':
                        out[:, :, k, i, j] = np.max(x_masked, axis=(2, 3, 4))

                elif data_format == 'NDHWC':
                    x_masked = x[
                        :, d_start:d_end, h_start:h_end, w_start:w_end, :
                    ]
                    if pool_type == 'avg':
                        if exclusive or adaptive:
                            field_size = (
                                (d_end - d_start)
                                * (h_end - h_start)
                                * (w_end - w_start)
                            )

                        out[:, k, i, j, :] = (
                            np.sum(x_masked, axis=(1, 2, 3)) / field_size
                        )
                    elif pool_type == 'max':
                        out[:, k, i, j, :] = np.max(x_masked, axis=(1, 2, 3))

    return out


def max_pool3D_forward_naive(
    x,
    ksize,
    strides,
    paddings,
    global_pool=0,
    ceil_mode=False,
    exclusive=True,
    adaptive=False,
):
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
        pool_type="max",
    )
    return out


def avg_pool3D_forward_naive(
    x,
    ksize,
    strides,
    paddings,
    global_pool=0,
    ceil_mode=False,
    exclusive=True,
    adaptive=False,
):
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
        pool_type="avg",
    )
    return out


class XPUTestPool3DOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'pool3d'
        self.use_dynamic_create_class = False

    class TestPool3D_Op(XPUOpTest):
        def setUp(self):
            self.op_type = "pool3d"
            self.init_kernel_type()
            self.dtype = self.in_type
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
                input,
                self.ksize,
                self.strides,
                self.paddings,
                self.global_pool,
                self.ceil_mode,
                self.exclusive,
                self.adaptive,
                self.data_format,
                self.pool_type,
                self.padding_algorithm,
            ).astype(self.dtype)

            self.inputs = {'X': XPUOpTest.np_dtype_to_base_dtype(input)}

            self.attrs = {
                'strides': self.strides,
                'paddings': self.paddings,
                'ksize': self.ksize,
                'pooling_type': self.pool_type,
                'global_pooling': self.global_pool,
                'ceil_mode': self.ceil_mode,
                'data_format': self.data_format,
                'exclusive': self.exclusive,
                'adaptive': self.adaptive,
                "padding_algorithm": self.padding_algorithm,
            }

            self.outputs = {'Out': output}

        def test_check_output(self):
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place)

        def test_check_grad(self):
            if self.dtype == np.float16:
                return

            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, {'X'}, 'Out')

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

    class TestAvgInclude(TestCase2):
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

    # -------test pool3d with asymmetric padding------
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

    class TestAvgInclude_AsyPadding(TestCase2):
        def init_exclusive(self):
            self.exclusive = False

        def init_paddings(self):
            self.paddings = [2, 2, 1, 1, 0, 0]

    class TestAvgPoolAdaptive_AsyPadding(TestCase1):
        def init_adaptive(self):
            self.adaptive = True

        def init_paddings(self):
            self.paddings = [1, 0, 2, 1, 2, 1]

    class TestCase5_Max(TestCase2):
        def init_pool_type(self):
            self.pool_type = "max"

        def test_check_grad(self):
            if self.dtype == np.float16:
                return
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, {'X'}, 'Out')


support_types = get_xpu_op_support_types('pool3d')
for stype in ["float32"]:
    create_test_class(globals(), XPUTestPool3DOp, stype)

if __name__ == '__main__':
    unittest.main()
