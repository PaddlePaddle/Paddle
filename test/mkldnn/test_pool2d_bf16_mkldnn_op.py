# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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


import sys
import unittest

import numpy as np
from op_test import OpTest, OpTestTool, convert_float_to_uint16

sys.path.append("../deprecated/legacy_test")
from test_pool2d_op import (
    TestPool2D_Op_Mixin,
    adaptive_end_index,
    adaptive_start_index,
    max_pool2D_forward_naive,
)

from paddle import enable_static
from paddle.base import core


def pool2d_backward_naive(
    x,
    ksize,
    strides,
    paddings,
    global_pool=0,
    ceil_mode=False,
    exclusive=True,
    adaptive=False,
    data_format='NCHW',
    pool_type="max",
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
            paddings = [0, 0, 0, 0]
            if ceil_mode is not False:
                raise ValueError(
                    'When Attr(pool_padding) is "VALID", Attr(ceil_mode)'
                    " must be False. "
                    "Received ceil_mode: True."
                )
        elif padding_algorithm == "SAME":
            input_data_shape = []
            if data_format == "NCHW":
                input_data_shape = x.shape[2:4]
            elif data_format == "NHWC":
                input_data_shape = x.shape[1:3]
            paddings = _get_padding_with_SAME(input_data_shape, ksize, strides)

    assert len(paddings) == 2 or len(paddings) == 4
    is_sys = True if len(paddings) == 2 else False

    if data_format == "NHWC":
        x = x.transpose([0, 3, 1, 2])

    N, C, H, W = x.shape

    if global_pool == 1:
        ksize = [H, W]
        paddings = [0 for _ in range(len(paddings))]

    pad_h_up = paddings[0] if is_sys else paddings[0]
    pad_h_down = paddings[0] if is_sys else paddings[1]
    pad_w_left = paddings[1] if is_sys else paddings[2]
    pad_w_right = paddings[1] if is_sys else paddings[3]

    if adaptive:
        H_out, W_out = ksize
    else:
        H_out = (
            (H - ksize[0] + pad_h_up + pad_h_down + strides[0] - 1)
            // strides[0]
            + 1
            if ceil_mode
            else (H - ksize[0] + pad_h_up + pad_h_down) // strides[0] + 1
        )
        W_out = (
            (W - ksize[1] + pad_w_left + pad_w_right + strides[1] - 1)
            // strides[1]
            + 1
            if ceil_mode
            else (W - ksize[1] + pad_w_left + pad_w_right) // strides[1] + 1
        )

    x_grad = np.zeros_like(x)
    for i in range(H_out):
        if adaptive:
            in_h_start = adaptive_start_index(i, H, ksize[0])
            in_h_end = adaptive_end_index(i, H, ksize[0])
        else:
            in_h_start = np.max((i * strides[0] - pad_h_up, 0))
            in_h_end = np.min((i * strides[0] + ksize[0] - pad_h_up, H))

        for j in range(W_out):
            if adaptive:
                in_w_start = adaptive_start_index(j, W, ksize[1])
                in_w_end = adaptive_end_index(j, W, ksize[1])
            else:
                in_h_start = i * strides[0] - pad_h_up
                in_w_start = j * strides[1] - pad_w_left
                in_h_end = i * strides[0] + ksize[0] - pad_h_up
                in_w_end = j * strides[1] + ksize[1] - pad_w_left

                field_size = (in_h_end - in_h_start) * (in_w_end - in_w_start)
                in_h_start = np.max((in_h_start, 0))
                in_w_start = np.max((in_w_start, 0))
                in_h_end = np.min((in_h_end, H))
                in_w_end = np.min((in_w_end, W))

            if pool_type == 'avg':
                if exclusive or adaptive:
                    field_size = (in_h_end - in_h_start) * (
                        in_w_end - in_w_start
                    )
                x_grad[:, :, in_h_start:in_h_end, in_w_start:in_w_end] += (
                    1 / field_size
                )
            elif pool_type == 'max':
                for n in range(N):
                    for c in range(C):
                        idx = np.argmax(
                            x[
                                n, c, in_h_start:in_h_end, in_w_start:in_w_end
                            ].flatten()
                        )
                        idx_h = idx // (in_w_end - in_w_start)
                        idx_w = idx % (in_w_end - in_w_start)
                        x_grad[
                            n, c, in_h_start + idx_h, in_w_start + idx_w
                        ] += 1

    if data_format == "NHWC":
        x_grad = x_grad.transpose([0, 2, 3, 1])
    return x_grad


@OpTestTool.skip_if_not_cpu_bf16()
class TestPoolBf16MklDNNOpGrad(TestPool2D_Op_Mixin, OpTest):
    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_data_type(self):
        self.dtype = np.uint16

    def setUp(self):
        super().setUp()
        self.attrs['mkldnn_data_type'] = "bfloat16"
        self.x_fp32 = np.random.random(self.shape).astype(np.float32)

        output = self.pool2D_forward_naive(
            self.x_fp32,
            self.ksize,
            self.strides,
            self.paddings,
            self.global_pool,
            self.ceil_mode,
            self.exclusive,
            self.adaptive,
            "float32",
        ).astype(np.float32)

        self.inputs = {'X': convert_float_to_uint16(self.x_fp32)}
        self.outputs = {'Out': convert_float_to_uint16(output)}

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace(), check_pir_onednn=True)

    def test_check_grad(self):
        x_grad = pool2d_backward_naive(
            self.x_fp32,
            ksize=self.ksize,
            strides=self.strides,
            paddings=self.paddings,
            global_pool=self.global_pool,
            ceil_mode=False,
            exclusive=self.exclusive,
            adaptive=self.adaptive,
            data_format=self.data_format,
            pool_type=self.pool_type,
            padding_algorithm=self.padding_algorithm,
        )
        x_grad = x_grad / np.prod(self.outputs['Out'].shape)
        self.check_grad_with_place(
            core.CPUPlace(),
            {'X'},
            'Out',
            user_defined_grads=[x_grad],
            check_pir_onednn=True,
        )


@OpTestTool.skip_if_not_cpu_bf16()
class TestPoolBf16MklDNNOp(TestPool2D_Op_Mixin, OpTest):
    def init_kernel_type(self):
        self.use_mkldnn = True

    def setUp(self):
        TestPool2D_Op_Mixin.setUp(self)
        self.dtype = np.uint16

        input = np.random.random(self.shape).astype(np.float32)
        output = (
            self.pool2D_forward_naive(
                input,
                self.ksize,
                self.strides,
                self.paddings,
                self.global_pool,
                self.ceil_mode,
                self.exclusive,
                self.adaptive,
                "float32",
            )
        ).astype(np.float32)

        self.inputs = {'X': convert_float_to_uint16(input)}
        self.outputs = {'Out': convert_float_to_uint16(output)}

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace(), check_pir_onednn=True)

    def test_check_grad(self):
        pass


class TestCase1Avg(TestPoolBf16MklDNNOp):
    def init_test_case(self):
        self.shape = [2, 3, 7, 7]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [0, 0]

    def init_global_pool(self):
        self.global_pool = False

    def init_exclusive(self):
        self.exclusive = True


class TestCase2Avg(TestPoolBf16MklDNNOp):
    def init_test_case(self):
        self.shape = [2, 3, 7, 7]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 1]

    def init_global_pool(self):
        self.global_pool = False

    def init_exclusive(self):
        self.exclusive = False


class TestCase0Max(TestPoolBf16MklDNNOp):
    def init_pool_type(self):
        self.pool_type = "max"
        self.pool2D_forward_naive = max_pool2D_forward_naive


class TestCase1Max(TestCase1Avg):
    def init_pool_type(self):
        self.pool_type = "max"
        self.pool2D_forward_naive = max_pool2D_forward_naive


class TestCase2Max(TestCase2Avg):
    def init_pool_type(self):
        self.pool_type = "max"
        self.pool2D_forward_naive = max_pool2D_forward_naive


class TestCase1PadZeroExclusiveAvgGrad(TestPoolBf16MklDNNOpGrad):
    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]

    def init_shape(self):
        self.shape = [2, 3, 7, 7]

    def init_paddings(self):
        self.paddings = [0, 0]

    def init_global_pool(self):
        self.global_pool = False

    def init_exclusive(self):
        self.exclusive = True


class TestCase2PadOneNonExclusiveAvgGrad(TestCase1PadZeroExclusiveAvgGrad):
    def init_exclusive(self):
        self.exclusive = False


class TestCase0InitialMaxGrad(TestPoolBf16MklDNNOpGrad):
    def init_pool_type(self):
        self.pool_type = "max"
        self.pool2D_forward_naive = max_pool2D_forward_naive


class TestCase1PadZeroExclusiveMaxGrad(TestCase1PadZeroExclusiveAvgGrad):
    def init_pool_type(self):
        self.pool_type = "max"
        self.pool2D_forward_naive = max_pool2D_forward_naive


class TestCase2PadOneNonExclusiveMaxGrad(TestCase2PadOneNonExclusiveAvgGrad):
    def init_pool_type(self):
        self.pool_type = "max"
        self.pool2D_forward_naive = max_pool2D_forward_naive


if __name__ == "__main__":
    enable_static()
    unittest.main()
