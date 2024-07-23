#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import gradient_checker
import numpy as np
from decorator_helper import prog_scope

import paddle
import paddle.nn.functional as F
from paddle import base
from paddle.base import core
from paddle.pir_utils import test_with_pir_api


class TestConvTransposeDoubleGradCheck(unittest.TestCase):
    def conv_transpose_wrapper(self, x):
        return paddle.nn.functional.conv2d_transpose(x[0], x[1], groups=1)

    @prog_scope()
    def func(self, place):
        shape = [2, 4, 3, 3]
        eps = 0.005
        dtype = np.float64
        if core.is_compiled_with_rocm():
            dtype = np.float32
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.nn.conv2d_transpose(
            x, 2, filter_size=1, groups=1, bias_attr=False
        )
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        w = base.default_main_program().global_block().all_parameters()
        w_arr = []
        for p in w:
            w_arr.append(np.random.uniform(-1, 1, p.shape).astype(dtype))
        if core.is_compiled_with_rocm():
            # HIP will sometimes fail if no atol
            gradient_checker.double_grad_check(
                [x] + w,
                y,
                x_init=[x_arr] + w_arr,
                place=place,
                eps=eps,
                atol=1e-4,
            )
        else:
            gradient_checker.double_grad_check(
                [x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps
            )
        gradient_checker.double_grad_check_for_dygraph(
            self.conv_transpose_wrapper,
            [x] + w,
            y,
            x_init=[x_arr] + w_arr,
            place=place,
        )

    @test_with_pir_api
    @prog_scope()
    def func_pir(self, place):
        x_shape = [2, 4, 3, 3]
        w_shape = [4, 2, 1, 1]
        eps = 0.005
        dtype = np.float64
        if core.is_compiled_with_rocm():
            dtype = np.float32
        x = paddle.static.data('x', x_shape, dtype)
        w = paddle.static.data('w', w_shape, dtype)
        y = F.conv2d_transpose(x, w, groups=1)
        x_arr = np.random.uniform(-1, 1, x_shape).astype(dtype)
        w_arr = np.random.uniform(-1, 1, w_shape).astype(dtype)

        if core.is_compiled_with_rocm():
            # HIP will sometimes fail if no atol
            gradient_checker.double_grad_check(
                [x, w],
                y,
                x_init=[x_arr, w_arr],
                place=place,
                eps=eps,
                atol=1e-4,
            )
        else:
            gradient_checker.double_grad_check(
                [x, w], y, x_init=[x_arr, w_arr], place=place, eps=eps
            )
        gradient_checker.double_grad_check_for_dygraph(
            self.conv_transpose_wrapper,
            [x, w],
            y,
            x_init=[x_arr, w_arr],
            place=place,
        )

    def test_grad(self):
        places = []

        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            with paddle.pir_utils.OldIrGuard():
                self.func(p)
            self.func_pir(p)


class TestConvTranspose2DoubleGradCheck_AsyPadding(
    TestConvTransposeDoubleGradCheck
):
    def conv_transpose_wrapper(self, x):
        return paddle.nn.functional.conv2d_transpose(
            x[0], x[1], groups=1, padding=[1, 0, 0, 1]
        )

    @prog_scope()
    def func(self, place):
        shape = [2, 2, 3, 3]
        eps = 0.005
        dtype = np.float64
        if core.is_compiled_with_rocm():
            dtype = np.float32
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.nn.conv2d_transpose(
            input=x,
            num_filters=2,
            filter_size=1,
            padding=[1, 0, 0, 1],
            bias_attr=False,
            use_cudnn=True,
        )

        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        w = base.default_main_program().global_block().all_parameters()
        w_arr = []
        for p in w:
            w_arr.append(np.random.uniform(-1, 1, p.shape).astype(dtype))
        if core.is_compiled_with_rocm():
            # HIP will sometimes fail if no atol
            gradient_checker.double_grad_check(
                [x] + w,
                y,
                x_init=[x_arr] + w_arr,
                place=place,
                eps=eps,
                atol=1e-4,
            )
        else:
            gradient_checker.double_grad_check(
                [x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps
            )
        gradient_checker.double_grad_check_for_dygraph(
            self.conv_transpose_wrapper,
            [x] + w,
            y,
            x_init=[x_arr] + w_arr,
            place=place,
        )

    @test_with_pir_api
    @prog_scope()
    def func_pir(self, place):
        x_shape = [2, 2, 3, 3]
        w_shape = [2, 2, 1, 1]
        eps = 0.005
        dtype = np.float64
        if core.is_compiled_with_rocm():
            dtype = np.float32
        x = paddle.static.data('x', x_shape, dtype)
        w = paddle.static.data('w', w_shape, dtype)
        y = F.conv2d_transpose(x, w, padding=[1, 0, 0, 1])
        x_arr = np.random.uniform(-1, 1, x_shape).astype(dtype)
        w_arr = np.random.uniform(-1, 1, w_shape).astype(dtype)

        if core.is_compiled_with_rocm():
            # HIP will sometimes fail if no atol
            gradient_checker.double_grad_check(
                [x, w],
                y,
                x_init=[x_arr, w_arr],
                place=place,
                eps=eps,
                atol=1e-4,
            )
        else:
            gradient_checker.double_grad_check(
                [x, w], y, x_init=[x_arr, w_arr], place=place, eps=eps
            )
        gradient_checker.double_grad_check_for_dygraph(
            self.conv_transpose_wrapper,
            [x, w],
            y,
            x_init=[x_arr, w_arr],
            place=place,
        )


class TestConvTranspose2DoubleGradCheck_PaddingSAME(
    TestConvTransposeDoubleGradCheck
):
    def conv_transpose_wrapper(self, x):
        return paddle.nn.functional.conv2d_transpose(
            x[0], x[1], groups=1, padding="SAME"
        )

    @prog_scope()
    def func(self, place):
        shape = [2, 2, 3, 3]
        eps = 0.005
        dtype = np.float64
        if core.is_compiled_with_rocm():
            dtype = np.float32
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.nn.conv2d_transpose(
            input=x,
            num_filters=2,
            filter_size=1,
            padding="SAME",
            bias_attr=False,
            use_cudnn=True,
        )
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        w = base.default_main_program().global_block().all_parameters()
        w_arr = []
        for p in w:
            w_arr.append(np.random.uniform(-1, 1, p.shape).astype(dtype))
        if core.is_compiled_with_rocm():
            # HIP will sometimes fail if no atol
            gradient_checker.double_grad_check(
                [x] + w,
                y,
                x_init=[x_arr] + w_arr,
                place=place,
                eps=eps,
                atol=1e-4,
            )
        else:
            gradient_checker.double_grad_check(
                [x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps
            )
        gradient_checker.double_grad_check_for_dygraph(
            self.conv_transpose_wrapper,
            [x] + w,
            y,
            x_init=[x_arr] + w_arr,
            place=place,
        )

    @test_with_pir_api
    @prog_scope()
    def func_pir(self, place):
        x_shape = [2, 2, 3, 3]
        w_shape = [2, 2, 1, 1]
        eps = 0.005
        dtype = np.float64
        if core.is_compiled_with_rocm():
            dtype = np.float32
        x = paddle.static.data('x', x_shape, dtype)
        w = paddle.static.data('w', w_shape, dtype)
        y = F.conv2d_transpose(x, w, padding="SAME")
        x_arr = np.random.uniform(-1, 1, x_shape).astype(dtype)
        w_arr = np.random.uniform(-1, 1, w_shape).astype(dtype)

        if core.is_compiled_with_rocm():
            # HIP will sometimes fail if no atol
            gradient_checker.double_grad_check(
                [x, w],
                y,
                x_init=[x_arr, w_arr],
                place=place,
                eps=eps,
                atol=1e-4,
            )
        else:
            gradient_checker.double_grad_check(
                [x, w], y, x_init=[x_arr, w_arr], place=place, eps=eps
            )
        gradient_checker.double_grad_check_for_dygraph(
            self.conv_transpose_wrapper,
            [x, w],
            y,
            x_init=[x_arr, w_arr],
            place=place,
        )


class TestConvTranspose2DoubleGradCheck_PaddingVALID(
    TestConvTransposeDoubleGradCheck
):
    def conv_transpose_wrapper(self, x):
        return paddle.nn.functional.conv2d_transpose(
            x[0], x[1], groups=1, padding="VALID"
        )

    @prog_scope()
    def func(self, place):
        shape = [2, 2, 3, 3]
        eps = 0.005
        dtype = np.float64
        if core.is_compiled_with_rocm():
            dtype = np.float32
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.nn.conv2d_transpose(
            input=x,
            num_filters=2,
            filter_size=1,
            padding="VALID",
            bias_attr=False,
            use_cudnn=True,
        )

        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        w = base.default_main_program().global_block().all_parameters()
        w_arr = []
        for p in w:
            w_arr.append(np.random.uniform(-1, 1, p.shape).astype(dtype))
        if core.is_compiled_with_rocm():
            # HIP will sometimes fail if no atol
            gradient_checker.double_grad_check(
                [x] + w,
                y,
                x_init=[x_arr] + w_arr,
                place=place,
                eps=eps,
                atol=1e-4,
            )
        else:
            gradient_checker.double_grad_check(
                [x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps
            )
        gradient_checker.double_grad_check_for_dygraph(
            self.conv_transpose_wrapper,
            [x] + w,
            y,
            x_init=[x_arr] + w_arr,
            place=place,
        )

    @test_with_pir_api
    @prog_scope()
    def func_pir(self, place):
        x_shape = [2, 2, 3, 3]
        w_shape = [2, 2, 1, 1]
        eps = 0.005
        dtype = np.float64
        if core.is_compiled_with_rocm():
            dtype = np.float32
        x = paddle.static.data('x', x_shape, dtype)
        w = paddle.static.data('w', w_shape, dtype)
        y = F.conv2d_transpose(x, w, padding="VALID")
        x_arr = np.random.uniform(-1, 1, x_shape).astype(dtype)
        w_arr = np.random.uniform(-1, 1, w_shape).astype(dtype)

        if core.is_compiled_with_rocm():
            # HIP will sometimes fail if no atol
            gradient_checker.double_grad_check(
                [x, w],
                y,
                x_init=[x_arr, w_arr],
                place=place,
                eps=eps,
                atol=1e-4,
            )
        else:
            gradient_checker.double_grad_check(
                [x, w], y, x_init=[x_arr, w_arr], place=place, eps=eps
            )
        gradient_checker.double_grad_check_for_dygraph(
            self.conv_transpose_wrapper,
            [x, w],
            y,
            x_init=[x_arr, w_arr],
            place=place,
        )


class TestConvTranspose2DoubleGradCheck_ChannelLast(
    TestConvTransposeDoubleGradCheck
):
    def conv_transpose_wrapper(self, x):
        return paddle.nn.functional.conv2d_transpose(
            x[0], x[1], groups=1, padding=[1, 1], data_format="NHWC"
        )

    @prog_scope()
    def func(self, place):
        shape = [2, 3, 3, 2]
        eps = 0.005
        dtype = np.float64
        if core.is_compiled_with_rocm():
            dtype = np.float32
        x = paddle.static.data('x', shape, dtype)

        y = paddle.static.nn.conv2d_transpose(
            input=x,
            num_filters=2,
            filter_size=1,
            padding=[1, 1],
            bias_attr=False,
            use_cudnn=True,
            groups=1,
            data_format="NHWC",
        )
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        w = base.default_main_program().global_block().all_parameters()
        w_arr = []
        for p in w:
            w_arr.append(np.random.uniform(-1, 1, p.shape).astype(dtype))
        if core.is_compiled_with_rocm():
            # HIP will sometimes fail if no atol
            gradient_checker.double_grad_check(
                [x] + w,
                y,
                x_init=[x_arr] + w_arr,
                place=place,
                eps=eps,
                atol=1e-4,
            )
        else:
            gradient_checker.double_grad_check(
                [x] + w, y, x_init=[x_arr] + w_arr, place=place, eps=eps
            )
        gradient_checker.double_grad_check_for_dygraph(
            self.conv_transpose_wrapper,
            [x] + w,
            y,
            x_init=[x_arr] + w_arr,
            place=place,
        )

    @test_with_pir_api
    @prog_scope()
    def func_pir(self, place):
        x_shape = [2, 3, 3, 2]
        w_shape = [2, 2, 1, 1]
        eps = 0.005
        dtype = np.float64
        if core.is_compiled_with_rocm():
            dtype = np.float32
        x = paddle.static.data('x', x_shape, dtype)
        w = paddle.static.data('w', w_shape, dtype)
        y = F.conv2d_transpose(x, w, padding=[1, 1], data_format="NHWC")
        x_arr = np.random.uniform(-1, 1, x_shape).astype(dtype)
        w_arr = np.random.uniform(-1, 1, w_shape).astype(dtype)

        if core.is_compiled_with_rocm():
            # HIP will sometimes fail if no atol
            gradient_checker.double_grad_check(
                [x, w],
                y,
                x_init=[x_arr, w_arr],
                place=place,
                eps=eps,
                atol=1e-4,
            )
        else:
            gradient_checker.double_grad_check(
                [x, w], y, x_init=[x_arr, w_arr], place=place, eps=eps
            )
        gradient_checker.double_grad_check_for_dygraph(
            self.conv_transpose_wrapper,
            [x, w],
            y,
            x_init=[x_arr, w_arr],
            place=place,
        )


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
