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

import os
import unittest

import gradient_checker
import numpy as np
from decorator_helper import prog_scope

import paddle
from paddle import base
from paddle.base import core


class TestInstanceNormDoubleGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        prog = base.Program()
        with base.program_guard(prog):
            np.random.seed()
            shape = [2, 3, 4, 5]
            dtype = "float32"
            eps = 0.005
            atol = 1e-4
            x = paddle.create_parameter(dtype=dtype, shape=shape, name='x')
            z = paddle.nn.InstanceNorm2D(3)(x)
            x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
            gradient_checker.double_grad_check(
                [x], z, x_init=x_arr, atol=atol, place=place, eps=eps
            )

    @prog_scope()
    def func_pir(self, place):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            np.random.seed()
            shape = [2, 3, 4, 5]
            dtype = "float32"
            eps = 0.005
            atol = 1e-4
            x = paddle.static.data(dtype=dtype, shape=shape, name='x')
            z = paddle.nn.functional.instance_norm(x)
            x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
            gradient_checker.double_grad_check(
                [x], z, x_init=x_arr, atol=atol, place=place, eps=eps
            )

    def test_grad(self):
        paddle.enable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            with paddle.pir_utils.OldIrGuard():
                self.func(p)
            self.func_pir(p)


class TestInstanceNormDoubleGradCheckWithoutParamBias(
    TestInstanceNormDoubleGradCheck
):
    @prog_scope()
    def func(self, place):
        prog = base.Program()
        with base.program_guard(prog):
            np.random.seed()
            shape = [2, 3, 4, 5]
            dtype = "float32"
            eps = 0.005
            atol = 1e-4
            x = paddle.create_parameter(dtype=dtype, shape=shape, name='x')
            z = paddle.nn.InstanceNorm2D(3, weight_attr=False, bias_attr=False)(
                x
            )
            x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
            gradient_checker.double_grad_check(
                [x], z, x_init=x_arr, atol=atol, place=place, eps=eps
            )

    @prog_scope()
    def func_pir(self, place):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            np.random.seed()
            shape = [2, 3, 4, 5]
            dtype = "float32"
            eps = 0.005
            atol = 1e-4
            x = paddle.static.data(dtype=dtype, shape=shape, name='x')
            z = paddle.nn.functional.instance_norm(x, bias=None)
            x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
            gradient_checker.double_grad_check(
                [x], z, x_init=x_arr, atol=atol, place=place, eps=eps
            )


class TestInstanceNormDoubleGradEagerCheck(unittest.TestCase):
    def instance_norm_wrapper(self, x):
        return paddle.nn.functional.instance_norm(x[0])

    @prog_scope()
    def func(self, place):
        prog = base.Program()
        with base.program_guard(prog):
            np.random.seed()
            shape = [2, 3, 4, 5]
            dtype = "float32"
            eps = 0.005
            atol = 1e-4
            x = paddle.static.data(dtype=dtype, shape=shape, name='x')
            z = paddle.nn.functional.instance_norm(x)
            x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
            # check for static graph mode
            gradient_checker.double_grad_check(
                [x], z, x_init=x_arr, atol=atol, place=place, eps=eps
            )
            # check for eager mode
            gradient_checker.double_grad_check_for_dygraph(
                self.instance_norm_wrapper,
                [x],
                z,
                x_init=x_arr,
                atol=atol,
                place=place,
            )

    def test_grad(self):
        paddle.enable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestInstanceNormDoubleGradEagerCheckWithParams(
    TestInstanceNormDoubleGradEagerCheck
):
    def instance_norm_wrapper(self, x):
        instance_norm = paddle.nn.InstanceNorm2D(3)
        return instance_norm(x[0])

    @prog_scope()
    def func(self, place):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            np.random.seed()
            shape = [2, 3, 4, 5]
            dtype = "float32"
            eps = 0.005
            atol = 1e-4
            x = paddle.static.data(dtype=dtype, shape=shape, name='x')
            z = paddle.nn.InstanceNorm2D(3)(x)
            x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
            # check for static graph mode
            gradient_checker.double_grad_check(
                [x], z, x_init=x_arr, atol=atol, place=place, eps=eps
            )
            # check for eager mode
            gradient_checker.double_grad_check_for_dygraph(
                self.instance_norm_wrapper,
                [x],
                z,
                x_init=x_arr,
                atol=atol,
                place=place,
            )


class TestBatchNormDoubleGradCheck(unittest.TestCase):
    def setUp(self):
        self.init_test()

    def init_test(self):
        self.data_layout = 'NCHW'
        self.use_global_stats = False
        self.shape = [2, 3, 4, 5]
        self.channel_index = 1

    def batch_norm_wrapper(self, x):
        batch_norm = paddle.nn.BatchNorm2D(
            self.shape[self.channel_index],
            data_format=self.data_layout,
            use_global_stats=self.use_global_stats,
        )
        return batch_norm(x[0])

    @prog_scope()
    def func(self, place):
        prog = base.Program()
        with base.program_guard(prog):
            np.random.seed()
            dtype = "float32"
            eps = 0.005
            atol = 1e-4
            x = paddle.create_parameter(dtype=dtype, shape=self.shape, name='x')
            z = paddle.static.nn.batch_norm(
                input=x,
                data_layout=self.data_layout,
                use_global_stats=self.use_global_stats,
            )
            x_arr = np.random.uniform(-1, 1, self.shape).astype(dtype)
            gradient_checker.double_grad_check(
                [x], z, x_init=x_arr, atol=atol, place=place, eps=eps
            )
            gradient_checker.double_grad_check_for_dygraph(
                self.batch_norm_wrapper,
                [x],
                z,
                x_init=x_arr,
                atol=atol,
                place=place,
            )

    @prog_scope()
    def func_pir(self, place):
        prog = base.Program()
        with base.program_guard(prog):
            np.random.seed()
            dtype = "float32"
            eps = 0.005
            atol = 1e-4
            x = paddle.static.data(dtype=dtype, shape=self.shape, name='x')
            bn = paddle.nn.BatchNorm2D(
                self.shape[self.channel_index],
                data_format=self.data_layout,
                use_global_stats=self.use_global_stats,
            )
            z = bn(x)
            x_arr = np.random.uniform(-1, 1, self.shape).astype(dtype)
            gradient_checker.double_grad_check(
                [x], z, x_init=x_arr, atol=atol, place=place, eps=eps
            )
            gradient_checker.double_grad_check_for_dygraph(
                self.batch_norm_wrapper,
                [x],
                z,
                x_init=x_arr,
                atol=atol,
                place=place,
            )

    def test_grad(self):
        paddle.enable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            with paddle.pir_utils.OldIrGuard():
                self.func(p)
            self.func_pir(p)


class TestBatchNormDoubleGradCheckCase1(TestBatchNormDoubleGradCheck):
    def init_test(self):
        self.data_layout = 'NHWC'
        self.use_global_stats = False
        self.shape = [2, 3, 4, 5]
        self.channel_index = 3


class TestBatchNormDoubleGradCheckCase2(TestBatchNormDoubleGradCheck):
    def init_test(self):
        self.data_layout = 'NCHW'
        self.use_global_stats = True
        self.shape = [2, 3, 4, 5]
        self.channel_index = 1


class TestBatchNormDoubleGradCheckCase3(TestBatchNormDoubleGradCheck):
    def init_test(self):
        self.data_layout = 'NHWC'
        self.use_global_stats = True
        self.shape = [2, 3, 4, 5]
        self.channel_index = 3


class TestBatchNormDoubleGradCheckCase4(TestBatchNormDoubleGradCheck):
    def init_test(self):
        self.data_layout = 'NCHW'
        self.use_global_stats = False
        self.shape = [2, 2, 3, 4, 5]
        self.channel_index = 1

    def batch_norm_wrapper(self, x):
        batch_norm = paddle.nn.BatchNorm3D(
            self.shape[self.channel_index],
            data_format=self.data_layout,
            use_global_stats=self.use_global_stats,
        )
        return batch_norm(x[0])

    @prog_scope()
    def func_pir(self, place):
        prog = base.Program()
        with base.program_guard(prog):
            np.random.seed()
            dtype = "float32"
            eps = 0.005
            atol = 1e-4
            x = paddle.static.data(dtype=dtype, shape=self.shape, name='x')
            bn = paddle.nn.BatchNorm3D(
                self.shape[self.channel_index],
                data_format=self.data_layout,
                use_global_stats=self.use_global_stats,
            )
            z = bn(x)
            x_arr = np.random.uniform(-1, 1, self.shape).astype(dtype)
            gradient_checker.double_grad_check(
                [x], z, x_init=x_arr, atol=atol, place=place, eps=eps
            )
            gradient_checker.double_grad_check_for_dygraph(
                self.batch_norm_wrapper,
                [x],
                z,
                x_init=x_arr,
                atol=atol,
                place=place,
            )


class TestBatchNormDoubleGradCheckCase5(TestBatchNormDoubleGradCheck):
    @prog_scope()
    def func(self, place):
        prog = base.Program()
        with base.program_guard(prog):
            np.random.seed(37)
            dtype = "float32"
            eps = 0.005
            atol = 2e-4
            chn = (
                self.shape[1] if self.data_layout == 'NCHW' else self.shape[-1]
            )
            x = paddle.create_parameter(dtype=dtype, shape=self.shape, name='x')
            z = paddle.static.nn.batch_norm(
                input=x,
                data_layout=self.data_layout,
                use_global_stats=self.use_global_stats,
            )
            x_arr = np.random.uniform(-1, 1, self.shape).astype(dtype)
            w, b = prog.global_block().all_parameters()[1:3]
            w_arr = np.ones(chn).astype(dtype)
            b_arr = np.zeros(chn).astype(dtype)
            gradient_checker.double_grad_check(
                [x, w, b],
                z,
                x_init=[x_arr, w_arr, b_arr],
                atol=atol,
                place=place,
                eps=eps,
            )

    @prog_scope()
    def func_pir(self, place):
        prog = base.Program()
        with base.program_guard(prog):
            np.random.seed(37)
            dtype = "float32"
            eps = 0.005
            atol = 2e-4
            chn = (
                self.shape[1] if self.data_layout == 'NCHW' else self.shape[-1]
            )
            x = paddle.static.data(dtype=dtype, shape=self.shape, name='x')
            w = paddle.static.data(dtype=dtype, shape=[chn], name='w')
            b = paddle.static.data(dtype=dtype, shape=[chn], name='b')
            bn = paddle.nn.BatchNorm2D(
                self.shape[self.channel_index],
                data_format=self.data_layout,
                use_global_stats=self.use_global_stats,
            )
            z = bn(x)
            x_arr = np.random.uniform(-1, 1, self.shape).astype(dtype)
            w_arr = np.ones(chn).astype(dtype)
            b_arr = np.zeros(chn).astype(dtype)
            gradient_checker.double_grad_check(
                [x, w, b],
                z,
                x_init=[x_arr, w_arr, b_arr],
                atol=atol,
                place=place,
                eps=eps,
            )


class TestBatchNormDoubleGradCheckCase6(TestBatchNormDoubleGradCheckCase5):
    def init_test(self):
        self.data_layout = 'NCHW'
        self.use_global_stats = True
        self.shape = [2, 3, 4, 5]
        self.channel_index = 1


if __name__ == "__main__":
    unittest.main()
