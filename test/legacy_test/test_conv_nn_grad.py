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
import paddle.nn.functional as F
from paddle import base
from paddle.base import core


class TestConvDoubleGradCheck(unittest.TestCase):

    @prog_scope()
    def func_pir(self, place):
        shape = [2, 4, 3, 3]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', shape, dtype)
        w = paddle.static.data('w', shape, dtype)
        x.persistable = True
        w.persistable = True
        y = F.conv2d(x, w, groups=1)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        w_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x, w], y, x_init=[x_arr, w_arr], place=place, eps=eps
        )

    def test_grad(self):
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
            self.func_pir(p)


class TestConvDoubleGradCheckTest0(unittest.TestCase):

    @prog_scope()
    def func_pir(self, place):
        shape = [2, 4, 3, 3]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', shape, dtype)
        w = paddle.static.data('w', shape, dtype)
        x.persistable = True
        w.persistable = True
        y = F.conv2d(x, w)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        w_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x, w], y, x_init=[x_arr, w_arr], place=place, eps=eps
        )

    def test_grad(self):
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
            self.func_pir(p)


class TestConvDoubleGradCheckTest1(unittest.TestCase):

    @prog_scope()
    def func_pir(self, place):
        shape = [2, 3, 3, 3]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', shape, dtype)
        w = paddle.static.data('w', shape, dtype)
        x.persistable = True
        w.persistable = True
        y = F.conv2d(x, w, padding=1)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        w_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x, w], y, x_init=[x_arr, w_arr], place=place, eps=eps
        )

    def test_grad(self):
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
            self.func_pir(p)


class TestConv3DDoubleGradCheck(unittest.TestCase):

    @prog_scope()
    def func_pir(self, place):
        shape = [2, 4, 3, 4, 2]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', shape, dtype)
        w = paddle.static.data('w', shape, dtype)
        x.persistable = True
        w.persistable = True
        y = F.conv3d(x, w)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        w_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x, w], y, x_init=[x_arr, w_arr], place=place, eps=eps
        )

    def test_grad(self):
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
            self.func_pir(p)


class TestConv3DDoubleGradCheckTest1(unittest.TestCase):

    @prog_scope()
    def func_pir(self, place):
        shape = [2, 4, 5, 3, 2]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', shape, dtype)
        w = paddle.static.data('w', shape, dtype)
        x.persistable = True
        w.persistable = True
        y = F.conv3d(x, w, padding=1)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        w_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x, w], y, x_init=[x_arr, w_arr], place=place, eps=eps
        )

    def test_grad(self):
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
            self.func_pir(p)


class TestConv2DoubleGradCheck_AsyPadding(unittest.TestCase):

    @prog_scope()
    def func_pir(self, place):
        shape = [2, 2, 3, 3]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', shape, dtype)
        w = paddle.static.data('w', shape, dtype)
        x.persistable = True
        w.persistable = True
        y = F.conv2d(x, w, padding=[1, 0, 0, 1])
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        w_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x, w], y, x_init=[x_arr, w_arr], place=place, eps=eps
        )

    def test_grad(self):
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
            self.func_pir(p)


class TestConv2DoubleGradCheck_PaddingSAME(unittest.TestCase):

    @prog_scope()
    def func_pir(self, place):
        shape = [2, 2, 3, 3]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', shape, dtype)
        w = paddle.static.data('w', shape, dtype)
        x.persistable = True
        w.persistable = True
        y = F.conv2d(x, w, padding="SAME")
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        w_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x, w], y, x_init=[x_arr, w_arr], place=place, eps=eps
        )

    def test_grad(self):
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
            self.func_pir(p)


class TestConv2DoubleGradCheck_PaddingVALID(unittest.TestCase):

    @prog_scope()
    def func_pir(self, place):
        shape = [2, 2, 3, 3]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', shape, dtype)
        w = paddle.static.data('w', shape, dtype)
        x.persistable = True
        w.persistable = True
        y = F.conv2d(x, w, padding="VALID")
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        w_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x, w], y, x_init=[x_arr, w_arr], place=place, eps=eps
        )

    def test_grad(self):
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
            self.func_pir(p)


class TestConv2DoubleGradCheck_ChannelLast(unittest.TestCase):

    @prog_scope()
    def func_pir(self, place):
        x_shape = [2, 2, 3, 3]
        w_shape = [2, 3, 1, 1]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', x_shape, dtype)
        w = paddle.static.data('w', w_shape, dtype)
        x.persistable = True
        w.persistable = True
        y = F.conv2d(x, w, padding=[1, 1], groups=1, data_format="NHWC")
        x_arr = np.random.uniform(-1, 1, x_shape).astype(dtype)
        w_arr = np.random.uniform(-1, 1, w_shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x, w], y, x_init=[x_arr, w_arr], place=place, eps=eps
        )

    def test_grad(self):
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
            self.func_pir(p)


class TestConv2DoubleGradCheck_ChannelLast_AsyPadding(unittest.TestCase):

    @prog_scope()
    def func_pir(self, place):
        x_shape = [2, 2, 3, 3]
        w_shape = [2, 3, 1, 1]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', x_shape, dtype)
        w = paddle.static.data('w', w_shape, dtype)
        x.persistable = True
        w.persistable = True
        y = F.conv2d(x, w, padding=[1, 0, 1, 0], groups=1, data_format="NHWC")
        x_arr = np.random.uniform(-1, 1, x_shape).astype(dtype)
        w_arr = np.random.uniform(-1, 1, w_shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x, w], y, x_init=[x_arr, w_arr], place=place, eps=eps
        )

    def test_grad(self):
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
            self.func_pir(p)


class TestConv3DDoubleGradCheck_AsyPadding(unittest.TestCase):

    @prog_scope()
    def func_pir(self, place):
        shape = [2, 2, 2, 2, 2]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', shape, dtype)
        w = paddle.static.data('w', shape, dtype)
        x.persistable = True
        w.persistable = True
        y = F.conv3d(x, w, padding=[1, 0, 0, 1, 1, 2])
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        w_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x, w], y, x_init=[x_arr, w_arr], place=place, eps=eps
        )

    def test_grad(self):
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
            self.func_pir(p)


class TestConv3DoubleGradCheck_PaddingSAME(unittest.TestCase):

    @prog_scope()
    def func_pir(self, place):
        shape = [2, 2, 2, 2, 2]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', shape, dtype)
        w = paddle.static.data('w', shape, dtype)
        x.persistable = True
        w.persistable = True
        y = F.conv3d(x, w, padding="SAME")
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        w_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x, w], y, x_init=[x_arr, w_arr], place=place, eps=eps
        )

    def test_grad(self):
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
            self.func_pir(p)


class TestConv3DoubleGradCheck_PaddingVALID(unittest.TestCase):

    @prog_scope()
    def func_pir(self, place):
        shape = [2, 2, 3, 3, 2]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', shape, dtype)
        w = paddle.static.data('w', shape, dtype)
        x.persistable = True
        w.persistable = True
        y = F.conv3d(x, w, padding="VALID")
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        w_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x, w], y, x_init=[x_arr, w_arr], place=place, eps=eps
        )

    def test_grad(self):
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
            self.func_pir(p)


class TestConv3DDoubleGradCheck_ChannelLast(unittest.TestCase):

    @prog_scope()
    def func_pir(self, place):
        x_shape = [2, 2, 2, 2, 3]
        w_shape = [2, 3, 1, 1, 1]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', x_shape, dtype)
        w = paddle.static.data('w', w_shape, dtype)
        x.persistable = True
        w.persistable = True
        y = F.conv3d(x, w, padding=[1, 1, 1], data_format="NDHWC")
        x_arr = np.random.uniform(-1, 1, x_shape).astype(dtype)
        w_arr = np.random.uniform(-1, 1, w_shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x, w], y, x_init=[x_arr, w_arr], place=place, eps=eps
        )

    def test_grad(self):
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
            self.func_pir(p)


class TestConv3DDoubleGradCheck_ChannelLast_AsyPadding(unittest.TestCase):

    @prog_scope()
    def func_pir(self, place):
        x_shape = [2, 2, 2, 2, 3]
        w_shape = [2, 3, 1, 1, 1]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', x_shape, dtype)
        w = paddle.static.data('w', w_shape, dtype)
        x.persistable = True
        w.persistable = True
        y = F.conv3d(x, w, padding=[1, 0, 1, 0, 1, 0], data_format="NDHWC")
        x_arr = np.random.uniform(-1, 1, x_shape).astype(dtype)
        w_arr = np.random.uniform(-1, 1, w_shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x, w], y, x_init=[x_arr, w_arr], place=place, eps=eps
        )

    def test_grad(self):
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
            self.func_pir(p)


class TestDepthWiseConvDoubleGradCheck(unittest.TestCase):

    @prog_scope()
    def func_pir(self, place):
        x_shape = [2, 4, 3, 3]
        w_shape = [4, 1, 1, 1]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', x_shape, dtype)
        w = paddle.static.data('w', w_shape, dtype)
        # x.persistable = True
        # w.persistable = True
        y = F.conv2d(x, w, groups=4)
        x_arr = np.random.uniform(-1, 1, x_shape).astype(dtype)
        w_arr = np.random.uniform(-1, 1, w_shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x, w], y, x_init=[x_arr, w_arr], place=place, eps=eps
        )

    def test_grad(self):
        places = []
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func_pir(p)


class TestDepthWiseConvDoubleGradCheckCase1(unittest.TestCase):
    def depthwise_conv2d_wrapper(self, x):
        return paddle.nn.functional.conv2d(x[0], x[1], groups=4)

    @prog_scope()
    def func(self, place):
        x_shape = [2, 4, 3, 3]
        w_shape = [4, 1, 3, 3]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', x_shape, dtype)
        w = paddle.static.data('w', w_shape, dtype)

        # condition of depthwise conv:
        # use_cudnn == False
        # groups == filters
        # num_filters % num_channels == 0

        y = paddle.nn.functional.conv2d(x, w, groups=4)
        x_arr = np.random.uniform(-1, 1, x_shape).astype(dtype)
        w_arr = np.random.uniform(-1, 1, w_shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x, w], y, x_init=[x_arr, w_arr], place=place, eps=eps
        )
        gradient_checker.double_grad_check_for_dygraph(
            self.depthwise_conv2d_wrapper,
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
            self.func(p)


class TestConv3DDoubleGradCheck_NN(unittest.TestCase):
    def conv3d_wrapper(self, x):
        return paddle.nn.functional.conv3d(x[0], x[1])

    @prog_scope()
    def func(self, place):
        x_shape = [2, 3, 8, 8, 8]
        w_shape = [6, 3, 3, 3, 3]
        eps = 0.005
        dtype = np.float32 if base.core.is_compiled_with_rocm() else np.float64
        x = paddle.static.data('x', x_shape, dtype)
        w = paddle.static.data('w', w_shape, dtype)
        x.persistable = True
        w.persistable = True
        y = paddle.nn.functional.conv3d(x, w)
        x_arr = np.random.uniform(-1, 1, x_shape).astype(dtype)
        w_arr = np.random.uniform(-1, 1, w_shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x, w], y, x_init=[x_arr, w_arr], place=place, eps=eps
        )
        gradient_checker.double_grad_check_for_dygraph(
            self.conv3d_wrapper, [x, w], y, x_init=[x_arr, w_arr], place=place
        )

    def test_grad(self):
        places = []
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
