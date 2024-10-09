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

paddle.enable_static()


class TestSliceOpDoubleGradCheck(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        self.config()

        out = paddle.slice(
            self.inputs, axes=self.axes, starts=self.starts, ends=self.ends
        )
        gradient_checker.double_grad_check(
            [self.inputs], out, x_init=self.x_arr, place=place
        )

    def config(self):
        self.starts = [1, 0, -1]
        self.ends = [3, 3, 6]
        self.axes = [0, 1, 2]
        self.x_arr = np.random.random([3, 4, 5, 2]).astype("float64")
        self.inputs = paddle.static.data(
            dtype="float64", shape=[3, 4, 5, 2], name='x'
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
        for place in places:
            self.func(place)


class TestSliceOpDoubleGradCheckCase3(TestSliceOpDoubleGradCheck):
    def config(self):
        self.starts = [1, -1, 1]
        self.ends = [3, 3, 3]
        self.axes = [0, 1, 2]
        self.x_arr = np.random.random([3, 3, 3]).astype("float64")
        self.inputs = paddle.static.data(
            dtype="float64", shape=[3, 3, 3], name='x3'
        )


class TestReduceMeanWithDimDoubleGradCheck(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        shape = [7, 11]
        eps = 0.05
        dtype = np.float64

        x = paddle.static.data('x', shape, dtype)
        x.persistable = True
        y = paddle.mean(x, axis=0)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x], y, x_init=x_arr, place=place, eps=eps
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
            self.func(p)


class TestReduceSumWithDimDoubleGradCheck(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        shape = [7, 11]
        eps = 0.05
        dtype = np.float64

        x = paddle.static.data('x', shape, dtype)
        x.persistable = True
        y = paddle.sum(x, axis=0)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x], y, x_init=x_arr, place=place, eps=eps
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
            self.func(p)


class TestReshapeDoubleGradCheck(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        x_shape = [3, 12]
        new_shape = [4, 9]
        eps = 0.005
        dtype = np.float64

        x = paddle.static.data('x', x_shape, dtype)
        x.persistable = True
        out = paddle.reshape(x, new_shape)
        x_arr = np.random.uniform(-1, 1, x_shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x], out, x_init=x_arr, place=place, eps=eps
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
            self.func(p)


class TestTileDoubleGradCheck(unittest.TestCase):
    def tile_wrapper(self, x):
        return paddle.tile(x[0], [4, 9])

    @prog_scope()
    def func(self, place):
        x_shape = [3, 12]
        repeat_times = [4, 9]
        eps = 0.005
        dtype = np.float64

        x = paddle.static.data('x', x_shape, dtype)
        x.persistable = True
        out = paddle.tile(x, repeat_times)
        x_arr = np.random.uniform(-1, 1, x_shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x], out, x_init=x_arr, place=place, eps=eps
        )
        gradient_checker.double_grad_check_for_dygraph(
            self.tile_wrapper, [x], out, x_init=x_arr, place=place
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
            self.func(p)


class TestExpandV2DoubleGradCheck(unittest.TestCase):
    def expand_wrapper(self, x):
        return paddle.expand(x[0], [4, 12])

    @prog_scope()
    def func(self, place):
        x_shape = [1, 12]
        new_shape = [4, 12]
        eps = 0.005
        dtype = np.float64

        x = paddle.static.data('x', x_shape, dtype)
        x.persistable = True
        out = paddle.expand(x, new_shape)
        x_arr = np.random.uniform(-1, 1, x_shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x], out, x_init=x_arr, place=place, eps=eps
        )
        gradient_checker.double_grad_check_for_dygraph(
            self.expand_wrapper, [x], out, x_init=x_arr, place=place
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
            self.func(p)


class TestSqueezeDoubleGradCheck(unittest.TestCase):
    def squeeze_wrapper(self, x):
        axes = [0, 2]
        return paddle.squeeze(x[0], axes)

    @prog_scope()
    def func(self, place):
        x_shape = [1, 3, 1, 40]
        axes = [0, 2]
        eps = 0.005
        dtype = np.float64

        x = paddle.static.data('x', x_shape, dtype)
        x.persistable = True
        out = paddle.squeeze(x, axes)
        x_arr = np.random.uniform(-1, 1, x_shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x], out, x_init=x_arr, place=place, eps=eps
        )
        gradient_checker.double_grad_check_for_dygraph(
            self.squeeze_wrapper, [x], out, x_init=x_arr, place=place
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
            self.func(p)


class TestUnsqueezeDoubleGradCheck(unittest.TestCase):
    def unsqueeze_wrapper(self, x):
        axes = [1, 2]
        return paddle.unsqueeze(x[0], axes)

    @prog_scope()
    def func(self, place):
        x_shape = [3, 40]
        axes = [1, 2]
        eps = 0.005
        dtype = np.float64

        x = paddle.static.data('x', x_shape, dtype)
        x.persistable = True
        out = paddle.unsqueeze(x, axes)
        x_arr = np.random.uniform(-1, 1, x_shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x], out, x_init=x_arr, place=place, eps=eps
        )
        gradient_checker.double_grad_check_for_dygraph(
            self.unsqueeze_wrapper, [x], out, x_init=x_arr, place=place
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
            self.func(p)


class TestClipDoubleGradCheck(unittest.TestCase):
    def clip_wrapper(self, x):
        return paddle.clip(x[0], min=-1.0, max=1.0)

    @prog_scope()
    def func(self, place):
        x_shape = [2, 4, 10]
        dtype = np.float64

        x = paddle.static.data('x', x_shape, dtype)
        x.persistable = True
        out = paddle.clip(x, min=-1.0, max=1.0)
        x_arr = np.random.uniform(-5.0, 5.0, x_shape).astype(dtype)

        gradient_checker.double_grad_check([x], out, x_init=x_arr, place=place)
        gradient_checker.double_grad_check_for_dygraph(
            self.clip_wrapper, [x], out, x_init=x_arr, place=place
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
            self.func(p)


class TestTransposeDoubleGradCheck(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        x_shape = [3, 40]
        perm = [1, 0]
        dtype = np.float64

        x = paddle.static.data('x', x_shape, dtype)
        x.persistable = True
        out = paddle.transpose(x, perm)
        x_arr = np.random.uniform(-1, 1, x_shape).astype(dtype)

        gradient_checker.double_grad_check([x], out, x_init=x_arr, place=place)

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
            self.func(p)


class TestTransposeDoubleGradCheckCase1(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        x_shape = [2, 3, 4, 5]
        perm = [0, 2, 3, 1]
        dtype = np.float64

        x = paddle.static.data('x', x_shape, dtype)
        x.persistable = True
        out = paddle.transpose(x, perm)
        x_arr = np.random.uniform(-1, 1, x_shape).astype(dtype)

        gradient_checker.double_grad_check([x], out, x_init=x_arr, place=place)

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
            self.func(p)


class TestConstantPadDoubleGradCheck(unittest.TestCase):
    def pad_wrapper(self, x):
        pad = [1, 1, 1, 1]
        return paddle.nn.functional.pad(x[0], pad)

    @prog_scope()
    def func(self, place):
        x_shape = [2, 3, 4, 5]
        pad = [1, 1, 1, 1]
        eps = 0.005
        dtype = np.float64

        x = paddle.static.data('x', x_shape, dtype)
        x.persistable = True
        x.stop_gradient = False
        out = paddle.nn.functional.pad(x, pad)
        x_arr = np.random.uniform(-1, 1, x_shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x], out, x_init=x_arr, place=place, eps=eps
        )
        gradient_checker.double_grad_check_for_dygraph(
            self.pad_wrapper, [x], out, x_init=x_arr, place=place
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
            self.func(p)


class TestConstantPadDoubleGradCheckCase1(TestConstantPadDoubleGradCheck):

    @prog_scope()
    def func(self, place):
        x_shape = [2, 3, 4, 5]
        pad = [1, 0, 1, 0, 1, 0, 1, 0]
        dtype = np.float64

        x = paddle.static.data('x', x_shape, dtype)
        x.persistable = True
        out = paddle.nn.functional.pad(x, pad)
        x_arr = np.random.uniform(-1, 1, x_shape).astype(dtype)

        gradient_checker.double_grad_check([x], out, x_init=x_arr, place=place)


class TestConcatDoubleGradCheck(unittest.TestCase):
    def concat_wrapper(self, x):
        return paddle.concat(x, axis=0)

    @prog_scope()
    def func(self, place):
        x_shape = [2, 3, 4, 5]
        dtype = np.float64

        x1 = paddle.static.data('x', x_shape, dtype)
        x2 = paddle.static.data('x', x_shape, dtype)
        x1.persistable = True
        x1.stop_gradient = False
        x2.persistable = True
        x2.stop_gradient = False
        out = paddle.concat([x1, x2], axis=0)
        x2_arr = np.random.uniform(-1, 1, x_shape).astype(dtype)
        x1_arr = np.random.uniform(-1, 1, x_shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x1, x2], out, x_init=[x1_arr, x2_arr], place=place
        )
        gradient_checker.double_grad_check_for_dygraph(
            self.concat_wrapper,
            [x1, x2],
            out,
            x_init=[x1_arr, x2_arr],
            place=place,
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
            self.func(p)


class TestStackDoubleGradCheck(unittest.TestCase):
    def stack_wrapper(self, x):
        return paddle.stack(x, axis=1)

    @prog_scope()
    def func(self, place):
        x_shape = [2, 3, 4, 5]
        dtype = np.float64

        x1 = paddle.static.data('x', x_shape, dtype)
        x2 = paddle.static.data('x', x_shape, dtype)
        x1.persistable = True
        x1.stop_gradient = False
        x2.persistable = True
        x2.stop_gradient = False
        out = paddle.stack([x1, x2], axis=0)
        x2_arr = np.random.uniform(-1, 1, x_shape).astype(dtype)
        x1_arr = np.random.uniform(-1, 1, x_shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x1, x2], out, x_init=[x1_arr, x2_arr], place=place
        )
        gradient_checker.double_grad_check_for_dygraph(
            self.stack_wrapper,
            [x1, x2],
            out,
            x_init=[x1_arr, x2_arr],
            place=place,
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
            self.func(p)


class TestAvgPool2DDoubleGradCheckCase1(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        input_NCHW = paddle.static.data(
            name="input_NCHW",
            shape=[2, 3, 5, 5],
            dtype="float32",
        )

        input_NCHW.persistable = True
        y = paddle.nn.functional.avg_pool2d(input_NCHW, kernel_size=2)
        x_arr = np.random.uniform(-1, 1, [2, 3, 5, 5]).astype(np.float32)

        gradient_checker.double_grad_check(
            [input_NCHW], y, x_init=x_arr, place=place, eps=0.05
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
            self.func(p)


class TestAvgPool2DDoubleGradCheckCase2(unittest.TestCase):
    def pool2d_wrapper(self, x):
        return paddle.nn.functional.avg_pool2d(
            x[0], kernel_size=2, data_format="NHWC"
        )

    @prog_scope()
    def func(self, place):
        input_NHWC = paddle.static.data(
            name="input_NHWC",
            shape=[2, 5, 5, 3],
            dtype="float32",
        )

        input_NHWC.persistable = True
        y = paddle.nn.functional.avg_pool2d(
            input_NHWC, kernel_size=2, data_format="NHWC"
        )
        x_arr = np.random.uniform(-1, 1, [2, 5, 5, 3]).astype(np.float32)

        gradient_checker.double_grad_check(
            [input_NHWC], y, x_init=x_arr, place=place, eps=0.05
        )

        gradient_checker.double_grad_check_for_dygraph(
            self.pool2d_wrapper, [input_NHWC], y, x_init=x_arr, place=place
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
            self.func(p)


class TestAvgPool2DDoubleGradCheckCase3(unittest.TestCase):
    def pool2d_wrapper(self, x):
        return paddle.nn.functional.avg_pool2d(
            x[0], kernel_size=2, padding=[1, 1]
        )

    @prog_scope()
    def func(self, place):
        input_NCHW = paddle.static.data(
            name="input_NCHW",
            shape=[2, 3, 5, 5],
            dtype="float32",
        )

        input_NCHW.persistable = True
        y = paddle.nn.functional.avg_pool2d(
            input_NCHW, kernel_size=2, padding=[1, 1]
        )
        x_arr = np.random.uniform(-1, 1, [2, 3, 5, 5]).astype(np.float32)

        gradient_checker.double_grad_check(
            [input_NCHW], y, x_init=x_arr, place=place, eps=0.05
        )
        gradient_checker.double_grad_check_for_dygraph(
            self.pool2d_wrapper, [input_NCHW], y, x_init=x_arr, place=place
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
            self.func(p)


class TestAvgPool2DDoubleGradCheckCase4(unittest.TestCase):
    def pool2d_wrapper(self, x):
        return paddle.nn.functional.avg_pool2d(x[0], kernel_size=[4, 4])

    @prog_scope()
    def func(self, place):
        input_NCHW = paddle.static.data(
            name="input_NCHW",
            shape=[2, 3, 5, 5],
            dtype="float32",
        )

        input_NCHW.persistable = True
        y = paddle.nn.functional.avg_pool2d(input_NCHW, kernel_size=[4, 4])
        x_arr = np.random.uniform(-1, 1, [2, 3, 5, 5]).astype(np.float32)

        gradient_checker.double_grad_check(
            [input_NCHW], y, x_init=x_arr, place=place, eps=0.05
        )
        gradient_checker.double_grad_check_for_dygraph(
            self.pool2d_wrapper, [input_NCHW], y, x_init=x_arr, place=place
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
            self.func(p)


if __name__ == "__main__":
    unittest.main()
