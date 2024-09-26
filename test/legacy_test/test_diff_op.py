#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle
from paddle import base, static
from paddle.base import core


class TestDiffOp(unittest.TestCase):
    def set_args(self):
        self.input = np.array([1, 4, 5, 2]).astype('float32')
        self.n = 1
        self.axis = -1
        self.prepend = None
        self.append = None

    def get_output(self):
        if self.prepend is not None and self.append is not None:
            self.output = np.diff(
                self.input,
                n=self.n,
                axis=self.axis,
                prepend=self.prepend,
                append=self.append,
            )
        elif self.prepend is not None:
            self.output = np.diff(
                self.input, n=self.n, axis=self.axis, prepend=self.prepend
            )
        elif self.append is not None:
            self.output = np.diff(
                self.input, n=self.n, axis=self.axis, append=self.append
            )
        else:
            self.output = np.diff(self.input, n=self.n, axis=self.axis)

    def setUp(self):
        self.set_args()
        self.get_output()
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def func_dygraph(self):
        for place in self.places:
            paddle.disable_static()
            x = paddle.to_tensor(self.input, place=place)
            if self.prepend is not None:
                self.prepend = paddle.to_tensor(self.prepend, place=place)
            if self.append is not None:
                self.append = paddle.to_tensor(self.append, place=place)
            out = paddle.diff(
                x,
                n=self.n,
                axis=self.axis,
                prepend=self.prepend,
                append=self.append,
            )
            self.assertTrue((out.numpy() == self.output).all(), True)

    def test_dygraph(self):
        self.setUp()
        self.func_dygraph()

    def test_static(self):
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
        for place in places:
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data(
                    name="input", shape=self.input.shape, dtype=self.input.dtype
                )
                has_pend = False
                prepend = None
                append = None
                if self.prepend is not None:
                    has_pend = True
                    prepend = paddle.static.data(
                        name="prepend",
                        shape=self.prepend.shape,
                        dtype=self.prepend.dtype,
                    )
                if self.append is not None:
                    has_pend = True
                    append = paddle.static.data(
                        name="append",
                        shape=self.append.shape,
                        dtype=self.append.dtype,
                    )

                exe = static.Executor(place)
                out = paddle.diff(
                    x, n=self.n, axis=self.axis, prepend=prepend, append=append
                )

                fetches = exe.run(
                    feed={
                        "input": self.input,
                        "prepend": self.prepend,
                        "append": self.append,
                    },
                    fetch_list=[out],
                )
                self.assertTrue((fetches[0] == self.output).all(), True)

    def func_grad(self):
        for place in self.places:
            x = paddle.to_tensor(self.input, place=place, stop_gradient=False)
            if self.prepend is not None:
                self.prepend = paddle.to_tensor(self.prepend, place=place)
            if self.append is not None:
                self.append = paddle.to_tensor(self.append, place=place)
            out = paddle.diff(
                x,
                n=self.n,
                axis=self.axis,
                prepend=self.prepend,
                append=self.append,
            )
            try:
                out.backward()
                x_grad = x.grad
            except:
                raise RuntimeError("Check Diff Gradient Failed")

    def test_grad(self):
        self.setUp()
        self.func_grad()


class TestDiffOpN(TestDiffOp):
    def set_args(self):
        self.input = np.array([1, 4, 5, 2]).astype('float32')
        self.n = 2
        self.axis = 0
        self.prepend = None
        self.append = None


class TestDiffOpNAxis(TestDiffOp):
    def set_args(self):
        self.input = np.array([[1, 4, 5, 2], [1, 5, 4, 2]]).astype('float32')
        self.n = 2
        self.axis = 1
        self.prepend = None
        self.append = None


class TestDiffOpNPrepend(TestDiffOp):
    def set_args(self):
        self.input = np.array([[1, 4, 5, 2], [1, 5, 4, 2]]).astype('float32')
        self.n = 2
        self.axis = -1
        self.prepend = np.array([[2, 3, 4, 11], [1, 3, 5, 10]]).astype(
            'float32'
        )
        self.append = None


class TestDiffOpNAppend(TestDiffOp):
    def set_args(self):
        self.input = np.array([[1, 4, 5, 2], [1, 5, 4, 2]]).astype('float32')
        self.n = 2
        self.axis = -1
        self.prepend = None
        self.append = np.array([[2, 3, 4, 11], [1, 3, 5, 10]]).astype('float32')


class TestDiffOpNPreAppend(TestDiffOp):
    def set_args(self):
        self.input = np.array([[1, 4, 5, 2], [1, 5, 4, 2]]).astype('float32')
        self.n = 2
        self.axis = -1
        self.prepend = np.array([[2, 3, 4, 11], [1, 3, 5, 10]]).astype(
            'float32'
        )
        self.append = np.array([[2, 3, 4, 11], [1, 3, 5, 10]]).astype('float32')


class TestDiffOpNPrependAxis(TestDiffOp):
    def set_args(self):
        self.input = np.array([[1, 4, 5, 2], [1, 5, 4, 2]]).astype('float32')
        self.n = 2
        self.axis = 0
        self.prepend = np.array([[2, 3, 4, 11], [1, 3, 5, 10]]).astype(
            'float32'
        )
        self.append = None


class TestDiffOpNAppendAxis(TestDiffOp):
    def set_args(self):
        self.input = np.array([[1, 4, 5, 2], [1, 5, 4, 2]]).astype('float32')
        self.n = 2
        self.axis = 0
        self.prepend = None
        self.append = np.array([[2, 3, 4, 11], [1, 3, 5, 10]]).astype('float32')


class TestDiffOpNPreAppendAxis(TestDiffOp):
    def set_args(self):
        self.input = np.array([[1, 4, 5, 2], [1, 5, 4, 2]]).astype('float32')
        self.n = 2
        self.axis = 0
        self.prepend = np.array([[2, 3, 4, 11], [1, 3, 5, 10]]).astype(
            'float32'
        )
        self.append = np.array([[2, 3, 4, 11], [1, 3, 5, 10]]).astype('float32')


class TestDiffOpAxis(TestDiffOp):
    def set_args(self):
        self.input = np.array([[1, 4, 5, 2], [1, 5, 4, 2]]).astype('float32')
        self.n = 1
        self.axis = 0
        self.prepend = None
        self.append = None


class TestDiffOpNDim(TestDiffOp):
    def set_args(self):
        self.input = np.random.rand(10, 10).astype('float32')
        self.n = 1
        self.axis = -1
        self.prepend = None
        self.append = None


class TestDiffOpBool(TestDiffOp):
    def set_args(self):
        self.input = np.array([0, 1, 1, 0, 1, 0]).astype('bool')
        self.n = 1
        self.axis = -1
        self.prepend = None
        self.append = None


class TestDiffOpPrepend(TestDiffOp):
    def set_args(self):
        self.input = np.array([[1, 4, 5, 2], [1, 5, 4, 2]]).astype('float32')
        self.n = 1
        self.axis = -1
        self.prepend = np.array([[2, 3, 4], [1, 3, 5]]).astype('float32')
        self.append = None


class TestDiffOpPrependAxis(TestDiffOp):
    def set_args(self):
        self.input = np.array([[1, 4, 5, 2], [1, 5, 4, 2]]).astype('float32')
        self.n = 1
        self.axis = 0
        self.prepend = np.array(
            [[0, 2, 3, 4], [1, 3, 5, 7], [2, 5, 8, 0]]
        ).astype('float32')
        self.append = None


class TestDiffOpAppend(TestDiffOp):
    def set_args(self):
        self.input = np.array([[1, 4, 5, 2], [1, 5, 4, 2]]).astype('float32')
        self.n = 1
        self.axis = -1
        self.prepend = None
        self.append = np.array([[2, 3, 4], [1, 3, 5]]).astype('float32')


class TestDiffOpAppendAxis(TestDiffOp):
    def set_args(self):
        self.input = np.array([[1, 4, 5, 2], [1, 5, 4, 2]]).astype('float32')
        self.n = 1
        self.axis = 0
        self.prepend = None
        self.append = np.array([[2, 3, 4, 1]]).astype('float32')


class TestDiffOpPreAppend(TestDiffOp):
    def set_args(self):
        self.input = np.array([[1, 4, 5, 2], [1, 5, 4, 2]]).astype('float32')
        self.n = 1
        self.axis = -1
        self.prepend = np.array([[0, 4], [5, 9]]).astype('float32')
        self.append = np.array([[2, 3, 4], [1, 3, 5]]).astype('float32')


class TestDiffOpPreAppendAxis(TestDiffOp):
    def set_args(self):
        self.input = np.array([[1, 4, 5, 2], [1, 5, 4, 2]]).astype('float32')
        self.n = 1
        self.axis = 0
        self.prepend = np.array([[0, 4, 5, 9], [5, 9, 2, 3]]).astype('float32')
        self.append = np.array([[2, 3, 4, 7], [1, 3, 5, 6]]).astype('float32')


class TestDiffOpFp16(TestDiffOp):

    def test_fp16_with_gpu(self):
        paddle.enable_static()
        if paddle.base.core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                input = np.random.random([4, 4]).astype("float16")
                x = paddle.static.data(
                    name="input", shape=[4, 4], dtype="float16"
                )
                exe = paddle.static.Executor(place)
                out = paddle.diff(
                    x,
                    n=self.n,
                    axis=self.axis,
                    prepend=self.prepend,
                    append=self.append,
                )
                fetches = exe.run(
                    feed={
                        "input": input,
                    },
                    fetch_list=[out],
                )
        paddle.disable_static()


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
