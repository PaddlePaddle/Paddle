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

import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core
from paddle.fluid.framework import _enable_legacy_dygraph
_enable_legacy_dygraph()


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
                append=self.append)
        elif self.prepend is not None:
            self.output = np.diff(
                self.input, n=self.n, axis=self.axis, prepend=self.prepend)
        elif self.append is not None:
            self.output = np.diff(
                self.input, n=self.n, axis=self.axis, append=self.append)
        else:
            self.output = np.diff(self.input, n=self.n, axis=self.axis)

    def setUp(self):
        self.set_args()
        self.get_output()
        self.places = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_dygraph(self):
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
                append=self.append)
            self.assertTrue((out.numpy() == self.output).all(), True)

    def test_static(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for place in places:
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                x = paddle.fluid.data(
                    name="input",
                    shape=self.input.shape,
                    dtype=self.input.dtype)
                has_pend = False
                prepend = None
                append = None
                if self.prepend is not None:
                    has_pend = True
                    prepend = paddle.fluid.data(
                        name="prepend",
                        shape=self.prepend.shape,
                        dtype=self.prepend.dtype)
                if self.append is not None:
                    has_pend = True
                    append = paddle.fluid.data(
                        name="append",
                        shape=self.append.shape,
                        dtype=self.append.dtype)

                exe = fluid.Executor(place)
                out = paddle.diff(
                    x, n=self.n, axis=self.axis, prepend=prepend, append=append)
                fetches = exe.run(fluid.default_main_program(),
                                  feed={
                                      "input": self.input,
                                      "prepend": self.prepend,
                                      "append": self.append
                                  },
                                  fetch_list=[out])
                self.assertTrue((fetches[0] == self.output).all(), True)

    def test_grad(self):
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
                append=self.append)
            try:
                out.backward()
                x_grad = x.grad
            except:
                raise RuntimeError("Check Diff Gradient Failed")


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
            [[0, 2, 3, 4], [1, 3, 5, 7], [2, 5, 8, 0]]).astype('float32')
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


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
