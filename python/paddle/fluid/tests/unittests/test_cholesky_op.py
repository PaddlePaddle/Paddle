#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core
from op_test import OpTest, skip_check_grad_ci
from gradient_checker import grad_check
from decorator_helper import prog_scope


@skip_check_grad_ci(
    reason=
    "The input of cholesky_op should always be symmetric positive-definite. "
    "However, OpTest calculates the numeric gradient of each element in input "
    "via small finite difference, which makes the input no longer symmetric "
    "positive-definite thus can not compute the Cholesky decomposition. "
    "While we can use the gradient_checker.grad_check to perform gradient "
    "check of cholesky_op, since it supports check gradient with a program "
    "and we can construct symmetric positive-definite matrices in the program")
class TestCholeskyOp(OpTest):

    def setUp(self):
        self.op_type = "cholesky"
        self._input_shape = (2, 32, 32)
        self._upper = True
        self.init_config()
        self.trans_dims = list(range(len(self._input_shape) - 2)) + [
            len(self._input_shape) - 1,
            len(self._input_shape) - 2
        ]
        self.root_data = np.random.random(self._input_shape).astype("float64")
        # construct symmetric positive-definite matrice
        input_data = np.matmul(
            self.root_data, self.root_data.transpose(self.trans_dims)) + 1e-05
        output_data = np.linalg.cholesky(input_data).astype("float64")
        if self._upper:
            output_data = output_data.transpose(self.trans_dims)
        self.inputs = {"X": input_data}
        self.attrs = {"upper": self._upper}
        self.outputs = {"Out": output_data}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda() and (not core.is_compiled_with_rocm()):
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)

    @prog_scope()
    def func(self, place):
        # use small size since Jacobian gradients is time consuming
        root_data = self.root_data[..., :3, :3]
        prog = fluid.Program()
        with fluid.program_guard(prog):
            root = layers.create_parameter(dtype=root_data.dtype,
                                           shape=root_data.shape)
            root_t = layers.transpose(root, self.trans_dims)
            x = layers.matmul(x=root, y=root_t) + 1e-05
            out = paddle.cholesky(x, upper=self.attrs["upper"])
            grad_check(root, out, x_init=root_data, place=place)

    def init_config(self):
        self._upper = True


class TestCholeskyOpLower(TestCholeskyOp):

    def init_config(self):
        self._upper = False


class TestCholeskyOp2D(TestCholeskyOp):

    def init_config(self):
        self._input_shape = (64, 64)


class TestDygraph(unittest.TestCase):

    def test_dygraph(self):
        if core.is_compiled_with_rocm():
            paddle.disable_static(place=fluid.CPUPlace())
        else:
            paddle.disable_static()
        a = np.random.rand(3, 3)
        a_t = np.transpose(a, [1, 0])
        x_data = np.matmul(a, a_t) + 1e-03
        x = paddle.to_tensor(x_data)
        out = paddle.cholesky(x, upper=False)


class TestCholeskySingularAPI(unittest.TestCase):

    def setUp(self):
        self.places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda() and (not core.is_compiled_with_rocm()):
            self.places.append(fluid.CUDAPlace(0))

    def check_static_result(self, place, with_out=False):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input = fluid.data(name="input", shape=[4, 4], dtype="float64")
            result = paddle.cholesky(input)

            input_np = np.zeros([4, 4]).astype("float64")

            exe = fluid.Executor(place)
            try:
                fetches = exe.run(fluid.default_main_program(),
                                  feed={"input": input_np},
                                  fetch_list=[result])
            except RuntimeError as ex:
                print("The mat is singular")
                pass
            except ValueError as ex:
                print("The mat is singular")
                pass

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        for place in self.places:
            with fluid.dygraph.guard(place):
                input_np = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                     [[10, 11, 12], [13, 14, 15],
                                      [16, 17, 18]]]).astype("float64")
                input = fluid.dygraph.to_variable(input_np)
                try:
                    result = paddle.cholesky(input)
                except RuntimeError as ex:
                    print("The mat is singular")
                    pass
                except ValueError as ex:
                    print("The mat is singular")
                    pass


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
