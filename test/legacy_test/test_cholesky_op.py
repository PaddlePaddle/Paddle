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
from decorator_helper import prog_scope
from gradient_checker import grad_check
from op_test import OpTest, skip_check_grad_ci

import paddle
from paddle import base
from paddle.base import core
from paddle.base.backward import _as_list


@skip_check_grad_ci(
    reason="The input of cholesky_op should always be symmetric positive-definite. "
    "However, OpTest calculates the numeric gradient of each element in input "
    "via small finite difference, which makes the input no longer symmetric "
    "positive-definite thus can not compute the Cholesky decomposition. "
    "While we can use the gradient_checker.grad_check to perform gradient "
    "check of cholesky_op, since it supports check gradient with a program "
    "and we can construct symmetric positive-definite matrices in the program"
)
class TestCholeskyOp(OpTest):
    def setUp(self):
        self.op_type = "cholesky"
        self.python_api = paddle.cholesky
        self._input_shape = (2, 32, 32)
        self._upper = True
        self.init_config()
        self.trans_dims = list(range(len(self._input_shape) - 2)) + [
            len(self._input_shape) - 1,
            len(self._input_shape) - 2,
        ]
        self.root_data = np.random.random(self._input_shape).astype("float64")
        # construct symmetric positive-definite matrice
        input_data = (
            np.matmul(self.root_data, self.root_data.transpose(self.trans_dims))
            + 1e-05
        )
        output_data = np.linalg.cholesky(input_data).astype("float64")
        if self._upper:
            output_data = output_data.transpose(self.trans_dims)
        self.inputs = {"X": input_data}
        self.attrs = {"upper": self._upper}
        self.outputs = {"Out": output_data}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda() and (not core.is_compiled_with_rocm()):
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

    @prog_scope()
    def func(self, place):
        # use small size since Jacobian gradients is time consuming
        root_data = self.root_data[..., :3, :3]
        prog = base.Program()
        with base.program_guard(prog):
            root = paddle.create_parameter(
                dtype=root_data.dtype, shape=root_data.shape
            )
            root_t = paddle.transpose(root, self.trans_dims)
            x = paddle.matmul(x=root, y=root_t) + 1e-05
            out = paddle.cholesky(x, upper=self.attrs["upper"])
            # check input arguments
            root = _as_list(root)
            out = _as_list(out)

            for v in root:
                v.stop_gradient = False
                v.persistable = True
            for u in out:
                u.stop_gradient = False
                u.persistable = True

            # init variable in startup program
            scope = base.executor.global_scope()
            exe = base.Executor(place)
            exe.run(base.default_startup_program())

            x_init = _as_list(root_data)
            # init inputs if x_init is not None
            if x_init:
                if len(x_init) != len(root):
                    raise ValueError(
                        'len(x_init) (=%d) is not the same'
                        ' as len(x) (= %d)' % (len(x_init), len(root))
                    )
                # init variable in main program
                for var, arr in zip(root, x_init):
                    assert var.shape == arr.shape
                feeds = {k.name: v for k, v in zip(root, x_init)}
                exe.run(prog, feed=feeds, scope=scope)
            grad_check(x=root, y=out, x_init=x_init, place=place, program=prog)

    def init_config(self):
        self._upper = True


class TestCholeskyOpLower(TestCholeskyOp):
    def init_config(self):
        self._upper = False


class TestCholeskyOp2D(TestCholeskyOp):
    def init_config(self):
        self._input_shape = (32, 32)


class TestDygraph(unittest.TestCase):
    def test_dygraph(self):
        if core.is_compiled_with_rocm():
            paddle.disable_static(place=base.CPUPlace())
        else:
            paddle.disable_static()
        a = np.random.rand(3, 3)
        a_t = np.transpose(a, [1, 0])
        x_data = np.matmul(a, a_t) + 1e-03
        x = paddle.to_tensor([x_data])
        out = paddle.cholesky(x, upper=False)


class TestCholeskySingularAPI(unittest.TestCase):
    def setUp(self):
        self.places = [base.CPUPlace()]
        if core.is_compiled_with_cuda() and (not core.is_compiled_with_rocm()):
            self.places.append(base.CUDAPlace(0))

    def check_static_result(self, place, with_out=False):
        with base.program_guard(base.Program(), base.Program()):
            input = paddle.static.data(
                name="input", shape=[4, 4], dtype="float64"
            )
            result = paddle.cholesky(input)

            input_np = np.zeros([4, 4]).astype("float64")

            exe = base.Executor(place)
            try:
                fetches = exe.run(
                    base.default_main_program(),
                    feed={"input": input_np},
                    fetch_list=[result],
                )
            except RuntimeError as ex:
                print("The mat is singular")
            except ValueError as ex:
                print("The mat is singular")

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        for place in self.places:
            with base.dygraph.guard(place):
                input_np = np.array(
                    [
                        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                        [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                    ]
                ).astype("float64")
                input = base.dygraph.to_variable(input_np)
                try:
                    result = paddle.cholesky(input)
                except RuntimeError as ex:
                    print("The mat is singular")
                except ValueError as ex:
                    print("The mat is singular")


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
