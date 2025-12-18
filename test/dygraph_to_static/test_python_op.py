# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from dygraph_to_static_utils import (
    Dy2StTestBase,
    static_guard,
)

import paddle
from paddle import Tensor
from paddle.jit import sot


@paddle.static.register_op(
    name="fn_with_breakgraph",
    infer_meta=lambda x, y: paddle.static.MetaTensor(
        dtype=x.dtype, shape=x.shape
    ),
    input_names=["x", "y"],
    output_names=["out"],
    inplace_map={},
)
def fn_with_breakgraph(x: Tensor, y: Tensor) -> Tensor:
    x = x + 1
    sot.psdb.breakgraph()
    y = y + 1
    return x + y


@paddle.static.register_op(
    name="fn_with_numpy_operation",
    infer_meta=lambda x, y: paddle.static.MetaTensor(
        dtype=paddle.int32, shape=x.shape[:-1]
    ),
    input_names=["x", "y"],
    output_names=["out"],
)
def fn_with_numpy_operation(x: Tensor, y: Tensor) -> Tensor:
    x_np = x.numpy()
    y_np = y.numpy()
    x_np_reduce = x_np.sum(axis=-1)
    y_np_reduce = y_np.sum(axis=-1)
    return paddle.to_tensor(x_np_reduce + y_np_reduce).cast(paddle.int32)


class PythonOpTestMixin:
    def run_in_dygraph(self):
        return self.fn(**self.inputs)

    @static_guard()
    def run_in_static(self):
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            input_values = {
                k: paddle.static.data(name=k, shape=v.shape, dtype=v.dtype)
                for k, v in self.inputs.items()
            }
            out_value = self.fn(**input_values)
            exe = paddle.static.Executor()
        (out,) = exe.run(
            main_program,
            feed={k: v.numpy() for k, v in self.inputs.items()},
            fetch_list=[out_value],
        )
        return out

    def test_dy_st(self):
        np.testing.assert_allclose(self.run_in_dygraph(), self.run_in_static())


class TestFnWithBreakgraph(unittest.TestCase, PythonOpTestMixin):
    def setUp(self):
        self.fn = fn_with_breakgraph
        self.inputs = {
            "x": paddle.randn([2, 3, 4]),
            "y": paddle.randn([2, 3, 4]),
        }


class TestFnWithNumPyOperation(unittest.TestCase, PythonOpTestMixin):
    def setUp(self):
        self.fn = fn_with_numpy_operation
        self.inputs = {
            "x": paddle.randn([7, 8, 9]),
            "y": paddle.randn([7, 8, 9]),
        }


def fn_use_2_register_op(x: Tensor, y: Tensor) -> Tensor:
    z1 = fn_with_breakgraph(x, y)
    z2 = fn_with_numpy_operation(x, y)
    out = z1 * 100 + z2.unsqueeze(axis=-1).astype(paddle.float32)
    return out


class TestToStatic(Dy2StTestBase):
    def test_to_static_use_2_op(self):
        x = paddle.randn([2, 3, 4])
        y = paddle.randn([2, 3, 4])
        fn = fn_use_2_register_op
        dy_out = fn(x, y)
        static_fn = paddle.jit.to_static(fn)
        st_out = static_fn(x, y)
        np.testing.assert_allclose(dy_out, st_out)


if __name__ == "__main__":
    unittest.main()
