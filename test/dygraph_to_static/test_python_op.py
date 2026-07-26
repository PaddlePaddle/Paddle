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
from collections.abc import Callable

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


@paddle.static.register_op(
    name="fn_with_constant",
    infer_meta=lambda x, c: paddle.static.MetaTensor(
        dtype=x.dtype, shape=[np.prod(x.shape).item()]
    ),
    input_names=["x"],
    output_names=["out"],
)
def fn_with_constant(x: paddle.Tensor, c: int):
    return paddle.to_tensor(x.flatten() + c)


@paddle.static.register_op(
    name="fn_inplace_add_y",
    infer_meta=lambda x, y: paddle.static.MetaTensor(
        dtype=x.dtype, shape=x.shape
    ),
    input_names=["x", "y"],
    output_names=["out"],
    inplace_map={"x": "out"},  # output "out" reuses the buffer of input "x"
)
def fn_inplace_add_y(x: Tensor, y: Tensor) -> Tensor:
    x.add_(y)  # modify input in place
    return x


class PythonOpTestMixin:
    inputs: dict[str, paddle.Tensor]
    constants: dict[str, object]
    fn: Callable[..., paddle.Tensor]

    def run_in_dygraph(self):
        return self.fn(**self.inputs, **self.constants)

    @static_guard()
    def run_in_static(self):
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            input_values = {
                k: paddle.static.data(name=k, shape=v.shape, dtype=v.dtype)
                for k, v in self.inputs.items()
            }
            out_value = self.fn(**input_values, **self.constants)
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
        self.constants = {}


class TestFnWithNumPyOperation(unittest.TestCase, PythonOpTestMixin):
    def setUp(self):
        self.fn = fn_with_numpy_operation
        self.inputs = {
            "x": paddle.randn([7, 8, 9]),
            "y": paddle.randn([7, 8, 9]),
        }
        self.constants = {}


class TestFnWithConstant1(unittest.TestCase, PythonOpTestMixin):
    def setUp(self):
        self.fn = fn_with_constant
        self.inputs = {
            "x": paddle.randn([4, 5, 6]),
        }
        self.constants = {"c": -1}


class TestFnWithConstant2(unittest.TestCase, PythonOpTestMixin):
    def setUp(self):
        self.fn = fn_with_constant
        self.inputs = {
            "x": paddle.randn([4, 5, 6]),
        }
        self.constants = {"c": -2}  # Note that hash(-1) == hash(-2)


class TestFnInplaceOutput(unittest.TestCase):
    """Regression test for the inplace result remap of Python custom ops.

    Exercises both PR fixes together: ``run_python_op`` must populate
    ``input_shapes``/``input_dtypes`` for the inplace output, and the PIR
    lowering must remap ``map_value_pair[result]`` back to the input value so
    that downstream ops (out + 1) and ``fetch_list=[out]`` observe the correct
    inplace output. The ``shape=[]`` case guards the rank-0 scalar path, which
    must not be misclassified as an optional-None output.
    """

    # Include a rank-0 scalar (shape=[]) to cover the empty-shape edge case.
    shapes = [[2, 3, 4], []]

    @staticmethod
    def _randn(shape):
        arr = np.random.randn(*shape) if shape else np.random.randn()
        return np.array(arr, dtype="float32")

    def run_in_dygraph(self, x_np, y_np):
        x = paddle.to_tensor(x_np)
        y = paddle.to_tensor(y_np)
        out = fn_inplace_add_y(x, y)
        consumed = out + 1  # continue consuming the inplace output
        return out.numpy(), consumed.numpy()

    @static_guard()
    def run_in_static(self, x_np, y_np):
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x = paddle.static.data(name="x", shape=x_np.shape, dtype="float32")
            y = paddle.static.data(name="y", shape=y_np.shape, dtype="float32")
            out = fn_inplace_add_y(x, y)
            consumed = out + 1  # continue consuming the inplace output
            exe = paddle.static.Executor()
        # fetch the inplace output directly and the downstream consumer
        out_val, consumed_val = exe.run(
            main_program,
            feed={"x": x_np, "y": y_np},
            fetch_list=[out, consumed],
        )
        return out_val, consumed_val

    def test_inplace_output_semantics(self):
        for shape in self.shapes:
            with self.subTest(shape=shape):
                x_np = self._randn(shape)
                y_np = self._randn(shape)
                expected_out = x_np + y_np
                expected_consumed = expected_out + 1

                st_out, st_consumed = self.run_in_static(x_np, y_np)
                np.testing.assert_allclose(
                    st_out, expected_out, rtol=1e-6, atol=1e-6
                )
                np.testing.assert_allclose(
                    st_consumed, expected_consumed, rtol=1e-6, atol=1e-6
                )

                dy_out, dy_consumed = self.run_in_dygraph(x_np, y_np)
                np.testing.assert_allclose(
                    dy_out, expected_out, rtol=1e-6, atol=1e-6
                )
                np.testing.assert_allclose(
                    dy_consumed, expected_consumed, rtol=1e-6, atol=1e-6
                )


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
