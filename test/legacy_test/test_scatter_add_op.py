#  Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16
from utils import static_guard

import paddle
from paddle import base
from paddle.base import core
from paddle.base.dygraph.base import switch_to_static_graph


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "core is not compiled with CUDA",
)
class TestScatterAddOp(OpTest):
    def setUp(self):
        self.op_type = "scatter_add"
        self.python_api = paddle.scatter_add
        self.public_python_api = paddle.scatter_add
        self._set_dtype()
        self.if_enable_cinn()
        target_dtype = "float16" if self.dtype == np.float16 else "float32"
        ref_np = np.ones((10, 50)).astype(target_dtype)
        updates_np = np.random.random((10, 50)).astype(target_dtype)

        index_np = np.random.choice(
            np.arange(ref_np.shape[0]),
            size=(updates_np.shape[0],),
            replace=False,
        ).astype("int32")

        # randomly mapping index into equivalent negative index(mod ref_np.shape[0])
        # to test for negative index
        random_negative_mask = (np.random.rand(index_np.shape[0]) > 0.5).astype(
            "bool"
        )
        index_np[random_negative_mask] -= ref_np.shape[0]

        output_np = np.copy(ref_np)
        np.add.at(output_np, index_np, updates_np)
        if self.dtype == np.uint16:
            ref_np = convert_float_to_uint16(ref_np)
            updates_np = convert_float_to_uint16(updates_np)
            output_np = convert_float_to_uint16(output_np)
        self.inputs = {'x': ref_np, 'index': index_np, 'updates': updates_np}
        self.outputs = {'out': output_np}

    def if_enable_cinn(self):
        pass

    def _set_dtype(self):
        self.dtype = np.float32
        self.__class__.exist_fp64_check_grad = True

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad(self):
        self.check_grad(
            ["x", "updates"],
            "out",
            check_prim=False,
            check_pir=True,
            check_prim_pir=False,
            max_relative_error=0.008,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "core is not compiled with CUDA",
)
class TestScatterFP16Op(TestScatterAddOp):
    def _set_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestScatterBF16Op(TestScatterAddOp):
    def _set_dtype(self):
        self.dtype = np.uint16

    def if_enable_cinn(self):
        self.enable_cinn = False

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, check_pir=True)

    def test_check_grad(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place,
                ['x', 'updates'],
                'out',
                check_prim=False,
                check_pir=True,
                check_prim_pir=False,
            )


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "core is not compiled with CUDA",
)
class TestScatterFP64Op(TestScatterAddOp):
    def _set_dtype(self):
        self.dtype = np.float64
        self.__class__.exist_fp64_check_grad = True


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "core is not compiled with CUDA",
)
class TestScatterAPI(unittest.TestCase):
    def setUp(self):
        self.places = []
        self.places.append(base.CUDAPlace(0))
        self.executed_api()

    def executed_api(self):
        self.scatter = paddle.scatter_add

    def check_static_result(self, place):
        with static_guard():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                input = paddle.static.data(
                    name="input", shape=[3, 2], dtype="float64"
                )
                index = paddle.static.data(
                    name="index", shape=[4], dtype="int64"
                )
                updates = paddle.static.data(
                    name="updates", shape=[4, 2], dtype="float64"
                )
                result = self.scatter(input, index, updates)

                input_data = np.array([[1, 1], [2, 2], [3, 3]]).astype(
                    np.float64
                )
                index_data = np.array([2, 1, 0, 1]).astype(np.int64)
                updates_data = np.array(
                    [[1, 1], [2, 2], [3, 3], [4, 4]]
                ).astype(np.float64)

                exe = paddle.static.Executor(place)
                fetches = exe.run(
                    paddle.static.default_main_program(),
                    feed={
                        "input": input_data,
                        "index": index_data,
                        "updates": updates_data,
                    },
                    fetch_list=[result],
                )
                self.assertEqual(
                    (
                        fetches[0]
                        == np.array([[4.0, 4.0], [8.0, 8.0], [4.0, 4.0]])
                    ).all(),
                    True,
                )

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        for place in self.places:
            with base.dygraph.guard(place):
                x_data = np.array([[1, 1], [2, 2], [3, 3]]).astype(np.float64)
                index_data = np.array([2, 1, 0, 1]).astype(np.int64)
                updates_data = np.array(
                    [[1, 1], [2, 2], [3, 3], [4, 4]]
                ).astype(np.float64)

                x = paddle.to_tensor(x_data)
                index = paddle.to_tensor(index_data)
                updates = paddle.to_tensor(updates_data)

                output1 = self.scatter(x, index, updates)
                self.assertEqual(
                    (
                        output1.numpy()
                        == np.array([[4.0, 4.0], [8.0, 8.0], [4.0, 4.0]])
                    ).all(),
                    True,
                )

    def test_large_data(self):
        if os.name == "nt" or not paddle.is_compiled_with_cuda():
            return

        x = np.random.rand(183826, 256).astype("float32")
        index = np.ones(10759233, dtype="int64")
        updates = np.ones(shape=[10759233, 256], dtype="float32")

        def test_dygraph():
            with base.dygraph.guard():
                gpu_out = paddle.scatter_add(
                    paddle.to_tensor(x),
                    paddle.to_tensor(index),
                    paddle.to_tensor(updates),
                )
                return gpu_out.numpy()

        @switch_to_static_graph
        def test_static_graph():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                scope = paddle.static.Scope()
                with paddle.static.scope_guard(scope):
                    x_t = paddle.static.data(
                        name="x", dtype=x.dtype, shape=x.shape
                    )
                    index_t = paddle.static.data(
                        name="index", dtype=index.dtype, shape=index.shape
                    )
                    updates_t = paddle.static.data(
                        name="updates", dtype=updates.dtype, shape=updates.shape
                    )
                    out_t = paddle.scatter_add(x_t, index_t, updates_t)
                    feed = {
                        x_t.name: x,
                        index_t.name: index,
                        updates_t.name: updates,
                    }
                    fetch = [out_t]
                    gpu_exe = paddle.static.Executor(paddle.CUDAPlace(0))
                    gpu_value = gpu_exe.run(feed=feed, fetch_list=fetch)[0]
                    scope._remove_from_pool()
                    return gpu_value

        def test_pir_static_graph():
            with paddle.pir_utils.IrGuard():
                return test_static_graph()

        dy_out = test_dygraph()
        np.testing.assert_array_equal(dy_out, test_static_graph())
        np.testing.assert_array_equal(dy_out, test_pir_static_graph())


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "core is not compiled with CUDA",
)
class TestScatterAddInplace(unittest.TestCase):
    def test_api_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
            index = paddle.to_tensor([0, 1])
            updates = paddle.to_tensor([[1.0, 1.0], [1.0, 1.0]])
            target = paddle.scatter_add(x, index, updates)
            paddle.scatter_add_(x, index, updates)
            np.testing.assert_array_equal(target.numpy(), x.numpy())

        run(paddle.CUDAPlace(0))


if __name__ == '__main__':
    unittest.main()
