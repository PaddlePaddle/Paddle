#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import os
import paddle
import paddle.fluid as fluid
from op_test import OpTest
import paddle.fluid.core as core
from paddle.fluid.dygraph.base import switch_to_static_graph


class TestScatterOp(OpTest):
    def setUp(self):
        self.op_type = "scatter"
        self.python_api = paddle.scatter
        ref_np = np.ones((3, 50)).astype("float32")
        index_np = np.array([1, 2]).astype("int32")
        updates_np = np.random.random((2, 50)).astype("float32")
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output(check_eager=False)

    def test_check_grad(self):
        self.check_grad(["X", "Updates"], "Out", check_eager=False)


class TestScatterOp0(OpTest):
    def setUp(self):
        self.op_type = "scatter"
        self.python_api = paddle.scatter
        ref_np = np.ones((3, 3)).astype("float32")
        index_np = np.array([1, 2]).astype("int32")
        updates_np = np.random.random((2, 3)).astype("float32")
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.attrs = {'overwrite': True}
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output(check_eager=False)

    def test_check_grad(self):
        self.check_grad(["X", "Updates"], "Out", check_eager=False)


class TestScatterOp1(OpTest):
    def setUp(self):
        self.op_type = "scatter"
        self.python_api = paddle.scatter
        ref_np = np.ones((3, 3)).astype("float32")
        zeros_np = np.zeros([2, 3]).astype('float32')
        index_np = np.array([1, 1]).astype("int32")
        updates_np = np.random.random((2, 3)).astype("float32")
        output_np = np.copy(ref_np)
        output_np[index_np] = zeros_np
        for i in range(0, len(index_np)):
            output_np[index_np[i]] += updates_np[i]
        self.attrs = {'overwrite': False}
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output(check_eager=False)

    def test_check_grad(self):
        self.check_grad(["X", "Updates"], "Out", check_eager=False)


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestScatterOp2(OpTest):
    def setUp(self):
        self.op_type = "scatter"
        self.python_api = paddle.scatter
        ref_np = np.ones((3, 3)).astype("float32")
        index_np = np.array([1, 2]).astype("int32")
        updates_np = np.random.random((2, 3)).astype("float32")
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-3, check_eager=False)

    def test_check_grad(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place, ['X', 'Updates'], 'Out', check_eager=False
            )


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestScatterOp3(OpTest):
    def setUp(self):
        self.op_type = "scatter"
        self.python_api = paddle.scatter
        ref_np = np.ones((3, 3)).astype("float32")
        zeros_np = np.zeros([2, 3]).astype('float32')
        index_np = np.array([1, 1]).astype("int32")
        updates_np = np.random.random((2, 3)).astype("float32")
        output_np = np.copy(ref_np)
        output_np[index_np] = zeros_np
        for i in range(0, len(index_np)):
            output_np[index_np[i]] += updates_np[i]
        self.attrs = {'overwrite': False}
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-3, check_eager=False)

    def test_check_grad(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place, ['X', 'Updates'], 'Out', check_eager=False
            )


class TestScatterOp4(OpTest):
    def setUp(self):
        self.op_type = "scatter"
        self.python_api = paddle.scatter
        ref_np = np.ones((3, 3)).astype("float32")
        index_np = np.array([1, 2]).astype("int64")
        updates_np = np.random.random((2, 3)).astype("float32")
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output(check_eager=False)

    def test_check_grad(self):
        self.check_grad(['X', 'Updates'], 'Out', check_eager=False)


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestScatterOp5(OpTest):
    def setUp(self):
        self.op_type = "scatter"
        self.python_api = paddle.scatter
        ref_np = np.ones((3, 3)).astype("float32")
        index_np = np.array([1, 2]).astype("int64")
        updates_np = np.random.random((2, 3)).astype("float32")
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-3, check_eager=False)

    def test_check_grad(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place, ['X', 'Updates'], 'Out', check_eager=False
            )


class TestScatterOp6(OpTest):
    def setUp(self):
        self.op_type = "scatter"
        self.python_api = paddle.scatter
        ref_np = np.ones((3, 50)).astype("float32")
        index_np = np.array([[1], [2]]).astype("int32")
        updates_np = np.random.random((2, 50)).astype("float32")
        output_np = np.copy(ref_np)
        output_np[np.array([1, 2]).astype("int32")] = updates_np
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output(check_eager=False)

    def test_check_grad(self):
        self.check_grad(["X", "Updates"], "Out", check_eager=False)


class TestScatterAPI(unittest.TestCase):
    def setUp(self):
        self.places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(fluid.CUDAPlace(0))
        self.executed_api()

    def executed_api(self):
        self.scatter = paddle.scatter

    def check_static_result(self, place):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input = fluid.data(name="input", shape=[3, 2], dtype="float64")
            index = fluid.data(name="index", shape=[4], dtype="int64")
            updates = fluid.data(name="updates", shape=[4, 2], dtype="float64")
            result = self.scatter(input, index, updates, False)

            input_data = np.array([[1, 1], [2, 2], [3, 3]]).astype(np.float64)
            index_data = np.array([2, 1, 0, 1]).astype(np.int64)
            updates_data = np.array([[1, 1], [2, 2], [3, 3], [4, 4]]).astype(
                np.float64
            )

            exe = fluid.Executor(place)
            fetches = exe.run(
                fluid.default_main_program(),
                feed={
                    "input": input_data,
                    "index": index_data,
                    "updates": updates_data,
                },
                fetch_list=[result],
            )
            self.assertEqual(
                (
                    fetches[0] == np.array([[3.0, 3.0], [6.0, 6.0], [1.0, 1.0]])
                ).all(),
                True,
            )

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        for place in self.places:
            with fluid.dygraph.guard(place):
                x_data = np.array([[1, 1], [2, 2], [3, 3]]).astype(np.float64)
                index_data = np.array([2, 1, 0, 1]).astype(np.int64)
                updates_data = np.array(
                    [[1, 1], [2, 2], [3, 3], [4, 4]]
                ).astype(np.float64)

                x = fluid.dygraph.to_variable(x_data)
                index = fluid.dygraph.to_variable(index_data)
                updates = fluid.dygraph.to_variable(updates_data)

                output1 = self.scatter(x, index, updates, overwrite=False)
                self.assertEqual(
                    (
                        output1.numpy()
                        == np.array([[3.0, 3.0], [6.0, 6.0], [1.0, 1.0]])
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
            with fluid.dygraph.guard():
                gpu_out = paddle.scatter(
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
                x_t = paddle.static.data(name="x", dtype=x.dtype, shape=x.shape)
                index_t = paddle.static.data(
                    name="index", dtype=index.dtype, shape=index.shape
                )
                updates_t = paddle.static.data(
                    name="updates", dtype=updates.dtype, shape=updates.shape
                )
                out_t = paddle.scatter(x_t, index_t, updates_t)
                feed = {
                    x_t.name: x,
                    index_t.name: index,
                    updates_t.name: updates,
                }
                fetch = [out_t]

                gpu_exe = paddle.static.Executor(paddle.CUDAPlace(0))
                gpu_value = gpu_exe.run(feed=feed, fetch_list=fetch)[0]
                return gpu_value

        np.testing.assert_array_equal(test_dygraph(), test_static_graph())


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestScatterOpFp16(OpTest):
    def setUp(self):
        self.__class__.op_type = "scatter"
        self.python_api = paddle.scatter
        # compute grad in the following code handly.
        self.__class__.no_need_check_grad = True
        self.x_type = 'float16'
        self.x_np = np.ones((3, 3)).astype(self.x_type)
        self.index_np = np.array([1, 2]).astype("int32")
        self.updates_np = np.random.random((2, 3)).astype(self.x_type)
        self.output_np = np.copy(self.x_np)
        self.output_np[self.index_np] = self.updates_np
        self.dout_np = np.random.random((3, 3)).astype(self.x_type)

        # compute ref_dx
        self.ref_dx = np.copy(self.dout_np)
        zero_np = np.zeros((2, 3)).astype(self.x_type)
        self.ref_dx[self.index_np] = zero_np

    def compute_ref_grad_updates(self):
        ref_grad_updates = paddle.gather(
            paddle.to_tensor(self.dout_np), paddle.to_tensor(self.index_np)
        )
        return ref_grad_updates

    def test_scatter_fp16(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        x_tensor = paddle.to_tensor(self.x_np, stop_gradient=False)
        index_tensor = paddle.to_tensor(self.index_np)
        updates_tensor = paddle.to_tensor(self.updates_np, stop_gradient=False)
        out_tensor = paddle.scatter(x_tensor, index_tensor, updates_tensor)
        paddle.autograd.backward(
            [out_tensor], [paddle.to_tensor(self.dout_np)], retain_graph=True
        )
        ref_grad_updates = self.compute_ref_grad_updates()
        np.testing.assert_allclose(
            ref_grad_updates.numpy(),
            updates_tensor.grad.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            self.ref_dx, x_tensor.grad.numpy(), rtol=1e-5, atol=1e-5
        )


class TestScatterInplaceAPI(TestScatterAPI):
    def executed_api(self):
        self.scatter = paddle.scatter_


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
