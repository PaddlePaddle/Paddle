#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import unittest

import numpy as np
from op_test import get_places
from utils import dygraph_guard

import paddle
from paddle.framework import core
from paddle.static import InputSpec


def scatter_add_net(x, axis=-1):
    index = paddle.full_like(x, fill_value=2, dtype='int64')
    value = paddle.full_like(x, fill_value=-4.0, dtype=x.dtype)
    return paddle.scatter_add(x, axis, index, value)


class TestScatterAddAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.shape = [10, 10]
        self.index_shape = [10, 10]
        self.index_np = np.random.randint(0, 10, (10, 10)).astype('int64')
        self.x_np = np.random.random(self.shape).astype(np.float32)
        self.place = get_places()
        self.axis = 0
        self.value_np = np.random.randint(0, 10, (10, 10)).astype(np.float32)
        self.value_shape = [10, 10]
        self.x_feed = copy.deepcopy(self.x_np)

    def test_api_static(self):
        paddle.enable_static()

        def run(place):
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', self.shape)
                index = paddle.static.data('Index', self.index_shape, "int64")
                value = paddle.static.data('Value', self.value_shape)
                out = paddle.scatter_add(x, self.axis, index, value)
                exe = paddle.static.Executor(self.place[0])
                res = exe.run(
                    feed={
                        'X': self.x_feed,
                        'Value': self.value_np,
                        'Index': self.index_np,
                    },
                    fetch_list=[out],
                )
            target = copy.deepcopy(self.x_np)

            for i in range(10):
                for j in range(10):
                    target[self.index_np[i, j], j] += self.value_np[i, j]
            # numpy put_along_axis is an inplace operation.
            out_ref = target

            for out in res:
                np.testing.assert_allclose(out, out_ref, rtol=0.001)

        for place in self.place:
            run(place)

    def test_api_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            x_tensor = paddle.to_tensor(self.x_np)
            index_tensor = paddle.to_tensor(self.index_np)
            value_tensor = paddle.to_tensor(self.value_np)
            out = paddle.scatter_add(
                x_tensor, self.axis, index_tensor, value_tensor
            )

            target = copy.deepcopy(self.x_np)
            for i in range(10):
                for j in range(10):
                    target[self.index_np[i, j], j] += self.value_np[i, j]

            out_ref = target
            np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.001)

            paddle.enable_static()

        for place in self.place:
            run(place)


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "core is not compiled with CUDA",
)
class TestScatterAddAPILargeCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.shape = [64, 102400]
        self.index_shape = [64, 102400]
        self.index_np = np.zeros(self.index_shape).astype('int64')
        self.x_np = np.random.random(self.shape).astype(np.float32)
        self.axis = 1
        self.value_np = np.ones(self.index_shape).astype(np.float32)
        self.x_feed = copy.deepcopy(self.x_np)
        self.place = [paddle.CUDAPlace(0)]

    def test_api_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            x_tensor = paddle.to_tensor(self.x_np)
            index_tensor = paddle.to_tensor(self.index_np)
            value_tensor = paddle.to_tensor(self.value_np)
            out = paddle.scatter_add(
                x_tensor, self.axis, index_tensor, value_tensor
            )

            for i in range(64):
                for j in range(102400):
                    self.x_np[i, self.index_np[i, j]] += self.value_np[i, j]
            out_ref = self.x_np
            np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.001)

            paddle.enable_static()

        for place in self.place:
            run(place)


class TestScatterAddAPIOtherCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.shape = [3, 5]
        self.index1_shape = [1, 4]
        self.index_np1 = np.array([[0, 1, 2, 0]]).astype('int64')
        self.index2_shape = [2, 3]
        self.index_np2 = np.array([[0, 1, 2], [0, 1, 4]]).astype('int64')
        self.x_np = np.zeros((3, 5)).astype(np.float32)
        self.value_shape = [2, 5]
        self.value = (
            np.arange(1, 11).reshape(self.value_shape).astype(np.float32)
        )
        self.place = get_places()

    def test_api_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            x_tensor = paddle.to_tensor(self.x_np)
            index_tensor1 = paddle.to_tensor(self.index_np1)
            value_tensor = paddle.to_tensor(self.value)
            out = paddle.scatter_add(x_tensor, 0, index_tensor1, value_tensor)
            out_ref = copy.deepcopy(self.x_np)
            for i in range(self.index1_shape[0]):
                for j in range(self.index1_shape[1]):
                    out_ref[self.index_np1[i, j], j] += self.value[i, j]
            np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.001)

            index_tensor2 = paddle.to_tensor(self.index_np2)
            out = paddle.scatter_add(x_tensor, 1, index_tensor2, value_tensor)
            out_ref = copy.deepcopy(self.x_np)
            for i in range(self.index2_shape[0]):
                for j in range(self.index2_shape[1]):
                    out_ref[i, self.index_np2[i, j]] += self.value[i, j]
            np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.001)

            paddle.enable_static()

        for place in self.place:
            run(place)

    def test_api_static(self):
        paddle.enable_static()

        def run(place):
            with paddle.static.program_guard(paddle.static.Program()):
                x1 = paddle.static.data('X', self.shape)
                index1 = paddle.static.data('Index', self.index1_shape, "int64")
                value_tensor = paddle.to_tensor(self.value)
                out1 = paddle.scatter_add(x1, 0, index1, value_tensor)
                exe = paddle.static.Executor(place)
                res = exe.run(
                    feed={
                        'X': self.x_np,
                        'Value': self.value,
                        'Index': self.index_np1,
                    },
                    fetch_list=[out1],
                )
            out_ref = copy.deepcopy(self.x_np)
            for i in range(self.index1_shape[0]):
                for j in range(self.index1_shape[1]):
                    out_ref[self.index_np1[i, j], j] += self.value[i, j]

            for out in res:
                np.testing.assert_allclose(out, out_ref, rtol=0.001)

            with paddle.static.program_guard(paddle.static.Program()):
                x2 = paddle.static.data('X', self.shape)
                index2 = paddle.static.data('Index', self.index2_shape, "int64")
                value_tensor = paddle.to_tensor(self.value)
                out2 = paddle.scatter_add(x2, 1, index2, value_tensor)
                exe = paddle.static.Executor(place)
                res = exe.run(
                    feed={
                        'X': self.x_np,
                        'Value': self.value,
                        'Index': self.index_np2,
                    },
                    fetch_list=[out2],
                )
            out_ref = copy.deepcopy(self.x_np)
            for i in range(self.index2_shape[0]):
                for j in range(self.index2_shape[1]):
                    out_ref[i, self.index_np2[i, j]] += self.value[i, j]

            for out in res:
                np.testing.assert_allclose(out, out_ref, rtol=0.001)

        for place in self.place:
            run(place)

    def test_error(self):
        tensorx = paddle.to_tensor([[1, 2, 3], [4, 5, 6]]).astype("float32")
        indices = paddle.to_tensor([[1, 0, 1], [0, 1, 1]]).astype("int32")
        values = paddle.to_tensor([1])

        try:
            res = paddle.scatter_add(tensorx, 0, indices, values)
        except Exception as error:
            self.assertIsInstance(error, ValueError)

        indices = paddle.to_tensor([1]).astype("int32")
        values = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])

        try:
            res = paddle.scatter_add(tensorx, 0, indices, values)
        except Exception as error:
            self.assertIsInstance(error, ValueError)

        indices = paddle.to_tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        ).astype("int32")
        # indices too large
        try:
            res = paddle.scatter_add(tensorx, 0, indices, values)
        except Exception as error:
            self.assertIsInstance(error, RuntimeError)

        indices = paddle.to_tensor([[3, 0, 4], [0, 5, 10]]).astype("int32")
        # the element of indices out of range
        try:
            res = paddle.scatter_add(tensorx, 0, indices, values)
        except Exception as error:
            self.assertIsInstance(error, RuntimeError)

    def test_index_type_error(self):
        tensorx = paddle.to_tensor([[1, 2, 3], [4, 5, 6]]).astype("float32")
        indices = paddle.to_tensor([[1, 0, 1], [0, 1, 1]]).astype("float32")
        values = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(TypeError):
            res = paddle.scatter_add(tensorx, 0, indices, values)


class TestScatterAddAPIDynamicShape(unittest.TestCase):
    def setUp(self):
        np.random.seed(2024)
        self.net = scatter_add_net
        self.enable_cinn = False
        self.tol = 1e-6
        self.dtype = "float32"
        self.axis = -2
        self.input_specs = [
            InputSpec(
                shape=(-1, -1, -1, -1),
                dtype=self.dtype,
                stop_gradient=False,
            )
        ]
        self.arr = np.random.random([10, 10, 10, 10]).astype(self.dtype)

    def train(self, to_static):
        arr = paddle.to_tensor(self.arr, stop_gradient=False)
        if to_static:
            backend = "CINN" if self.enable_cinn else None
            net = paddle.jit.to_static(
                self.net,
                input_spec=self.input_specs,
                backend=backend,
                full_graph=True,
            )
            net.train()
        else:
            net = self.net

        res = net(arr, self.axis)
        res.backward()
        arr_grad = arr.grad
        return res, arr_grad

    def test_dynamic_static(self):
        with dygraph_guard():
            st_out, st_grads = self.train(to_static=True)
            dy_out, dy_grads = self.train(to_static=False)

            for ref, actual in zip(dy_out, st_out):
                np.testing.assert_allclose(
                    ref, actual, rtol=self.tol, atol=self.tol
                )

            for dr, d in zip(dy_grads, st_grads):
                np.testing.assert_allclose(dr, d, rtol=self.tol, atol=self.tol)


class TestScatterAddAPIDynamicShape1(TestScatterAddAPIDynamicShape):
    def setUp(self):
        np.random.seed(2024)
        self.net = scatter_add_net
        self.enable_cinn = False
        self.tol = 1e-6
        self.dtype = "float32"
        self.axis = 0
        self.input_specs = [
            InputSpec(
                shape=(-1, -1, -1, -1),
                dtype=self.dtype,
                stop_gradient=False,
            )
        ]
        self.arr = np.random.random([16, 16, 16, 16]).astype(self.dtype)


class TestScatterAddAPIDynamicShape2(TestScatterAddAPIDynamicShape):
    def setUp(self):
        np.random.seed(2024)
        self.net = scatter_add_net
        self.enable_cinn = False
        self.tol = 1e-6
        self.dtype = "float32"
        self.axis = -1
        self.input_specs = [
            InputSpec(
                shape=(-1, -1, -1, -1),
                dtype=self.dtype,
                stop_gradient=False,
            )
        ]
        self.arr = np.random.random([20, 20, 20, 20]).astype(self.dtype)


class TestScatterAddAPIDynamicShape3(TestScatterAddAPIDynamicShape):
    def setUp(self):
        np.random.seed(2024)
        self.net = scatter_add_net
        self.enable_cinn = False
        self.tol = 1e-6
        self.dtype = "float32"
        self.axis = 3
        self.input_specs = [
            InputSpec(
                shape=(-1, -1, -1, -1),
                dtype=self.dtype,
                stop_gradient=False,
            )
        ]
        self.arr = np.random.random([32, 32, 32, 32]).astype(self.dtype)


class TestScatterAddAPIDynamicShape_ZeroSize(TestScatterAddAPIDynamicShape):
    def setUp(self):
        np.random.seed(2024)
        self.net = scatter_add_net
        self.enable_cinn = False
        self.tol = 1e-6
        self.dtype = "float32"
        self.axis = -2
        self.input_specs = [
            InputSpec(
                shape=(-1, -1, -1, -1),
                dtype=self.dtype,
                stop_gradient=False,
            )
        ]
        self.arr = np.random.random([0, 10, 10, 10]).astype(self.dtype)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
