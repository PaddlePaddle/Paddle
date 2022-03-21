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

from __future__ import print_function

import unittest
import numpy as np
import copy
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from paddle.framework import core
from paddle.fluid.dygraph.base import switch_to_static_graph

paddle.enable_static()


class TestPutAlongAxisOp(OpTest):
    def setUp(self):
        self.init_data()
        self.reduce_op = "assign"
        self.dtype = 'float64'
        self.op_type = "put_along_axis"
        self.xnp = np.random.random(self.x_shape).astype(self.x_type)
        # numpy put_along_axis is an inplace opearion.
        self.xnp_result = copy.deepcopy(self.xnp)
        np.put_along_axis(self.xnp_result, self.index, self.value, self.axis)
        self.target = self.xnp_result
        broadcast_shape_list = list(self.x_shape)
        broadcast_shape_list[self.axis] = 1
        self.braodcast_shape = tuple(broadcast_shape_list)
        self.index_broadcast = np.broadcast_to(self.index, self.braodcast_shape)
        self.value_broadcast = np.broadcast_to(self.value, self.braodcast_shape)
        self.inputs = {
            'Input': self.xnp,
            'Index': self.index_broadcast,
            'Value': self.value_broadcast
        }
        self.attrs = {'Axis': self.axis, 'Reduce': self.reduce_op}
        self.outputs = {'Result': self.target}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["Input", "Value"], "Result")

    def init_data(self):
        self.x_type = "float64"
        self.x_shape = (10, 10, 10)
        self.value_type = "float64"
        self.value = np.array([99]).astype(self.value_type)
        self.index_type = "int32"
        self.index = np.array([[[0]]]).astype(self.index_type)
        self.axis = 1
        self.axis_type = "int64"


class TestPutAlongAxisAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.shape = [1, 3]
        self.index_shape = [1, 1]
        self.index_np = np.array([[0]]).astype('int64')
        self.x_np = np.random.random(self.shape).astype(np.float32)
        self.place = [paddle.CPUPlace()]
        self.axis = 0
        self.value_np = 99.0
        self.value_shape = [1]
        self.x_feed = copy.deepcopy(self.x_np)
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def test_api_static(self):
        paddle.enable_static()

        def run(place):
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.fluid.data('X', self.shape)
                index = paddle.fluid.data('Index', self.index_shape, "int64")
                value = paddle.fluid.data('Value', self.value_shape)
                out = paddle.put_along_axis(x, index, value, self.axis)
                exe = paddle.static.Executor(self.place[0])
                res = exe.run(feed={
                    'X': self.x_feed,
                    'Value': self.value_np,
                    'Index': self.index_np
                },
                              fetch_list=[out])

            np.put_along_axis(self.x_np, self.index_np, self.value_np,
                              self.axis)
            # numpy put_along_axis is an inplace opearion.
            out_ref = self.x_np

            for out in res:
                self.assertEqual(np.allclose(out, out_ref, rtol=1e-03), True)

        for place in self.place:
            run(place)

    def test_api_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            x_tensor = paddle.to_tensor(self.x_np)
            index_tensor = paddle.to_tensor(self.index_np)
            value_tensor = paddle.to_tensor(self.value_np)
            out = paddle.put_along_axis(x_tensor, index_tensor, value_tensor,
                                        self.axis)
            np.array(
                np.put_along_axis(self.x_np, self.index_np, self.value_np,
                                  self.axis))
            out_ref = self.x_np
            self.assertEqual(
                np.allclose(
                    out.numpy(), out_ref, rtol=1e-03), True)

            # for ci coverage, numpy put_along_axis did not support argument of 'reduce'
            paddle.put_along_axis(x_tensor, index_tensor, value_tensor,
                                  self.axis, 'mul')
            paddle.put_along_axis(x_tensor, index_tensor, value_tensor,
                                  self.axis, 'add')

            paddle.enable_static()

        for place in self.place:
            run(place)

    def test_inplace_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            x_tensor = paddle.to_tensor(self.x_np)
            index_tensor = paddle.to_tensor(self.index_np)
            value_tensor = paddle.to_tensor(self.value_np)

            x_tensor.put_along_axis_(index_tensor, value_tensor, self.axis)

            np.array(
                np.put_along_axis(self.x_np, self.index_np, self.value_np,
                                  self.axis))
            out_ref = self.x_np

            self.assertEqual(
                np.allclose(
                    x_tensor.numpy(), out_ref, rtol=1e-03), True)
            paddle.enable_static()

        for place in self.place:
            run(place)


class TestPutAlongAxisAPICase2(TestPutAlongAxisAPI):
    def setUp(self):
        np.random.seed(0)
        self.shape = [2, 2]
        self.index_shape = [2, 2]
        self.index_np = np.array([[0, 0], [1, 0]]).astype('int64')
        self.x_np = np.random.random(self.shape).astype(np.float32)
        self.place = [paddle.CPUPlace()]
        self.axis = 0
        self.value_np = 99.0
        self.value_shape = [1]
        self.x_feed = copy.deepcopy(self.x_np)
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))


class TestPutAlongAxisAPICase3(TestPutAlongAxisAPI):
    def setUp(self):
        np.random.seed(0)
        self.shape = [2, 2]
        self.index_shape = [4, 2]
        self.index_np = np.array(
            [[0, 0], [1, 0], [0, 0], [1, 0]]).astype('int64')
        self.x_np = np.random.random(self.shape).astype(np.float32)
        self.place = [paddle.CPUPlace()]
        self.axis = 0
        self.value_np = 99.0
        self.value_shape = [1]
        self.x_feed = copy.deepcopy(self.x_np)
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def test_inplace_dygraph(self):
        pass


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
