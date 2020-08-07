# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid as fluid
import numpy as np
import unittest


def pairwise_distance(x, y, ord=2.0, eps=1e-6, keepdim=False):
    return np.linalg.norm(x - y, ord=ord, axis=1, keepdims=keepdim)


class TestPairwiseDistance(unittest.TestCase):
    def test_pairwise_distance(self):
        all_shape = [[100, 100], [4, 5, 6, 7]]
        dtypes = ['float32', 'float64']
        keeps = [False, True]
        for shape in all_shape:
            for dtype in dtypes:
                for keepdim in keeps:
                    x_np = np.random.random(shape).astype(dtype)
                    y_np = np.random.random(shape).astype(dtype)

                    prog = paddle.Program()
                    startup_prog = paddle.Program()

                    place = paddle.CUDAPlace(
                        0) if paddle.fluid.core.is_compiled_with_cuda(
                        ) else fluid.CPUPlace()

                    with paddle.program_guard(prog, startup_prog):
                        x = paddle.data(name='x', shape=shape, dtype=dtype)
                        y = paddle.data(name='y', shape=shape, dtype=dtype)
                        dist = paddle.nn.layer.distance.PairwiseDistance(
                            keepdim=keepdim)
                        distance = dist(x, y)
                        exe = fluid.Executor(place)
                        static_ret = exe.run(prog,
                                             feed={'x': x_np,
                                                   'y': y_np},
                                             fetch_list=[distance])
                        self.assertIsNotNone(static_ret)
                        static_ret = static_ret[0]

                    paddle.enable_imperative()
                    x = paddle.imperative.to_variable(x_np)
                    y = paddle.imperative.to_variable(y_np)
                    dist = paddle.nn.layer.distance.PairwiseDistance(
                        keepdim=keepdim)
                    distance = dist(x, y)
                    dygraph_ret = distance.numpy()
                    paddle.disable_imperative()

                    excepted_value = pairwise_distance(
                        x_np, y_np, keepdim=keepdim)

                    self.assertTrue(np.allclose(static_ret, dygraph_ret))
                    self.assertTrue(np.allclose(static_ret, excepted_value))
                    self.assertTrue(np.allclose(dygraph_ret, excepted_value))


if __name__ == "__main__":
    unittest.main()
