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

import os
import unittest

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle import nn, static
from paddle.base import Executor, core


class TestCosineSimilarityAPI(unittest.TestCase):
    def setUp(self):
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def _get_numpy_out(self, x1, x2, axis=1, eps=1e-8):
        w12 = np.sum(x1 * x2, axis=axis)
        w1 = np.sum(x1 * x1, axis=axis)
        w2 = np.sum(x2 * x2, axis=axis)
        n12 = np.sqrt(np.clip(w1 * w2, eps * eps, None))
        cos_sim = w12 / n12
        return cos_sim

    def check_static_result(self, place):
        paddle.enable_static()

        main_program = static.Program()
        startup_program = static.Program()
        with static.program_guard(
            main_program=main_program, startup_program=startup_program
        ):
            shape = [10, 15]
            axis = 1
            eps = 1e-8
            np.random.seed(0)
            np_x1 = np.random.rand(*shape).astype(np.float32)
            np_x2 = np.random.rand(*shape).astype(np.float32)

            x1 = paddle.static.data(name="x1", shape=shape)
            x2 = paddle.static.data(name="x2", shape=shape)
            result = F.cosine_similarity(x1, x2, axis=axis, eps=eps)
            exe = Executor(place)
            fetches = exe.run(
                feed={"x1": np_x1, "x2": np_x2},
                fetch_list=[result],
            )

            np_out = self._get_numpy_out(np_x1, np_x2, axis=axis, eps=eps)
            np.testing.assert_allclose(fetches[0], np_out, rtol=1e-05)

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph_1(self):
        paddle.disable_static()

        shape = [10, 15]
        axis = 1
        eps = 1e-8
        np.random.seed(1)
        np_x1 = np.random.rand(*shape).astype(np.float32)
        np_x2 = np.random.rand(*shape).astype(np.float32)
        np_out = self._get_numpy_out(np_x1, np_x2, axis=axis, eps=eps)

        tensor_x1 = paddle.to_tensor(np_x1)
        tensor_x2 = paddle.to_tensor(np_x2)
        y = F.cosine_similarity(tensor_x1, tensor_x2, axis=axis, eps=eps)

        np.testing.assert_allclose(y.numpy(), np_out, rtol=1e-05)

    def test_dygraph_2(self):
        paddle.disable_static()

        shape = [12, 13]
        axis = 0
        eps = 1e-6
        np.random.seed(1)
        np_x1 = np.random.rand(*shape).astype(np.float32)
        np_x2 = np.random.rand(*shape).astype(np.float32)
        np_out = self._get_numpy_out(np_x1, np_x2, axis=axis, eps=eps)

        tensor_x1 = paddle.to_tensor(np_x1)
        tensor_x2 = paddle.to_tensor(np_x2)
        y = F.cosine_similarity(tensor_x1, tensor_x2, axis=axis, eps=eps)

        np.testing.assert_allclose(y.numpy(), np_out, rtol=1e-05)

    def test_dygraph_3(self):
        paddle.disable_static()

        shape1 = [10, 12, 10]
        shape2 = [10, 1, 10]
        axis = 2
        eps = 1e-6
        np.random.seed(1)
        np_x1 = np.random.rand(*shape1).astype(np.float32)
        np_x2 = np.random.rand(*shape2).astype(np.float32)
        np_out = self._get_numpy_out(np_x1, np_x2, axis=axis, eps=eps)

        tensor_x1 = paddle.to_tensor(np_x1)
        tensor_x2 = paddle.to_tensor(np_x2)
        y = F.cosine_similarity(tensor_x1, tensor_x2, axis=axis, eps=eps)

        np.testing.assert_allclose(y.numpy(), np_out, rtol=1e-05)

    def test_dygraph_4(self):
        paddle.disable_static()

        shape1 = [23, 12, 1]
        shape2 = [23, 1, 10]
        axis = 2
        eps = 1e-6
        np.random.seed(1)
        np_x1 = np.random.rand(*shape1).astype(np.float32)
        np_x2 = np.random.rand(*shape2).astype(np.float32)
        np_out = self._get_numpy_out(np_x1, np_x2, axis=axis, eps=eps)

        cos_sim_func = nn.CosineSimilarity(axis=axis, eps=eps)
        tensor_x1 = paddle.to_tensor(np_x1)
        tensor_x2 = paddle.to_tensor(np_x2)
        y = cos_sim_func(tensor_x1, tensor_x2)

        np.testing.assert_allclose(y.numpy(), np_out, rtol=1e-05)


if __name__ == '__main__':
    unittest.main()
