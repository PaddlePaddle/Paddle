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

import unittest

import numpy as np
from test_pool2d_op import (
    lp_pool2D_forward_naive,
)

import paddle
from paddle import base
from paddle.base import core
from paddle.nn.functional import lp_pool2d
from paddle.pir_utils import test_with_pir_api


class TestPool2D_API(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def check_lp_static_results(self, place):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            input = paddle.static.data(
                name="input", shape=[2, 3, 32, 32], dtype="float32"
            )
            norm_type = 2
            result = lp_pool2d(
                input,
                norm_type,
                kernel_size=2,
                stride=2,
                ceil_mode=True,
            )

            input_np = np.random.random([2, 3, 32, 32]).astype("float32")
            result_np = lp_pool2D_forward_naive(
                input_np,
                norm_type,
                ksize=[2, 2],
                strides=[2, 2],
                ceil_mode=True,
            )

            exe = base.Executor(place)
            fetches = exe.run(
                feed={"input": input_np},
                fetch_list=[result],
            )
            np.testing.assert_allclose(fetches[0], result_np, rtol=1e-05)

    @test_with_pir_api
    def test_pool2d_static_cpu(self):
        paddle.enable_static()
        self.check_lp_static_results(self.places[0])
        paddle.disable_static()

    @test_with_pir_api
    def test_pool2d_static_gpu(self):
        paddle.enable_static()
        self.check_lp_static_results(self.places[1])
        paddle.disable_static()


if __name__ == '__main__':
    unittest.main()
