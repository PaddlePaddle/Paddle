#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.base import core

SEED = 100
np.random.seed(SEED)
paddle.seed(SEED)


def output_log_normal(shape, mean, std):
    return np.exp(np.random.normal(mean, std, shape))


class TestLogNormalAPI(unittest.TestCase):
    DTYPE = "float64"
    SHAPE = [2, 4]
    MEAN = 0
    STD = 1

    def setUp(self):
        self.x = output_log_normal(self.SHAPE, self.MEAN, self.STD)
        self.place = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def test_api_static(self):
        def run(place):
            paddle.enable_static()
            with paddle.static.program_guard(paddle.static.Program()):
                out = paddle.log_normal()
                exe = paddle.static.Executor(place)
                res = exe.run(
                    paddle.static.default_main_program(),
                    feed={},
                    fetch_list=[out],
                )
                return res[0]

        for place in self.place:
            res = run(place)
            self.assertTrue(np.allclose(res, self.x))

    def test_api_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            out = paddle.log_normal(self.SHAPE, self.MEAN, self.STD, seed=SEED)

            out_ref = output_log_normal(self.SHAPE, self.MEAN, self.STD)
            np.testing.assert_allclose(out.numpy(), out_ref, rtol=1e-5)
            paddle.enable_static()

        for place in self.place:
            run(place)


if __name__ == "__main__":
    unittest.main()
